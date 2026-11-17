//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal templatized implementation of if_nameindex for Linux.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_NET_IF_LINUX_IF_NAMEINDEX_IMPL_H
#define LLVM_LIBC_SRC_NET_IF_LINUX_IF_NAMEINDEX_IMPL_H

#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/blockstore.h"
#include "src/__support/checked_ptr.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/string_utils.h"

#include <linux/if_link.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

namespace LIBC_NAMESPACE_DECL {
namespace net {

namespace detail {
/// Sends a RTM_GETLINK dump request packet on the given socket. The kernel will
/// respond with one or more messages containing the list of network interfaces
/// and their attributes.
template <typename Policy>
LIBC_INLINE ErrorOr<ssize_t> send_netlink_dump_request(int sockfd) {
  // The message consists of a standard netlink header followed by the
  // RTM_GETLINK payload.
  struct {
    struct nlmsghdr nlh;
    struct ifinfomsg ifm;
  } req = {};
  static_assert(sizeof(req) >= NLMSG_LENGTH(sizeof(req.ifm)));
  req.nlh.nlmsg_len = NLMSG_LENGTH(sizeof(req.ifm));
  req.nlh.nlmsg_type = RTM_GETLINK;
  req.nlh.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
  req.ifm.ifi_family = AF_UNSPEC;

  return Policy::sendto(sockfd, &req, req.nlh.nlmsg_len, 0, nullptr, 0);
}

/// Size of the initial receive buffer. As long as we don't request any
/// extended information (IFLA_EXT_MASK), the kernel limits the size of dump
/// datagrams to the larger of NLMSG_GOODSIZE (which is at most 8K) and the
/// largest buffer we have passed to recvfrom, so they always fit.
constexpr size_t NLMSG_BUFFER_SIZE = 8192;

/// Size of the buffer we switch to if the dump gets interrupted. A larger
/// buffer means fewer datagrams, and so less time for the interface list to
/// change while we are reading it. The kernel caps the size of dump datagrams
/// at (a bit less than) 32K, so there is no point in going any higher.
constexpr size_t NLMSG_LARGE_BUFFER_SIZE = 32768;

/// How many times we try to get a consistent dump before giving up.
constexpr size_t MAX_DUMP_ATTEMPTS = 3;

struct InterfaceEntry {
  unsigned int index;
  uint8_t name_length;        // Length of next field.
  char name[IF_NAMESIZE - 1]; // No null terminator.
  static_assert(IF_NAMESIZE - 1 < (1u << 8));
};

/// Parses the netlink messages in the buffer and adds the interfaces they
/// describe to the store. Returns true if the buffer contained the NLMSG_DONE
/// message terminating the dump, or false if more messages should follow.
/// Returns an error if the kernel reported one or if the data is malformed.
/// Sets `interrupted` if the kernel indicated that the interface list changed
/// during the dump.
LIBC_INLINE ErrorOr<bool>
parse_netlink_messages(cpp::span<uint8_t> buf,
                       BlockStore<InterfaceEntry, 16> &store,
                       bool &interrupted) {
  size_t len = buf.size();
  for (auto *nh = reinterpret_cast<struct nlmsghdr *>(buf.data());
       NLMSG_OK(nh, len); nh = NLMSG_NEXT(nh, len)) {
    if (nh->nlmsg_flags & NLM_F_DUMP_INTR)
      interrupted = true;
    if (nh->nlmsg_type == NLMSG_DONE) {
      // The payload contains the result of the dump. This is how the kernel
      // reports errors which happen in the middle of the dump.
      if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(int)))
        return Error(EIO);
      int error = *reinterpret_cast<int *>(NLMSG_DATA(nh));
      if (error < 0)
        return Error(-error);
      return true;
    }
    if (nh->nlmsg_type == NLMSG_ERROR) {
      if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(struct nlmsgerr)))
        return Error(EINVAL);
      auto *err = reinterpret_cast<struct nlmsgerr *>(NLMSG_DATA(nh));
      if (err->error == 0) {
        // Zero means an ACK, which we shouldn't get because we didn't ask for
        // it...
        continue;
      }
      return Error(-err->error);
    }
    if (nh->nlmsg_type != RTM_NEWLINK)
      continue;
    if (nh->nlmsg_len < NLMSG_LENGTH(sizeof(struct ifinfomsg)))
      continue;

    auto *ifm = reinterpret_cast<struct ifinfomsg *>(NLMSG_DATA(nh));
    size_t attrlen = nh->nlmsg_len - NLMSG_LENGTH(sizeof(struct ifinfomsg));
    struct rtattr *rta = IFLA_RTA(ifm);
    for (; RTA_OK(rta, attrlen); rta = RTA_NEXT(rta, attrlen)) {
      if (rta->rta_type != IFLA_IFNAME)
        continue;

      size_t rta_payload_len = RTA_PAYLOAD(rta);
      const char *name_data = reinterpret_cast<const char *>(RTA_DATA(rta));
      size_t name_len = internal::strnlen(name_data, rta_payload_len);
      // Defensive check: kernel should not be providing us with names that
      // don't fit.
      if (name_len >= IF_NAMESIZE)
        name_len = IF_NAMESIZE - 1;

      InterfaceEntry entry;
      entry.index = static_cast<unsigned int>(ifm->ifi_index);
      entry.name_length = static_cast<uint8_t>(name_len);
      inline_memcpy(entry.name, name_data, name_len);

      if (!store.push_back(entry))
        return Error(ENOBUFS);
      break;
    }
  }
  // Anything left over is garbage or a truncated message.
  if (len != 0)
    return Error(EIO);
  return false;
}

LIBC_INLINE ErrorOr<struct if_nameindex *>
build_if_nameindex_list(BlockStore<InterfaceEntry, 16> &store) {
  size_t count = 0;
  size_t strings_size = 0;
  for (const InterfaceEntry &entry : store) {
    ++count;
    strings_size += entry.name_length + 1;
  }

  size_t total_size = (count + 1) * sizeof(struct if_nameindex) + strings_size;
  AllocChecker ac;
  uint8_t *buffer = new (ac) uint8_t[total_size];
  if (!ac)
    return Error(ENOBUFS);

  cpp::span<struct if_nameindex> result(
      reinterpret_cast<struct if_nameindex *>(buffer), count + 1);
  char *str_ptr = reinterpret_cast<char *>(result.end());

  size_t idx = 0;
  for (const InterfaceEntry &entry : store) {
    result[idx].if_index = entry.index;
    result[idx].if_name = str_ptr;
    inline_memcpy(str_ptr, entry.name, entry.name_length);
    str_ptr[entry.name_length] = '\0';
    str_ptr += entry.name_length + 1;
    ++idx;
  }

  result[count].if_index = 0;
  result[count].if_name = nullptr;

  return result.data();
}

/// Receives the response to a dump request and adds the interfaces it describes
/// to the store. Returns true if the dump was interrupted, in which case the
/// result may be inconsistent.
template <typename Policy>
LIBC_INLINE ErrorOr<bool>
receive_netlink_dump(int fd, cpp::span<uint8_t> buf,
                     BlockStore<InterfaceEntry, 16> &store) {
  bool interrupted = false;
  // The dump can come in multiple datagrams. Keep reading until we get the
  // NLMSG_DONE message. We need to do that even if the dump was interrupted, as
  // the kernel won't start a new dump on this socket until this one is done.
  while (true) {
    ErrorOr<ssize_t> recv_res =
        Policy::recvfrom(fd, buf.data(), buf.size(), 0, nullptr, nullptr);
    if (!recv_res.has_value())
      return Error(recv_res.error());
    // A netlink socket should never return an empty datagram. Bail out instead
    // of spinning if it does.
    if (*recv_res == 0)
      return Error(EIO);

    ErrorOr<bool> parse_res = parse_netlink_messages(
        buf.first(static_cast<size_t>(*recv_res)), store, interrupted);
    if (!parse_res.has_value())
      return Error(parse_res.error());
    if (*parse_res)
      return interrupted;
  }
}

} // namespace detail

template <typename Policy>
LIBC_INLINE ErrorOr<struct if_nameindex *> if_nameindex() {
  ErrorOr<int> fd_or_err =
      Policy::socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE);
  if (!fd_or_err.has_value())
    return Error(fd_or_err.error());
  int fd = *fd_or_err;
  cpp::scope_exit close_fd([fd]() { Policy::close(fd); });

  alignas(struct nlmsghdr) uint8_t stack_buf[detail::NLMSG_BUFFER_SIZE];
  cpp::span<uint8_t> buf(stack_buf);
  cpp::unique_ptr<uint8_t[]> heap_buf;

  // If the interface list changes while the kernel is sending it to us, the
  // result may be inconsistent: it can contain stale or duplicate entries, or
  // miss some interfaces altogether. In that case, we throw it away and try
  // again.
  for (size_t attempt = 0; attempt < detail::MAX_DUMP_ATTEMPTS; ++attempt) {
    ErrorOr<ssize_t> send_res = detail::send_netlink_dump_request<Policy>(fd);
    if (!send_res.has_value())
      return Error(send_res.error());

    BlockStore<detail::InterfaceEntry, 16> store;
    cpp::scope_exit destroy_store([&store]() {
      BlockStore<detail::InterfaceEntry, 16>::destroy(&store);
    });

    ErrorOr<bool> interrupted =
        detail::receive_netlink_dump<Policy>(fd, buf, store);
    if (!interrupted.has_value())
      return Error(interrupted.error());

    if (!*interrupted) {
      close_fd.release();
      if (ErrorOr<int> close_res = Policy::close(fd); !close_res.has_value())
        return Error(close_res.error());
      return detail::build_if_nameindex_list(store);
    }

    // Use a larger buffer for the next attempt to reduce the chance of another
    // interruption. If the allocation fails, we keep using the stack buffer.
    // Once we've switched to the larger buffer, we must not switch back, as the
    // kernel may now send datagrams of up to that size.
    if (!heap_buf) {
      CheckedPtr<uint8_t[]> new_buf =
          make_unique_for_overwrite<uint8_t[]>(detail::NLMSG_LARGE_BUFFER_SIZE);
      if (new_buf) {
        buf = {new_buf.get(), detail::NLMSG_LARGE_BUFFER_SIZE};
        heap_buf = new_buf.take_unique();
      }
    }
  }
  return Error(EAGAIN);
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NET_IF_LINUX_IF_NAMEINDEX_IMPL_H
