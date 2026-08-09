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

#ifndef LLVM_LIBC_SRC_NET_LINUX_IF_NAMEINDEX_IMPL_H
#define LLVM_LIBC_SRC_NET_LINUX_IF_NAMEINDEX_IMPL_H

#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/alloc-checker.h"
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

/// A reasonable buffer size for netlink messages (see NLMSG_GOODSIZE in the
/// kernel).
constexpr size_t NLMSG_BUFFER_SIZE = 8192;
} // namespace detail

template <typename Policy>
LIBC_INLINE ErrorOr<struct if_nameindex *> if_nameindex() {
  ErrorOr<int> fd_or_err =
      Policy::socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE);
  if (!fd_or_err.has_value())
    return Error(fd_or_err.error());
  int fd = *fd_or_err;
  cpp::scope_exit close_fd([fd]() { Policy::close(fd); });

  ErrorOr<ssize_t> send_res = detail::send_netlink_dump_request<Policy>(fd);
  if (!send_res.has_value())
    return Error(send_res.error());

  // TODO: Figure out if we need to dynamically allocate a buffer.
  alignas(struct nlmsghdr) uint8_t buf[detail::NLMSG_BUFFER_SIZE];
  ErrorOr<ssize_t> recv_res =
      Policy::recvfrom(fd, buf, sizeof(buf), 0, nullptr, nullptr);
  if (!recv_res.has_value())
    return Error(recv_res.error());

  close_fd.release();
  if (ErrorOr<int> close_res = Policy::close(fd); !close_res.has_value())
    return Error(close_res.error());

  // TODO: Read more than one message.
  // TODO: Read more than one interface per message.
  // TODO: Deduplicate interfaces to handle restarts.
  auto len = static_cast<size_t>(*recv_res);
  for (auto *nh = reinterpret_cast<struct nlmsghdr *>(buf); NLMSG_OK(nh, len);
       nh = NLMSG_NEXT(nh, len)) {
    if (nh->nlmsg_type == NLMSG_DONE)
      break;
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
      auto index = static_cast<unsigned int>(ifm->ifi_index);
      const char *name_data = reinterpret_cast<const char *>(RTA_DATA(rta));
      size_t name_len = internal::strnlen(name_data, rta_payload_len);

      size_t total_size = 2 * sizeof(struct if_nameindex) + name_len + 1;
      AllocChecker ac;
      uint8_t *buffer = new (ac) uint8_t[total_size];
      if (!ac)
        return Error(ENOBUFS);

      cpp::span<uint8_t> buffer_span(buffer, total_size);
      cpp::span<struct if_nameindex> result(
          reinterpret_cast<struct if_nameindex *>(buffer_span.data()), 2);
      cpp::span<char> string_span(reinterpret_cast<char *>(result.end()),
                                  reinterpret_cast<char *>(buffer_span.end()));

      result[0].if_index = index;
      result[0].if_name = string_span.data();
      inline_memcpy(string_span.data(), name_data, name_len);
      string_span[name_len] = '\0';

      result[1].if_index = 0;
      result[1].if_name = nullptr;

      return result.data();
    }
  }

  AllocChecker ac;
  uint8_t *buffer = new (ac) uint8_t[sizeof(struct if_nameindex)];
  if (!ac)
    return Error(ENOBUFS);
  cpp::span<struct if_nameindex> result(
      reinterpret_cast<struct if_nameindex *>(buffer), 1);
  result[0] = {};
  return result.data();
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NET_LINUX_IF_NAMEINDEX_IMPL_H
