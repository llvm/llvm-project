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
#include "hdr/types/struct_if_nameindex.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/blockstore.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/net/linux/netlink.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include <linux/if_link.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

namespace LIBC_NAMESPACE_DECL {
namespace net {

namespace detail {
struct InterfaceEntry {
  unsigned int index;
  uint8_t name_length;        // Length of next field.
  char name[IF_NAMESIZE - 1]; // No null terminator.
  static_assert(IF_NAMESIZE - 1 < (1u << 8));
};

/// Adds the interface described by an RTM_NEWLINK message to the store.
/// Messages which are too short, or which don't have an IFLA_IFNAME attribute,
/// are ignored. Returns ENOBUFS if the store cannot grow.
LIBC_INLINE ErrorOr<void> add_interface(struct nlmsghdr &nh,
                                        BlockStore<InterfaceEntry, 16> &store) {
  auto *ifm = get_nlmsg_header<struct ifinfomsg>(nh);
  if (!ifm)
    return {};

  cpp::optional<cpp::string_view> name;
  for_each_rtattr(IFLA_RTA(ifm), IFLA_PAYLOAD(&nh),
                  [&](unsigned short type, cpp::span<const uint8_t> payload) {
                    if (type == IFLA_IFNAME && !name)
                      name = cpp::optional<cpp::string_view>(
                          parse_ifname(payload));
                  });
  if (!name)
    return {};

  InterfaceEntry entry;
  entry.index = static_cast<unsigned int>(ifm->ifi_index);
  // parse_ifname truncates the name to IF_NAMESIZE - 1 characters, so it fits.
  entry.name_length = static_cast<uint8_t>(name->size());
  inline_memcpy(entry.name, name->data(), name->size());
  if (!store.push_back(entry))
    return Error(ENOBUFS);
  return {};
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

} // namespace detail

template <typename Policy>
LIBC_INLINE ErrorOr<struct if_nameindex *> if_nameindex() {
  NetlinkSocket<Policy> sock;
  if (ErrorOr<void> res = sock.open(); !res.has_value())
    return Error(res.error());

  // AF_UNSPEC means we want all interfaces.
  struct ifinfomsg request = {};
  request.ifi_family = AF_UNSPEC;

  // If the interface list changes while the kernel is sending it to us, the
  // result may be inconsistent: it can contain stale or duplicate entries, or
  // miss some interfaces altogether. In that case, we throw it away and try
  // again.
  for (size_t attempt = 0; attempt < MAX_DUMP_ATTEMPTS; ++attempt) {
    BlockStore<detail::InterfaceEntry, 16> store;
    cpp::scope_exit destroy_store([&store]() {
      BlockStore<detail::InterfaceEntry, 16>::destroy(&store);
    });

    ErrorOr<DumpResult> dump_res = sock.dump(
        RTM_GETLINK, request, RTM_NEWLINK, [&store](struct nlmsghdr &nh) {
          return detail::add_interface(nh, store);
        });
    if (!dump_res.has_value())
      return Error(dump_res.error());
    if (*dump_res == DumpResult::Interrupted)
      continue;

    if (ErrorOr<void> close_res = sock.close(); !close_res.has_value())
      return Error(close_res.error());
    return detail::build_if_nameindex_list(store);
  }
  return Error(EAGAIN);
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_NET_IF_LINUX_IF_NAMEINDEX_IMPL_H
