//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal templatized implementation of getifaddrs for Linux.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_IFADDRS_LINUX_GETIFADDRS_IMPL_H
#define LLVM_LIBC_SRC_IFADDRS_LINUX_GETIFADDRS_IMPL_H

#include "hdr/errno_macros.h"
#include "hdr/net_if_macros.h"
#include "hdr/offsetof_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/sa_family_t.h"
#include "hdr/types/struct_ifaddrs.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/blockstore.h"
#include "src/__support/checked_ptr.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/math_extras.h"
#include "src/__support/net/linux/netlink.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include <linux/if_link.h>
#include <linux/if_packet.h>
#include <linux/netdevice.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

namespace LIBC_NAMESPACE_DECL {
namespace net {

namespace detail {

/// Like struct sockaddr_ll, but with room for the largest hardware address
/// supported by the kernel (MAX_ADDR_LEN = 32), rather than the eight bytes in
/// sockaddr_ll (e.g. InfiniBand addresses have 20).
struct __attribute__((may_alias)) SockAddrLL {
  sa_family_t sll_family;
  uint16_t sll_protocol;
  int sll_ifindex;
  unsigned short sll_hatype;
  unsigned char sll_pkttype;
  unsigned char sll_halen;
  unsigned char sll_addr[MAX_ADDR_LEN];
};
static_assert(offsetof(SockAddrLL, sll_family) ==
              offsetof(struct sockaddr_ll, sll_family));
static_assert(offsetof(SockAddrLL, sll_ifindex) ==
              offsetof(struct sockaddr_ll, sll_ifindex));
static_assert(offsetof(SockAddrLL, sll_hatype) ==
              offsetof(struct sockaddr_ll, sll_hatype));
static_assert(offsetof(SockAddrLL, sll_halen) ==
              offsetof(struct sockaddr_ll, sll_halen));
static_assert(offsetof(SockAddrLL, sll_addr) ==
              offsetof(struct sockaddr_ll, sll_addr));

/// Storage for the addresses we return.
union SockAddrStorage {
  struct sockaddr sa;
  SockAddrLL ll;
};

/// The data of a single element of the list. Addresses with sa_family ==
/// AF_UNSPEC are not present.
struct IfAddrsEntry {
  unsigned int index;
  unsigned int flags;
  SockAddrStorage addr;
  SockAddrStorage netmask;
  SockAddrStorage ifu; // The broadcast or point-to-point destination address.
  char name[IF_NAMESIZE];
};

using IfAddrsStore = BlockStore<IfAddrsEntry, 4>;

/// A single element of the list we return. The pointers in `ifa` point into
/// `entry`. `ifa` must be the first member so `&nodes[0].ifa` points to the
/// start of the uint8_t[] buffer freed by freeifaddrs.
struct IfAddrsNode {
  struct ifaddrs ifa;
  IfAddrsEntry entry;
};
static_assert(offsetof(IfAddrsNode, ifa) == 0);
static_assert(alignof(IfAddrsNode) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);

/// Stores the hardware address from an IFLA_ADDRESS or IFLA_BROADCAST
/// attribute in `dst`, unless it is too long.
LIBC_INLINE void set_link_address(SockAddrStorage &dst,
                                  const struct ifinfomsg &ifm,
                                  cpp::span<const uint8_t> addr) {
  if (addr.size() > sizeof(dst.ll.sll_addr))
    return;
  dst.ll = {};
  dst.ll.sll_family = AF_PACKET;
  dst.ll.sll_ifindex = ifm.ifi_index;
  dst.ll.sll_hatype = ifm.ifi_type;
  dst.ll.sll_halen = static_cast<unsigned char>(addr.size());
  if (!addr.empty())
    inline_memcpy(dst.ll.sll_addr, addr.data(), addr.size());
}

/// Adds an (AF_PACKET) entry for the interface described by an RTM_NEWLINK
/// message.
LIBC_INLINE ErrorOr<void> add_link(struct nlmsghdr &nh, IfAddrsStore &store) {
  auto *ifm = get_nlmsg_header<struct ifinfomsg>(nh);
  if (!ifm)
    return {};

  IfAddrsEntry entry = {};
  entry.index = static_cast<unsigned int>(ifm->ifi_index);
  entry.flags = ifm->ifi_flags;
  bool has_name = false;
  for_each_rtattr(IFLA_RTA(ifm), IFLA_PAYLOAD(&nh),
                  [&](unsigned short type, cpp::span<const uint8_t> payload) {
                    switch (type) {
                    case IFLA_IFNAME: {
                      cpp::string_view name = parse_ifname(payload);
                      inline_memcpy(entry.name, name.data(), name.size());
                      entry.name[name.size()] = '\0';
                      has_name = true;
                      break;
                    }
                    case IFLA_ADDRESS:
                      set_link_address(entry.addr, *ifm, payload);
                      break;
                    case IFLA_BROADCAST:
                      set_link_address(entry.ifu, *ifm, payload);
                      break;
                    }
                  });
  // The kernel always sends the name, but if it didn't, we would have nothing
  // to put into ifa_name.
  if (!has_name)
    return {};

  if (!store.push_back(entry))
    return Error(ENOMEM);
  return {};
}

LIBC_INLINE struct sockaddr *get_sockaddr(SockAddrStorage &addr) {
  return addr.sa.sa_family == AF_UNSPEC ? nullptr : &addr.sa;
}

/// Copies the entries into a single allocation, which freeifaddrs can free
/// with a single delete[].
LIBC_INLINE ErrorOr<struct ifaddrs *> build_ifaddrs_list(IfAddrsStore &store) {
  size_t count = 0;
  for ([[maybe_unused]] const IfAddrsEntry &entry : store)
    ++count;
  if (count == 0)
    return nullptr;

  size_t alloc_size;
  if (mul_overflow(count, sizeof(IfAddrsNode), alloc_size))
    return Error(ENOMEM);

  // Allocate as uint8_t[] rather than IfAddrsNode[] to match the
  // delete[] reinterpret_cast<uint8_t *>(ifa) in freeifaddrs.
  CheckedPtr<uint8_t[]> buffer =
      make_unique_for_overwrite<uint8_t[]>(alloc_size);
  if (!buffer)
    return Error(ENOMEM);

  uint8_t *raw_nodes = buffer.get();
  IfAddrsNode *prev_node = nullptr;
  for (const IfAddrsEntry &entry : store) {
    auto *node = ::new (raw_nodes) IfAddrsNode;
    raw_nodes += sizeof(IfAddrsNode);

    if (prev_node)
      prev_node->ifa.ifa_next = &node->ifa;
    prev_node = node;

    node->entry = entry;
    node->ifa.ifa_next = nullptr;
    node->ifa.ifa_name = node->entry.name;
    node->ifa.ifa_flags = entry.flags;
    node->ifa.ifa_addr = get_sockaddr(node->entry.addr);
    node->ifa.ifa_netmask = get_sockaddr(node->entry.netmask);
    node->ifa.ifa_broadaddr = get_sockaddr(node->entry.ifu);
    node->ifa.ifa_data = nullptr;
  }
  return reinterpret_cast<struct ifaddrs *>(buffer.release());
}

} // namespace detail

template <typename Policy> LIBC_INLINE ErrorOr<struct ifaddrs *> getifaddrs() {
  NetlinkSocket<Policy> sock;
  if (ErrorOr<void> res = sock.open(); !res.has_value())
    return Error(res.error());

  // If the data changes while the kernel is sending it to us, the result may
  // be inconsistent. In that case, we throw it away and try again.
  for (size_t attempt = 0; attempt < MAX_DUMP_ATTEMPTS; ++attempt) {
    detail::IfAddrsStore store;
    cpp::scope_exit destroy_store(
        [&store]() { detail::IfAddrsStore::destroy(&store); });

    ErrorOr<DumpResult> res = sock.dump(
        RTM_GETLINK, ifinfomsg{}, RTM_NEWLINK,
        [&store](struct nlmsghdr &nh) { return detail::add_link(nh, store); });
    if (!res.has_value())
      return Error(res.error());
    if (*res == DumpResult::Interrupted)
      continue;

    if (ErrorOr<void> close_res = sock.close(); !close_res.has_value())
      return Error(close_res.error());
    return detail::build_ifaddrs_list(store);
  }
  return Error(EAGAIN);
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_IFADDRS_LINUX_GETIFADDRS_IMPL_H
