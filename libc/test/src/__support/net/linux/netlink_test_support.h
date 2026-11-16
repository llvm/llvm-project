//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helpers for testing code which talks to the kernel over netlink sockets: a
/// fake network syscall policy, and functions for constructing the netlink
/// messages it should return.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC___SUPPORT_NET_LINUX_NETLINK_TEST_SUPPORT_H
#define LLVM_LIBC_TEST_SRC___SUPPORT_NET_LINUX_NETLINK_TEST_SUPPORT_H

#include "hdr/errno_macros.h"
#include "hdr/types/socklen_t.h"
#include "hdr/types/ssize_t.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/tuple.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/fixedvector.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

#include <linux/netlink.h>
#include <linux/rtnetlink.h>

namespace LIBC_NAMESPACE_DECL {
namespace testing {

template <typename T, size_t CAPACITY>
LIBC_INLINE cpp::optional<T> pop_front(FixedVector<T, CAPACITY> &vec) {
  if (vec.empty())
    return cpp::nullopt;
  // TODO: Add front() and erase() to FixedVector, then clean this up.
  T first = vec[0];
  for (size_t i = 1; i < vec.size(); ++i)
    vec[i - 1] = vec[i];
  vec.pop_back();
  return first;
}

struct FakeNetworkSyscallPolicyData {
  FixedVector<cpp::tuple<int, int, int>, 10> socket_calls;
  FixedVector<ErrorOr<int>, 10> socket_results;
  FixedVector<cpp::tuple<int, size_t, int>, 10> sendto_calls;
  FixedVector<ErrorOr<ssize_t>, 10> sendto_results;
  // TODO: Move this into sendto_calls when FixedVector and tuple support
  // non-trivial types.
  cpp::string sendto_data;
  FixedVector<int, 10> close_calls;
  FixedVector<ErrorOr<int>, 10> close_results;
  FixedVector<cpp::tuple<int, size_t, int>, 10> recv_calls;
  // TODO: Make this ErrorOr<string> when ErrorOr and FixedVector support
  // non-trivial types.
  FixedVector<ErrorOr<cpp::span<const uint8_t>>, 10> recv_results;
};

/// A replacement for net::DefaultNetworkSyscallPolicy, which records the calls
/// in, and returns the results programmed into, the object pointed to by DATA.
template <FakeNetworkSyscallPolicyData *DATA> struct FakeNetworkSyscallPolicy {
  static ErrorOr<int> socket(int domain, int type, int protocol) {
    DATA->socket_calls.push_back(
        cpp::tuple<int, int, int>(domain, type, protocol));
    return pop_front(DATA->socket_results).value_or(Error(ENFILE));
  }

  static ErrorOr<ssize_t> sendto(int fd, const void *buf, size_t len, int flags,
                                 const struct sockaddr *, socklen_t) {
    DATA->sendto_calls.push_back(cpp::tuple<int, size_t, int>(fd, len, flags));
    DATA->sendto_data += cpp::string_view(static_cast<const char *>(buf), len);
    return pop_front(DATA->sendto_results).value_or(static_cast<ssize_t>(len));
  }

  static ErrorOr<ssize_t> recvfrom(int fd, void *buf, size_t len, int flags,
                                   struct sockaddr *, socklen_t *) {
    DATA->recv_calls.push_back(cpp::tuple<int, size_t, int>(fd, len, flags));
    ErrorOr<cpp::span<const uint8_t>> chunk =
        pop_front(DATA->recv_results).value_or(cpp::span<const uint8_t>());
    if (!chunk.has_value())
      return Error(chunk.error());
    if (chunk->size() > len)
      return Error(EINVAL);
    inline_memcpy(buf, chunk->data(), chunk->size());
    return static_cast<ssize_t>(chunk->size());
  }

  static ErrorOr<int> close(int fd) {
    DATA->close_calls.push_back(fd);
    return pop_front(DATA->close_results).value_or(0);
  }
};

/// Helper structs for constructing netlink attributes. AttrString holds a
/// string (e.g. an interface name), which may or may not be null-terminated.
template <uint16_t ATTR_TYPE> struct AttrString {
  cpp::string_view value;
  bool null_terminate = true;
  size_t payload_len() const { return value.size() + (null_terminate ? 1 : 0); }
  void write(uint8_t *dest) const {
    inline_memcpy(dest, value.data(), value.size());
    if (null_terminate)
      dest[value.size()] = '\0';
  }
  static constexpr uint16_t TYPE = ATTR_TYPE;
};

template <uint16_t ATTR_TYPE> struct AttrInt {
  unsigned int value;
  size_t payload_len() const { return sizeof(unsigned int); }
  void write(uint8_t *dest) const {
    inline_memcpy(dest, &value, sizeof(unsigned int));
  }
  static constexpr uint16_t TYPE = ATTR_TYPE;
};

template <uint16_t ATTR_TYPE> struct AttrBytes {
  cpp::span<const uint8_t> value;
  size_t payload_len() const { return value.size(); }
  void write(uint8_t *dest) const {
    if (!value.empty())
      inline_memcpy(dest, value.data(), value.size());
  }
  static constexpr uint16_t TYPE = ATTR_TYPE;
};

/// Writes a netlink message of the given type, consisting of the fixed-size
/// header (e.g. struct ifinfomsg) followed by the attributes, into the buffer.
/// Returns the length of the message.
template <typename Header, typename... Attrs>
LIBC_INLINE size_t build_nlmsg_packet(uint8_t *buf, uint16_t type,
                                      const Header &hdr,
                                      const Attrs &...attrs) {
  size_t total_len = NLMSG_SPACE(sizeof(Header)) +
                     (0 + ... + RTA_ALIGN(RTA_LENGTH(attrs.payload_len())));

  inline_memset(buf, 0, total_len);

  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = type;
  inline_memcpy(NLMSG_DATA(nh), &hdr, sizeof(Header));

  uint8_t *attr_ptr = buf + NLMSG_SPACE(sizeof(Header));

  auto write_attr = [&attr_ptr](const auto &attr) {
    size_t rta_len = RTA_LENGTH(attr.payload_len());
    struct rtattr *rta = reinterpret_cast<struct rtattr *>(attr_ptr);
    rta->rta_len = static_cast<unsigned short>(rta_len);
    rta->rta_type = attr.TYPE;
    attr.write(reinterpret_cast<uint8_t *>(RTA_DATA(rta)));
    attr_ptr += RTA_ALIGN(rta_len);
  };

  if constexpr (sizeof...(Attrs) > 0)
    (..., write_attr(attrs));

  return total_len;
}

LIBC_INLINE size_t build_nlmsg_done_packet(uint8_t *buf, int errcode = 0) {
  size_t total_len = NLMSG_LENGTH(sizeof(int));
  inline_memset(buf, 0, total_len);
  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = NLMSG_DONE;
  *reinterpret_cast<int *>(NLMSG_DATA(nh)) = errcode;
  return total_len;
}

LIBC_INLINE size_t build_nlmsg_error_packet(uint8_t *buf, int errcode) {
  size_t total_len = NLMSG_LENGTH(sizeof(struct nlmsgerr));
  inline_memset(buf, 0, total_len);
  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = NLMSG_ERROR;
  struct nlmsgerr *err = reinterpret_cast<struct nlmsgerr *>(NLMSG_DATA(nh));
  err->error = errcode;
  return total_len;
}

LIBC_INLINE size_t build_generic_nlmsg_packet(uint8_t *buf, uint16_t type) {
  size_t total_len = NLMSG_LENGTH(0);
  inline_memset(buf, 0, total_len);
  struct nlmsghdr *nh = reinterpret_cast<struct nlmsghdr *>(buf);
  nh->nlmsg_len = static_cast<uint32_t>(total_len);
  nh->nlmsg_type = type;
  return total_len;
}

/// Marks the packet in the buffer the way the kernel does when the dump was
/// interrupted by a change to the data.
LIBC_INLINE void mark_interrupted(uint8_t *buf) {
  reinterpret_cast<struct nlmsghdr *>(buf)->nlmsg_flags |= NLM_F_DUMP_INTR;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_SRC___SUPPORT_NET_LINUX_NETLINK_TEST_SUPPORT_H
