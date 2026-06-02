//===-- Helpers for socket tests --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_SYS_SOCKET_LINUX_SOCKET_TEST_SUPPORT_H
#define LLVM_LIBC_TEST_SRC_SYS_SOCKET_LINUX_SOCKET_TEST_SUPPORT_H

#include "hdr/sys_socket_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/socklen_t.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/string/strncpy.h"
#include "src/string/strnlen.h"
#include "test/UnitTest/LibcTest.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

[[nodiscard]] LIBC_INLINE bool make_sockaddr_un(cpp::string_view path,
                                                struct sockaddr_un &sun) {
  sun.sun_family = AF_UNIX;
  // The kernel accepts addresses which fill the entire sun_path buffer (without
  // the terminating '\0' character), but we don't do that as it makes matching
  // the returned values more difficult.
  if (path.size() + 1 > sizeof(sun.sun_path))
    return false;
  strncpy(sun.sun_path, path.data(), sizeof(sun.sun_path));
  return true;
}

struct SocketAddress {
  struct sockaddr_un addr;
  socklen_t addrlen;
};

class SocketAddressMatcher : public Matcher<SocketAddress> {
  cpp::string_view expected_path;
  struct sockaddr_un actual_addr;
  socklen_t actual_addrlen;

  static constexpr size_t SUN_PATH_OFFSET =
      offsetof(struct sockaddr_un, sun_path);

public:
  explicit SocketAddressMatcher(cpp::string_view path) : expected_path(path) {}

  bool match(const SocketAddress &actual) {
    actual_addr = actual.addr;
    actual_addrlen = actual.addrlen;
    if (actual_addr.sun_family != AF_UNIX)
      return false;
    if (actual_addrlen > sizeof(actual_addr))
      return false;
    size_t expected_path_len = expected_path.size();
    if (expected_path_len + 1 + SUN_PATH_OFFSET > actual_addrlen)
      return false;
    if (actual_addrlen < SUN_PATH_OFFSET)
      return false;
    cpp::string_view actual_path(
        actual_addr.sun_path,
        strnlen(actual_addr.sun_path, actual_addrlen - SUN_PATH_OFFSET));
    return actual_path == expected_path;
  }

  void explainError() override {
    if (actual_addr.sun_family != AF_UNIX) {
      tlog << "Expected address family to be AF_UNIX but got "
           << actual_addr.sun_family << "\n";
      return;
    }
    if (actual_addrlen > sizeof(actual_addr)) {
      tlog << "Expected address length to be less than or equal to "
           << sizeof(actual_addr) << " but got " << actual_addrlen << "\n";
      return;
    }
    size_t expected_path_len = expected_path.size();
    if (expected_path_len + 1 + SUN_PATH_OFFSET > actual_addrlen) {
      tlog << "Expected address length to be less than or equal to "
           << expected_path_len + 1 + SUN_PATH_OFFSET << " but got "
           << actual_addrlen << "\n";
      return;
    }
    if (actual_addrlen < SUN_PATH_OFFSET) {
      tlog << "Expected address length to be greater than or equal to "
           << SUN_PATH_OFFSET << " but got " << actual_addrlen << "\n";
      return;
    }
    cpp::string_view actual_path(
        actual_addr.sun_path,
        strnlen(actual_addr.sun_path, actual_addrlen - SUN_PATH_OFFSET));
    if (actual_path != expected_path) {
      tlog << "Expected address path to be " << expected_path << " but got "
           << actual_path << "\n";
      return;
    }
  }
};

LIBC_INLINE SocketAddressMatcher MatchesAddress(cpp::string_view path) {
  return SocketAddressMatcher(path);
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_SRC_SYS_SOCKET_LINUX_SOCKET_TEST_SUPPORT_H
