//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for inet_pton
///
//===----------------------------------------------------------------------===//

#include "inet_pton.h"

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_in6_addr.h"
#include "hdr/types/struct_in_addr.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/str_to_integer.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, inet_pton,
                   (int af, const char *__restrict src, void *__restrict dst)) {
  LIBC_CRASH_ON_NULLPTR(src);
  LIBC_CRASH_ON_NULLPTR(dst);
  if (af == AF_INET6) {
    return 0;
  } else if (af != AF_INET) {
    libc_errno = EAFNOSUPPORT;
    return -1;
  }
  uint8_t bytes[4];
  size_t start = 0;
  cpp::string_view str(src);
  size_t i{0};
  for (; i < 4; ++i) {
    size_t end = str.find_first_of('.', start);
    if (i < 3 && start == cpp::string_view::npos)
      return 0; // Missing dot
    if (i == 3 && end != cpp::string_view::npos)
      return 0; // Extra dot
    cpp::string_view part =
        (i == 3) ? str.substr(start) : str.substr(start, end - start);
    if (part.empty())
      return 0; // empty part e.g. 192..1.1
    if (part.size() > 1 && part[0] == '0')
      return 0;
    // Ensure all characters are valid ascii
    for (char c : part) {
      if (c < '0' || c > '9')
        return 0;
    }
    auto result = internal::strtointeger<uint32_t>(part.data(), 10);
    if (result.has_error() || result.value > 255 || result.value < 0)
      return 0;
    bytes[i] = static_cast<uint8_t>(result.value);
    start = end + 1;
  }
  auto *addr = reinterpret_cast<struct in_addr *>(dst);
  __builtin_memcpy(&addr->s_addr, bytes, 4);
  return 1;
}

} // namespace LIBC_NAMESPACE_DECL
