//===-- Implementation of inet_aton function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_aton.h"
#include "src/__support/common.h"
#include "src/__support/str_to_integer.h"
#include "src/arpa/inet/htonl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, inet_aton, (const char *cp, in_addr *inp)) {
  unsigned long parts[4] = {0};
  int dot_num = 0;

  for (; dot_num < 4; ++dot_num) {
    auto result = internal::strtointeger<unsigned long>(cp, 0);
    parts[dot_num] = result;

    if (result.has_error() || result.parsed_len == 0)
      return 0;
    char next_char = *(cp + result.parsed_len);
    if (next_char != '.' && next_char != '\0')
      return 0;
    else if (next_char == '\0')
      break;
    else
      cp += (result.parsed_len + 1);
  }

  unsigned long result = 0;
  if (dot_num == 0) {
    if (parts[0] > 0xffffffff)
      return 0;
    result = parts[0];
  } else if (dot_num == 1) {
    if (parts[0] > 0xff || parts[1] > 0xffffff)
      return 0;
    result = (parts[0] << 24) | parts[1];
  } else if (dot_num == 2) {
    if (parts[0] > 0xff || parts[1] > 0xff || parts[2] > 0xffff)
      return 0;
    result = (parts[0] << 24) | (parts[1] << 16) | parts[2];
  } else if (dot_num == 3) {
    if (parts[0] > 0xff || parts[1] > 0xff || parts[2] > 0xff ||
        parts[3] > 0xff)
      return 0;
    result = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3];
  } else {
    return 0;
  }

  if (inp)
    inp->s_addr = LIBC_NAMESPACE::htonl(static_cast<uint32_t>(result));

  return 1;
}

} // namespace LIBC_NAMESPACE_DECL
