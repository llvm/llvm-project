//===-- Implementation of inet_aton function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/inet_aton.h"
#include "src/__support/common.h"
#include "src/__support/endian_internal.h"
#include "src/__support/str_to_integer.h"

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

  if (dot_num > 3)
    return 0;

  unsigned long result = 0;
  for (int i = 0; i <= dot_num; ++i) {
    unsigned max_part = i == dot_num ? (0xffffffffUL >> (8 * dot_num)) : 0xffUL;
    if (parts[i] > max_part)
      return 0;
    int shift = i == dot_num ? 0 : 8 * (3 - i);
    result |= parts[i] << shift;
  }

  if (inp)
    inp->s_addr = Endian::to_big_endian(static_cast<uint32_t>(result));

  return 1;
}

} // namespace LIBC_NAMESPACE_DECL
