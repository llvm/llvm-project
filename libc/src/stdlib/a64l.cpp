//===-- Implementation of a64l --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/a64l.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

// I'm not sure this should go in ctype_utils since the specific ordering of
// base64 is so very implementation specific, and also this set is unusual.
// Returns -1 on any char without a specified value.
constexpr static int32_t b64_char_to_int(char ch) {
  // from the standard: "The characters used to represent digits are '.' (dot)
  // for 0, '/' for 1, '0' through '9' for [2,11], 'A' through 'Z' for [12,37],
  // and 'a' through 'z' for [38,63]."
  if (ch == '.')
    return 0;
  if (ch == '/')
    return 1;

  // handle the case of an unspecified char.
  if (!internal::isalnum(ch))
    return -1;

  bool is_lower = internal::islower(ch);
  // add 2 to account for '.' and '/', then b36_char_to_int is case insensitive
  // so add case sensitivity back.
  return internal::b36_char_to_int(ch) + 2 + (is_lower ? 26 : 0);
}

// This function takes a base 64 string and writes it to the low 32 bits of a
// long.
LLVM_LIBC_FUNCTION(long, a64l, (const char *s)) {
  // the standard says to only use up to 6 characters.
  constexpr size_t MAX_LENGTH = 6;
  int32_t result = 0;

  for (size_t i = 0; i < MAX_LENGTH && s[i] != '\0'; ++i) {
    int32_t cur_val = b64_char_to_int(s[i]);
    // The standard says what happens on an unspecified character is undefined,
    // here we treat it as the end of the string.
    if (cur_val == -1)
      break;

    // the first digit is the least significant, so for each subsequent digit we
    // shift it more. 6 bits since 2^6 = 64
    result += (cur_val << (6 * i));
  }

  // standard says to sign extend from 32 bits.
  return static_cast<long>(result);
}

} // namespace LIBC_NAMESPACE_DECL
