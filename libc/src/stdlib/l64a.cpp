//===-- Implementation of l64a --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/l64a.h"
#include "hdr/types/size_t.h"
#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

// the standard says to only use up to 6 characters. Null terminator is
// unnecessary, but we'll add it for ease-of-use. Also going from 48 -> 56 bits
// probably won't matter since it's likely 32-bit aligned anyways.
constexpr size_t MAX_BASE64_LENGTH = 6;
LIBC_THREAD_LOCAL char BASE64_BUFFER[MAX_BASE64_LENGTH + 1];

constexpr static char b64_int_to_char(uint32_t num) {
  // from the standard: "The characters used to represent digits are '.' (dot)
  // for 0, '/' for 1, '0' through '9' for [2,11], 'A' through 'Z' for [12,37],
  // and 'a' through 'z' for [38,63]."
  LIBC_ASSERT(num < 64);
  if (num == 0)
    return '.';
  if (num == 1)
    return '/';
  if (num < 38)
    return static_cast<char>(
        internal::toupper(internal::int_to_b36_char(num - 2)));

  // this tolower is technically unnecessary, but it provides safety if we
  // change the default behavior of int_to_b36_char. Also the compiler
  // completely elides it so there's no performance penalty, see:
  // https://godbolt.org/z/o5ennv7fc
  return static_cast<char>(
      internal::tolower(internal::int_to_b36_char(num - 2 - 26)));
}

// This function takes a long and converts the low 32 bits of it into at most 6
// characters. It's returned as a pointer to a static buffer.
LLVM_LIBC_FUNCTION(char *, l64a, (long value)) {
  // static cast to uint32_t to get just the low 32 bits in a consistent way.
  // The standard says negative values are undefined, so I'm just defining them
  // to be treated as unsigned.
  uint32_t cur_value = static_cast<uint32_t>(value);
  for (size_t i = 0; i < MAX_BASE64_LENGTH; ++i) {
    uint32_t cur_char = cur_value % 64;
    BASE64_BUFFER[i] = b64_int_to_char(cur_char);
    cur_value /= 64;
  }

  BASE64_BUFFER[MAX_BASE64_LENGTH] = '\0'; // force null termination.
  return BASE64_BUFFER;
}

} // namespace LIBC_NAMESPACE_DECL
