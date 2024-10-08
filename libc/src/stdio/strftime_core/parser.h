//===-- Format string parser for printf -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_PARSER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_PARSER_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/algorithm.h" // max
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"
#include "src/__support/str_to_integer.h"
#include "src/stdio/strftime_core/core_structs.h"
// #include "src/stdio/strftime_core/printf_config.h"

#include <stddef.h>

#ifdef LIBC_INTERNAL_PRINTF_HAS_FIXED_POINT
#include "src/__support/fixed_point/fx_rep.h"
#endif // LIBC_INTERNAL_PRINTF_HAS_FIXED_POINT
#ifndef LIBC_COPT_PRINTF_DISABLE_STRERROR
#include "src/errno/libc_errno.h"
#endif // LIBC_COPT_PRINTF_DISABLE_STRERROR

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

class Parser {
  const char *__restrict str;
  const struct tm &time;
  size_t cur_pos = 0;

public:
  LIBC_INLINE Parser(const char *__restrict new_str, const tm &time)
      : str(new_str), time(time) {}

  // get_next_section will parse the format string until it has a fully
  // specified format section. This can either be a raw format section with no
  // conversion, or a format section with a conversion that has all of its
  // variables stored in the format section.
  LIBC_INLINE FormatSection get_next_section() {
    FormatSection section;
    size_t starting_pos = cur_pos;
    if (str[cur_pos] != '%') {
      // raw section
      section.has_conv = false;
      while (str[cur_pos] != '%' && str[cur_pos] != '\0')
        ++cur_pos;
    } else {
      // format section
      section.has_conv = true;
      section.time = &time;
      // locale-specific modifiers
      if (str[cur_pos] == 'E')
        section.isE = true;
      if (str[cur_pos] == 'O')
        section.isO = true;
      ++cur_pos;
      section.conv_name = str[cur_pos];

      switch (str[cur_pos]) {
      case ('%'):
      case ('a'):
      case ('A'):
      case ('b'):
      case ('B'):
      case ('c'):
      case ('C'):
      case ('d'):
      case ('D'):
      case ('e'):
      case ('F'):
      case ('g'):
      case ('G'):
      case ('h'):
      case ('H'):
      case ('I'):
      case ('j'):
      case ('m'):
      case ('M'):
      case ('n'):
      case ('p'):
      case ('r'):
      case ('R'):
      case ('S'):
      case ('t'):
      case ('T'):
      case ('u'):
      case ('U'):
      case ('V'):
      case ('w'):
      case ('W'):
      case ('x'):
      case ('X'):
      case ('y'):
      case ('Y'):
      case ('z'):
      case ('Z'):
        section.has_conv = true;
        break;
      default:
        // if the conversion is undefined, change this to a raw section.
        section.has_conv = false;
        break;
      }
          // If the end of the format section is on the '\0'. This means we need to
    // not advance the cur_pos.
    if (str[cur_pos] != '\0')
      ++cur_pos;
    }
    section.raw_string = {str + starting_pos, cur_pos - starting_pos};
    return section;
}

};

} // namespace strftime_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_PARSER_H
