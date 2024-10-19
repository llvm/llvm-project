//===-- Format string parser for printf -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_PARSER_H
#define LLVM_LIBC_SRC_STDIO_STRFTIME_CORE_PARSER_H

#include "core_structs.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/string/string_utils.h"
#include <time.h>

namespace LIBC_NAMESPACE_DECL {
namespace strftime_core {

static constexpr cpp::string_view valid_conversions_after_E = "cCxXyY";
static constexpr cpp::string_view valid_conversions_after_O =
    "dHeHIOmMSuUVwWyY";
static constexpr cpp::string_view all_valid_conversions =
    "%aAbBcCdDeFgGhHIjmMnprRSuUVwWxXyYzZ";

int min_width(char conv) {
  if (internal::strchr_implementation("CdegHImMSUVWy", conv))
    return 2;
  if (conv == 'j')
    return 3;
  return 0;
}

char get_padding(char conv) {
  if (internal::strchr_implementation("CdgHIjmMSUVWy", conv))
    return '0';
  return ' ';
}

class Parser {
  const char *str;
  const struct tm &time;
  size_t cur_pos = 0;

public:
  LIBC_INLINE Parser(const char *new_str, const struct tm &time)
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
      ++cur_pos;
      // locale-specific modifiers
      if (str[cur_pos] == 'E') {
        section.isE = true;
        ++cur_pos;
      }
      if (str[cur_pos] == 'O') {
        section.isO = true;
        ++cur_pos;
      }
      section.conv_name = str[cur_pos];

      // Check if modifiers are valid
      if ((section.isE &&
           !internal::strchr_implementation(valid_conversions_after_E.data(),
                                            str[cur_pos])) ||
          (section.isO &&
           !internal::strchr_implementation(valid_conversions_after_O.data(),
                                            str[cur_pos])) ||
          (!internal::strchr_implementation(all_valid_conversions.data(),
                                            str[cur_pos]))) {
        section.has_conv = false;
      }

      section.min_width = min_width(str[cur_pos]);
      section.padding = get_padding(str[cur_pos]);

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
