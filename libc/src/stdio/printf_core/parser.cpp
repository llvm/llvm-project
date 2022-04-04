//===-- Format string parser implementation for printf ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "parser.h"

#include "src/__support/arg_list.h"

#include "src/__support/CPP/Bit.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_integer.h"

namespace __llvm_libc {
namespace printf_core {

#define LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE 1 // This will be a compile flag.

#ifndef LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE
#define GET_ARG_VAL_SIMPLEST(arg_type, index) get_arg_value<arg_type>(index)
#else
#define GET_ARG_VAL_SIMPLEST(arg_type, _) get_next_arg_value<arg_type>()
#endif // LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

FormatSection Parser::get_next_section() {
  FormatSection section;
  section.raw_string = str + cur_pos;
  size_t starting_pos = cur_pos;
  if (str[cur_pos] == '%') {
    // format section
    section.has_conv = true;

    ++cur_pos;
    [[maybe_unused]] size_t conv_index = 0;

    section.flags = parse_flags(&cur_pos);

    // handle width
    section.min_width = 0;
    if (str[cur_pos] == '*') {
      ++cur_pos;

      section.min_width = GET_ARG_VAL_SIMPLEST(int, parse_index(&cur_pos));
    } else if (internal::isdigit(str[cur_pos])) {
      char *int_end;
      section.min_width =
          internal::strtointeger<int>(str + cur_pos, &int_end, 10);
      cur_pos = int_end - str;
    }
    if (section.min_width < 0) {
      section.min_width = -section.min_width;
      section.flags =
          static_cast<FormatFlags>(section.flags | FormatFlags::LEFT_JUSTIFIED);
    }

    // handle precision
    section.precision = -1; // negative precisions are ignored.
    if (str[cur_pos] == '.') {
      ++cur_pos;
      section.precision = 0; // if there's a . but no specified precision, the
                             // precision is implicitly 0.
      if (str[cur_pos] == '*') {
        ++cur_pos;

        section.precision = GET_ARG_VAL_SIMPLEST(int, parse_index(&cur_pos));

      } else if (internal::isdigit(str[cur_pos])) {
        char *int_end;
        section.precision =
            internal::strtointeger<int>(str + cur_pos, &int_end, 10);
        cur_pos = int_end - str;
      }
    }

    LengthModifier lm = parse_length_modifier(&cur_pos);

    section.length_modifier = lm;
    section.conv_name = str[cur_pos];
    switch (str[cur_pos]) {
    case ('%'):
      break;
    case ('c'):
      section.conv_val_raw = GET_ARG_VAL_SIMPLEST(int, conv_index);
      break;
    case ('d'):
    case ('i'):
    case ('o'):
    case ('x'):
    case ('X'):
    case ('u'):
      switch (lm) {
      case (LengthModifier::hh):
      case (LengthModifier::h):
      case (LengthModifier::none):
        section.conv_val_raw = GET_ARG_VAL_SIMPLEST(int, conv_index);
        break;
      case (LengthModifier::l):
        section.conv_val_raw = GET_ARG_VAL_SIMPLEST(long, conv_index);
        break;
      case (LengthModifier::ll):
      case (LengthModifier::L): // This isn't in the standard, but is in other
                                // libc implementations.
        section.conv_val_raw = GET_ARG_VAL_SIMPLEST(long long, conv_index);
        break;
      case (LengthModifier::j):
        section.conv_val_raw = GET_ARG_VAL_SIMPLEST(intmax_t, conv_index);
        break;
      case (LengthModifier::z):
        section.conv_val_raw = GET_ARG_VAL_SIMPLEST(size_t, conv_index);
        break;
      case (LengthModifier::t):
        section.conv_val_raw = GET_ARG_VAL_SIMPLEST(ptrdiff_t, conv_index);
        break;
      }
      break;
    case ('f'):
    case ('F'):
    case ('e'):
    case ('E'):
    case ('a'):
    case ('A'):
    case ('g'):
    case ('G'):
      if (lm != LengthModifier::L)
        section.conv_val_raw =
            bit_cast<uint64_t>(GET_ARG_VAL_SIMPLEST(double, conv_index));
      else
        section.conv_val_raw = bit_cast<__uint128_t>(
            GET_ARG_VAL_SIMPLEST(long double, conv_index));
      break;
    case ('n'):
    case ('p'):
    case ('s'):
      section.conv_val_ptr = GET_ARG_VAL_SIMPLEST(void *, conv_index);
      break;
    default:
      // if the conversion is undefined, change this to a raw section.
      section.has_conv = false;
      break;
    }
    ++cur_pos;
  } else {
    // raw section
    section.has_conv = false;
    while (str[cur_pos] != '%' && str[cur_pos] != '\0')
      ++cur_pos;
  }
  section.raw_len = cur_pos - starting_pos;
  return section;
}

FormatFlags Parser::parse_flags(size_t *local_pos) {
  bool found_flag = true;
  FormatFlags flags = FormatFlags(0);
  while (found_flag) {
    switch (str[*local_pos]) {
    case '-':
      flags = static_cast<FormatFlags>(flags | FormatFlags::LEFT_JUSTIFIED);
      break;
    case '+':
      flags = static_cast<FormatFlags>(flags | FormatFlags::FORCE_SIGN);
      break;
    case ' ':
      flags = static_cast<FormatFlags>(flags | FormatFlags::SPACE_PREFIX);
      break;
    case '#':
      flags = static_cast<FormatFlags>(flags | FormatFlags::ALTERNATE_FORM);
      break;
    case '0':
      flags = static_cast<FormatFlags>(flags | FormatFlags::LEADING_ZEROES);
      break;
    default:
      found_flag = false;
    }
    if (found_flag)
      ++*local_pos;
  }
  return flags;
}

LengthModifier Parser::parse_length_modifier(size_t *local_pos) {
  switch (str[*local_pos]) {
  case ('l'):
    if (str[*local_pos + 1] == 'l') {
      *local_pos += 2;
      return LengthModifier::ll;
    } else {
      ++*local_pos;
      return LengthModifier::l;
    }
  case ('h'):
    if (str[cur_pos + 1] == 'h') {
      *local_pos += 2;
      return LengthModifier::hh;
    } else {
      ++*local_pos;
      return LengthModifier::h;
    }
  case ('L'):
    ++*local_pos;
    return LengthModifier::L;
  case ('j'):
    ++*local_pos;
    return LengthModifier::j;
  case ('z'):
    ++*local_pos;
    return LengthModifier::z;
  case ('t'):
    ++*local_pos;
    return LengthModifier::t;
  default:
    return LengthModifier::none;
  }
}

} // namespace printf_core
} // namespace __llvm_libc
