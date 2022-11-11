//===-- Format string parser implementation for scanf ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #define LLVM_LIBC_SCANF_DISABLE_INDEX_MODE 1 // This will be a compile flag.

#include "src/stdio/scanf_core/parser.h"

#include "src/__support/arg_list.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/bitset.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_integer.h"

namespace __llvm_libc {
namespace scanf_core {

#ifndef LLVM_LIBC_SCANF_DISABLE_INDEX_MODE
#define GET_ARG_VAL_SIMPLEST(arg_type, index) get_arg_value<arg_type>(index)
#else
#define GET_ARG_VAL_SIMPLEST(arg_type, _) get_next_arg_value<arg_type>()
#endif // LLVM_LIBC_SCANF_DISABLE_INDEX_MODE

FormatSection Parser::get_next_section() {
  FormatSection section;
  size_t starting_pos = cur_pos;
  if (str[cur_pos] == '%') {
    // format section
    section.has_conv = true;

    ++cur_pos;
    [[maybe_unused]] size_t conv_index = 0;

#ifndef LLVM_LIBC_SCANF_DISABLE_INDEX_MODE
    conv_index = parse_index(&cur_pos);
#endif // LLVM_LIBC_SCANF_DISABLE_INDEX_MODE

    if (str[cur_pos] == '*') {
      ++cur_pos;
      section.flags = FormatFlags::NO_WRITE;
    }

    // handle width
    section.max_width = -1;
    if (internal::isdigit(str[cur_pos])) {
      char *int_end;
      section.max_width =
          internal::strtointeger<int>(str + cur_pos, &int_end, 10);
      cur_pos = int_end - str;
    }

    // TODO(michaelrj): add posix allocate flag support.
    // if (str[cur_pos] == 'm') {
    //   ++cur_pos;
    //   section.flags = FormatFlags::ALLOCATE;
    // }

    LengthModifier lm = parse_length_modifier(&cur_pos);
    section.length_modifier = lm;

    section.conv_name = str[cur_pos];

    // If NO_WRITE is not set, then read the next arg as the output pointer.
    if ((section.flags & FormatFlags::NO_WRITE) == 0) {
      // Since all outputs are pointers, there's no need to distinguish when
      // reading from va_args. They're all the same size and stored the same.
      section.output_ptr = GET_ARG_VAL_SIMPLEST(void *, conv_index);
    }

    // If the end of the format section is on the '\0'. This means we need to
    // not advance the cur_pos and we should not count this has having a
    // conversion.
    if (str[cur_pos] != '\0') {
      ++cur_pos;
    } else {
      section.has_conv = false;
    }

    // If the format is a bracketed one, then we need to parse out the insides
    // of the brackets.
    if (section.conv_name == '[') {
      constexpr char CLOSING_BRACKET = ']';
      constexpr char INVERT_FLAG = '^';
      constexpr char RANGE_OPERATOR = '-';

      cpp::bitset<256> scan_set;
      bool invert = false;

      // The circumflex in the first position represents the inversion flag, but
      // it's easier to apply that at the end so we just store it for now.
      if (str[cur_pos] == INVERT_FLAG) {
        invert = true;
        ++cur_pos;
      }

      // This is used to determine if a hyphen is being used as a literal or as
      // a range operator.
      size_t set_start_pos = cur_pos;

      // Normally the right bracket closes the set, but if it's the first
      // character (possibly after the inversion flag) then it's instead
      // included as a character in the set and the second right bracket closes
      // the set.
      if (str[cur_pos] == CLOSING_BRACKET) {
        scan_set.set(CLOSING_BRACKET);
        ++cur_pos;
      }

      while (str[cur_pos] != '\0' && str[cur_pos] != CLOSING_BRACKET) {
        // If a hyphen is being used as a range operator, since it's neither at
        // the beginning nor end of the set.
        if (str[cur_pos] == RANGE_OPERATOR && cur_pos != set_start_pos &&
            str[cur_pos + 1] != CLOSING_BRACKET && str[cur_pos + 1] != '\0') {
          // Technically there is no requirement to correct the ordering of the
          // range, but since the range operator is entirely implementation
          // defined it seems like a good convenience.
          char a = str[cur_pos - 1];
          char b = str[cur_pos + 1];
          char start = (a < b ? a : b);
          char end = (a < b ? b : a);
          scan_set.set_range(start, end);
          cur_pos += 2;
        } else {
          scan_set.set(str[cur_pos]);
          ++cur_pos;
        }
      }
      if (invert)
        scan_set.flip();

      if (str[cur_pos] == CLOSING_BRACKET) {
        ++cur_pos;
        section.scan_set = scan_set;
      } else {
        // if the end of the string was encountered, this is not a valid set.
        section.has_conv = false;
      }
    }
  } else {
    // raw section
    section.has_conv = false;
    while (str[cur_pos] != '%' && str[cur_pos] != '\0')
      ++cur_pos;
  }
  section.raw_string = {str + starting_pos, cur_pos - starting_pos};
  return section;
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
    if (str[*local_pos + 1] == 'h') {
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
    return LengthModifier::NONE;
  }
}

//----------------------------------------------------
// INDEX MODE ONLY FUNCTIONS AFTER HERE:
//----------------------------------------------------

#ifndef LLVM_LIBC_SCANF_DISABLE_INDEX_MODE

size_t Parser::parse_index(size_t *local_pos) {
  if (internal::isdigit(str[*local_pos])) {
    char *int_end;
    size_t index =
        internal::strtointeger<size_t>(str + *local_pos, &int_end, 10);
    if (int_end[0] != '$')
      return 0;
    *local_pos = 1 + int_end - str;
    return index;
  }
  return 0;
}

void Parser::args_to_index(size_t index) {
  if (args_index > index) {
    args_index = 1;
    args_cur = args_start;
  }

  while (args_index < index) {
    // Since all arguments must be pointers, we can just read all of them as
    // void * and not worry about type issues.
    args_cur.next_var<void *>();
    ++args_index;
  }
}

#endif // LLVM_LIBC_SCANF_DISABLE_INDEX_MODE

} // namespace scanf_core
} // namespace __llvm_libc
