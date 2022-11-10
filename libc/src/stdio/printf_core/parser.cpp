//===-- Format string parser implementation for printf ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #define LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE 1 // This will be a compile flag.

#include "parser.h"

#include "src/__support/arg_list.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_integer.h"

namespace __llvm_libc {
namespace printf_core {

#ifndef LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE
#define GET_ARG_VAL_SIMPLEST(arg_type, index) get_arg_value<arg_type>(index)
#else
#define GET_ARG_VAL_SIMPLEST(arg_type, _) get_next_arg_value<arg_type>()
#endif // LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

FormatSection Parser::get_next_section() {
  FormatSection section;
  size_t starting_pos = cur_pos;
  if (str[cur_pos] == '%') {
    // format section
    section.has_conv = true;

    ++cur_pos;
    [[maybe_unused]] size_t conv_index = 0;

#ifndef LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE
    conv_index = parse_index(&cur_pos);
#endif // LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

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
#ifndef LLVM_LIBC_PRINTF_DISABLE_FLOAT
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
            cpp::bit_cast<uint64_t>(GET_ARG_VAL_SIMPLEST(double, conv_index));
      else
        section.conv_val_raw =
            cpp::bit_cast<fputil::FPBits<long double>::UIntType>(
                GET_ARG_VAL_SIMPLEST(long double, conv_index));
      break;
#endif // LLVM_LIBC_PRINTF_DISABLE_FLOAT
#ifndef LLVM_LIBC_PRINTF_DISABLE_WRITE_INT
    case ('n'):
#endif // LLVM_LIBC_PRINTF_DISABLE_WRITE_INT
    case ('p'):
    case ('s'):
      section.conv_val_ptr = GET_ARG_VAL_SIMPLEST(void *, conv_index);
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

  } else {
    // raw section
    section.has_conv = false;
    while (str[cur_pos] != '%' && str[cur_pos] != '\0')
      ++cur_pos;
  }
  section.raw_string = {str + starting_pos, cur_pos - starting_pos};
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
    return LengthModifier::none;
  }
}

//----------------------------------------------------
// INDEX MODE ONLY FUNCTIONS AFTER HERE:
//----------------------------------------------------

#ifndef LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

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

Parser::TypeDesc Parser::get_type_desc(size_t index) {
  // index mode is assumed, and the indicies start at 1, so an index
  // of 0 is invalid.
  size_t local_pos = 0;

  while (str[local_pos]) {
    if (str[local_pos] == '%') {
      ++local_pos;

      size_t conv_index = parse_index(&local_pos);

      // the flags aren't relevant for this situation, but I need to skip past
      // them so they're parsed but the result is discarded.
      parse_flags(&local_pos);

      // handle width
      if (str[local_pos] == '*') {
        ++local_pos;

        size_t width_index = parse_index(&local_pos);
        set_type_desc(width_index, TYPE_DESC<int>);
        if (width_index == index)
          return TYPE_DESC<int>;

      } else if (internal::isdigit(str[local_pos])) {
        while (internal::isdigit(str[local_pos]))
          ++local_pos;
      }

      // handle precision
      if (str[local_pos] == '.') {
        ++local_pos;
        if (str[local_pos] == '*') {
          ++local_pos;

          size_t precision_index = parse_index(&local_pos);
          set_type_desc(precision_index, TYPE_DESC<int>);
          if (precision_index == index)
            return TYPE_DESC<int>;

        } else if (internal::isdigit(str[local_pos])) {
          while (internal::isdigit(str[local_pos]))
            ++local_pos;
        }
      }

      LengthModifier lm = parse_length_modifier(&local_pos);

      // if we don't have an index for this conversion, then its position is
      // unknown and all this information is irrelevant. The rest of this logic
      // has been for skipping past this conversion properly to avoid
      // weirdness with %%.
      if (conv_index == 0) {
        ++local_pos;
        continue;
      }

      TypeDesc conv_size = TYPE_DESC<void>;
      switch (str[local_pos]) {
      case ('%'):
        conv_size = TYPE_DESC<void>;
        break;
      case ('c'):
        conv_size = TYPE_DESC<int>;
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
          conv_size = TYPE_DESC<int>;
          break;
        case (LengthModifier::l):
          conv_size = TYPE_DESC<long>;
          break;
        case (LengthModifier::ll):
        case (LengthModifier::L): // This isn't in the standard, but is in other
                                  // libc implementations.
          conv_size = TYPE_DESC<long long>;
          break;
        case (LengthModifier::j):
          conv_size = TYPE_DESC<intmax_t>;
          break;
        case (LengthModifier::z):
          conv_size = TYPE_DESC<size_t>;
          break;
        case (LengthModifier::t):
          conv_size = TYPE_DESC<ptrdiff_t>;
          break;
        }
        break;
#ifndef LLVM_LIBC_PRINTF_DISABLE_FLOAT
      case ('f'):
      case ('F'):
      case ('e'):
      case ('E'):
      case ('a'):
      case ('A'):
      case ('g'):
      case ('G'):
        if (lm != LengthModifier::L)
          conv_size = TYPE_DESC<double>;
        else
          conv_size = TYPE_DESC<long double>;
        break;
#endif // LLVM_LIBC_PRINTF_DISABLE_FLOAT
#ifndef LLVM_LIBC_PRINTF_DISABLE_WRITE_INT
      case ('n'):
#endif // LLVM_LIBC_PRINTF_DISABLE_WRITE_INT
      case ('p'):
      case ('s'):
        conv_size = TYPE_DESC<void *>;
        break;
      default:
        conv_size = TYPE_DESC<int>;
        break;
      }

      set_type_desc(conv_index, conv_size);
      if (conv_index == index)
        return conv_size;
    }
    // If the end of the format section is on the '\0'. This means we need to
    // not advance the local_pos.
    if (str[local_pos] != '\0')
      ++local_pos;
  }

  // If there is no size for the requested index, then just guess that it's an
  // int.
  return TYPE_DESC<int>;
}

void Parser::args_to_index(size_t index) {
  if (args_index > index) {
    args_index = 1;
    args_cur = args_start;
  }

  while (args_index < index) {
    Parser::TypeDesc cur_type_desc = TYPE_DESC<void>;
    if (args_index <= DESC_ARR_LEN)
      cur_type_desc = desc_arr[args_index - 1];

    if (cur_type_desc == TYPE_DESC<void>)
      cur_type_desc = get_type_desc(args_index);

    if (cur_type_desc == TYPE_DESC<uint32_t>)
      args_cur.next_var<uint32_t>();
    else if (cur_type_desc == TYPE_DESC<uint64_t>)
      args_cur.next_var<uint64_t>();
#ifndef LLVM_LIBC_PRINTF_DISABLE_FLOAT
    // Floating point numbers are stored separately from the other arguments.
    else if (cur_type_desc == TYPE_DESC<double>)
      args_cur.next_var<double>();
    else if (cur_type_desc == TYPE_DESC<long double>)
      args_cur.next_var<long double>();
#endif // LLVM_LIBC_PRINTF_DISABLE_FLOAT
    // pointers may be stored separately from normal values.
    else if (cur_type_desc == TYPE_DESC<void *>)
      args_cur.next_var<void *>();
    else
      args_cur.next_var<uint32_t>();

    ++args_index;
  }
}

#endif // LLVM_LIBC_PRINTF_DISABLE_INDEX_MODE

} // namespace printf_core
} // namespace __llvm_libc
