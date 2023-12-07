//===-- Format string parser implementation for printf ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #define LIBC_COPT_PRINTF_DISABLE_INDEX_MODE 1 // This will be a compile flag.

#include "parser.h"

#include "src/__support/arg_list.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/ctype_utils.h"
#include "src/__support/str_to_integer.h"
#include "src/stdio/printf_core/core_structs.h"

namespace __llvm_libc {
namespace printf_core {

template <typename T> struct int_type_of {
  using type = T;
};
template <> struct int_type_of<double> {
  using type = fputil::FPBits<double>::UIntType;
};
template <> struct int_type_of<long double> {
  using type = fputil::FPBits<long double>::UIntType;
};
template <typename T> using int_type_of_v = typename int_type_of<T>::type;

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
#define WRITE_ARG_VAL_SIMPLEST(dst, arg_type, index)                           \
  {                                                                            \
    auto temp = get_arg_value<arg_type>(index);                                \
    if (!temp.has_value()) {                                                   \
      section.has_conv = false;                                                \
    } else {                                                                   \
      dst = cpp::bit_cast<int_type_of_v<arg_type>>(temp.value());              \
    }                                                                          \
  }
#else
#define WRITE_ARG_VAL_SIMPLEST(dst, arg_type, _)                               \
  dst = cpp::bit_cast<int_type_of_v<arg_type>>(get_next_arg_value<arg_type>())
#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

FormatSection Parser::get_next_section() {
  FormatSection section;
  size_t starting_pos = cur_pos;
  if (str[cur_pos] == '%') {
    // format section
    section.has_conv = true;

    ++cur_pos;
    [[maybe_unused]] size_t conv_index = 0;

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
    conv_index = parse_index(&cur_pos);
#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

    section.flags = parse_flags(&cur_pos);

    // handle width
    section.min_width = 0;
    if (str[cur_pos] == '*') {
      ++cur_pos;

      WRITE_ARG_VAL_SIMPLEST(section.min_width, int, parse_index(&cur_pos));
    } else if (internal::isdigit(str[cur_pos])) {
      auto result = internal::strtointeger<int>(str + cur_pos, 10);
      section.min_width = result.value;
      cur_pos = cur_pos + result.parsed_len;
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

        WRITE_ARG_VAL_SIMPLEST(section.precision, int, parse_index(&cur_pos));

      } else if (internal::isdigit(str[cur_pos])) {
        auto result = internal::strtointeger<int>(str + cur_pos, 10);
        section.precision = result.value;
        cur_pos = cur_pos + result.parsed_len;
      }
    }

    LengthModifier lm = parse_length_modifier(&cur_pos);

    section.length_modifier = lm;
    section.conv_name = str[cur_pos];
    switch (str[cur_pos]) {
    case ('%'):
      // Regardless of options, a % conversion is always safe. The standard says
      // that "The complete conversion specification shall be %%" but it also
      // says that "If a conversion specification is invalid, the behavior is
      // undefined." Based on that we define that any conversion specification
      // ending in '%' shall display as '%' regardless of any valid or invalid
      // options.
      section.has_conv = true;
      break;
    case ('c'):
      WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, int, conv_index);
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
        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, int, conv_index);
        break;
      case (LengthModifier::l):
        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, long, conv_index);
        break;
      case (LengthModifier::ll):
      case (LengthModifier::L): // This isn't in the standard, but is in other
                                // libc implementations.

        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, long long, conv_index);
        break;
      case (LengthModifier::j):

        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, intmax_t, conv_index);
        break;
      case (LengthModifier::z):

        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, size_t, conv_index);
        break;
      case (LengthModifier::t):

        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, ptrdiff_t, conv_index);
        break;
      }
      break;
#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT
    case ('f'):
    case ('F'):
    case ('e'):
    case ('E'):
    case ('a'):
    case ('A'):
    case ('g'):
    case ('G'):
      if (lm != LengthModifier::L) {
        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, double, conv_index);
      } else {
        WRITE_ARG_VAL_SIMPLEST(section.conv_val_raw, long double, conv_index);
      }
      break;
#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT
#ifndef LIBC_COPT_PRINTF_DISABLE_WRITE_INT
    case ('n'):
#endif // LIBC_COPT_PRINTF_DISABLE_WRITE_INT
    case ('p'):
    case ('s'):
      WRITE_ARG_VAL_SIMPLEST(section.conv_val_ptr, void *, conv_index);
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

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

size_t Parser::parse_index(size_t *local_pos) {
  if (internal::isdigit(str[*local_pos])) {
    auto result = internal::strtointeger<int>(str + *local_pos, 10);
    size_t index = result.value;
    if (str[*local_pos + result.parsed_len] != '$')
      return 0;
    *local_pos = 1 + result.parsed_len + *local_pos;
    return index;
  }
  return 0;
}

TypeDesc Parser::get_type_desc(size_t index) {
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
        set_type_desc(width_index, type_desc_from_type<int>());
        if (width_index == index)
          return type_desc_from_type<int>();

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
          set_type_desc(precision_index, type_desc_from_type<int>());
          if (precision_index == index)
            return type_desc_from_type<int>();

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
        if (str[local_pos] != '\0')
          ++local_pos;
        continue;
      }

      TypeDesc conv_size = type_desc_from_type<void>();
      switch (str[local_pos]) {
      case ('%'):
        conv_size = type_desc_from_type<void>();
        break;
      case ('c'):
        conv_size = type_desc_from_type<int>();
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
          conv_size = type_desc_from_type<int>();
          break;
        case (LengthModifier::l):
          conv_size = type_desc_from_type<long>();
          break;
        case (LengthModifier::ll):
        case (LengthModifier::L): // This isn't in the standard, but is in other
                                  // libc implementations.
          conv_size = type_desc_from_type<long long>();
          break;
        case (LengthModifier::j):
          conv_size = type_desc_from_type<intmax_t>();
          break;
        case (LengthModifier::z):
          conv_size = type_desc_from_type<size_t>();
          break;
        case (LengthModifier::t):
          conv_size = type_desc_from_type<ptrdiff_t>();
          break;
        }
        break;
#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT
      case ('f'):
      case ('F'):
      case ('e'):
      case ('E'):
      case ('a'):
      case ('A'):
      case ('g'):
      case ('G'):
        if (lm != LengthModifier::L)
          conv_size = type_desc_from_type<double>();
        else
          conv_size = type_desc_from_type<long double>();
        break;
#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT
#ifndef LIBC_COPT_PRINTF_DISABLE_WRITE_INT
      case ('n'):
#endif // LIBC_COPT_PRINTF_DISABLE_WRITE_INT
      case ('p'):
      case ('s'):
        conv_size = type_desc_from_type<void *>();
        break;
      default:
        conv_size = type_desc_from_type<int>();
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

  // If there is no size for the requested index, then it's unknown. Return
  // void.
  return type_desc_from_type<void>();
}

bool Parser::args_to_index(size_t index) {
  if (args_index > index) {
    args_index = 1;
    args_cur = args_start;
  }

  while (args_index < index) {
    TypeDesc cur_type_desc = type_desc_from_type<void>();
    if (args_index <= DESC_ARR_LEN)
      cur_type_desc = desc_arr[args_index - 1];

    if (cur_type_desc == type_desc_from_type<void>())
      cur_type_desc = get_type_desc(args_index);

    // A type of void represents the type being unknown. If the type for the
    // requested index isn't in the desc_arr and isn't found by parsing the
    // string, then then advancing to the requested index is impossible. In that
    // case the function returns false.
    if (cur_type_desc == type_desc_from_type<void>())
      return false;

    if (cur_type_desc == type_desc_from_type<uint32_t>())
      args_cur.next_var<uint32_t>();
    else if (cur_type_desc == type_desc_from_type<uint64_t>())
      args_cur.next_var<uint64_t>();
#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT
    // Floating point numbers are stored separately from the other arguments.
    else if (cur_type_desc == type_desc_from_type<double>())
      args_cur.next_var<double>();
    else if (cur_type_desc == type_desc_from_type<long double>())
      args_cur.next_var<long double>();
#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT
    // pointers may be stored separately from normal values.
    else if (cur_type_desc == type_desc_from_type<void *>())
      args_cur.next_var<void *>();
    else
      args_cur.next_var<uint32_t>();

    ++args_index;
  }
  return true;
}

#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

} // namespace printf_core
} // namespace __llvm_libc
