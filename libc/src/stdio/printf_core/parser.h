//===-- Format string parser for printf -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PARSER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PARSER_H

#include "src/__support/CPP/optional.h"
#include "src/__support/str_to_integer.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/stdio/printf_core/printf_config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace printf_core {

template <typename T> struct int_type_of {
  using type = T;
};
template <> struct int_type_of<double> {
  using type = fputil::FPBits<double>::StorageType;
};
template <> struct int_type_of<long double> {
  using type = fputil::FPBits<long double>::StorageType;
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

template <typename ArgProvider> class Parser {
  const char *__restrict str;

  size_t cur_pos = 0;
  ArgProvider args_cur;

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
  // args_start stores the start of the va_args, which is allows getting the
  // value of arguments that have already been passed. args_index is tracked so
  // that we know which argument args_cur is on.
  ArgProvider args_start;
  size_t args_index = 1;

  // Defined in printf_config.h
  static constexpr size_t DESC_ARR_LEN = LIBC_COPT_PRINTF_INDEX_ARR_LEN;

  // desc_arr stores the sizes of the variables in the ArgProvider. This is used
  // in index mode to reduce repeated string parsing. The sizes are stored as
  // TypeDesc objects, which store the size as well as minimal type information.
  // This is necessary because some systems separate the floating point and
  // integer values in va_args.
  TypeDesc desc_arr[DESC_ARR_LEN] = {type_desc_from_type<void>()};

  // TODO: Look into object stores for optimization.

#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

public:
#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
  LIBC_INLINE Parser(const char *__restrict new_str, ArgProvider &args)
      : str(new_str), args_cur(args), args_start(args) {}
#else
  LIBC_INLINE Parser(const char *__restrict new_str, ArgProvider &args)
      : str(new_str), args_cur(args) {}
#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

  // get_next_section will parse the format string until it has a fully
  // specified format section. This can either be a raw format section with no
  // conversion, or a format section with a conversion that has all of its
  // variables stored in the format section.
  LIBC_INLINE FormatSection get_next_section() {
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
        section.flags = static_cast<FormatFlags>(section.flags |
                                                 FormatFlags::LEFT_JUSTIFIED);
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
        // Regardless of options, a % conversion is always safe. The standard
        // says that "The complete conversion specification shall be %%" but it
        // also says that "If a conversion specification is invalid, the
        // behavior is undefined." Based on that we define that any conversion
        // specification ending in '%' shall display as '%' regardless of any
        // valid or invalid options.
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
        WRITE_ARG_VAL_SIMPLEST(section.conv_val_ptr, void *, conv_index);
        break;
      case ('s'):
        WRITE_ARG_VAL_SIMPLEST(section.conv_val_ptr, char *, conv_index);
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

private:
  // parse_flags parses the flags inside a format string. It assumes that
  // str[*local_pos] is inside a format specifier, and parses any flags it
  // finds. It returns a FormatFlags object containing the set of found flags
  // arithmetically or'd together. local_pos will be moved past any flags found.
  LIBC_INLINE FormatFlags parse_flags(size_t *local_pos) {
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

  // parse_length_modifier parses the length modifier inside a format string. It
  // assumes that str[*local_pos] is inside a format specifier. It returns a
  // LengthModifier with the length modifier it found. It will advance local_pos
  // after the format specifier if one is found.
  LIBC_INLINE LengthModifier parse_length_modifier(size_t *local_pos) {
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

  // get_next_arg_value gets the next value from the arg list as type T.
  template <class T> LIBC_INLINE T get_next_arg_value() {
    return args_cur.template next_var<T>();
  }

  //----------------------------------------------------
  // INDEX MODE ONLY FUNCTIONS AFTER HERE:
  //----------------------------------------------------

#ifndef LIBC_COPT_PRINTF_DISABLE_INDEX_MODE

  // parse_index parses the index of a value inside a format string. It
  // assumes that str[*local_pos] points to character after a '%' or '*', and
  // returns 0 if there is no closing $, or if it finds no number. If it finds a
  // number, it will move local_pos past the end of the $, else it will not move
  // local_pos.
  LIBC_INLINE size_t parse_index(size_t *local_pos) {
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

  LIBC_INLINE void set_type_desc(size_t index, TypeDesc value) {
    if (index != 0 && index <= DESC_ARR_LEN)
      desc_arr[index - 1] = value;
  }

  // get_arg_value gets the value from the arg list at index (starting at 1).
  // This may require parsing the format string. An index of 0 is interpreted as
  // the next value. If the format string is not valid, it may have gaps in its
  // indexes. Requesting the value for any index after a gap will fail, since
  // the arg list must be read in order and with the correct types.
  template <class T> LIBC_INLINE cpp::optional<T> get_arg_value(size_t index) {
    if (!(index == 0 || index == args_index)) {
      bool success = args_to_index(index);
      if (!success) {
        // If we can't get to this index, then the value of the arg can't be
        // found.
        return cpp::optional<T>();
      }
    }

    set_type_desc(index, type_desc_from_type<T>());

    ++args_index;
    return get_next_arg_value<T>();
  }

  // the ArgProvider can only return the next item in the list. This function is
  // used in index mode when the item that needs to be read is not the next one.
  // It moves cur_args to the index requested so the appropriate value may
  // be read. This may involve parsing the format string, and is in the worst
  // case an O(n^2) operation.
  LIBC_INLINE bool args_to_index(size_t index) {
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
      // string, then then advancing to the requested index is impossible. In
      // that case the function returns false.
      if (cur_type_desc == type_desc_from_type<void>())
        return false;

      if (cur_type_desc == type_desc_from_type<uint32_t>())
        args_cur.template next_var<uint32_t>();
      else if (cur_type_desc == type_desc_from_type<uint64_t>())
        args_cur.template next_var<uint64_t>();
#ifndef LIBC_COPT_PRINTF_DISABLE_FLOAT
      // Floating point numbers are stored separately from the other arguments.
      else if (cur_type_desc == type_desc_from_type<double>())
        args_cur.template next_var<double>();
      else if (cur_type_desc == type_desc_from_type<long double>())
        args_cur.template next_var<long double>();
#endif // LIBC_COPT_PRINTF_DISABLE_FLOAT
      // pointers may be stored separately from normal values.
      else if (cur_type_desc == type_desc_from_type<void *>())
        args_cur.template next_var<void *>();
      else
        args_cur.template next_var<uint32_t>();

      ++args_index;
    }
    return true;
  }

  // get_type_desc assumes that this format string uses index mode. It iterates
  // through the format string until it finds a format specifier that defines
  // the type of index, and returns a TypeDesc describing that type. It does not
  // modify cur_pos.
  LIBC_INLINE TypeDesc get_type_desc(size_t index) {
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
        // unknown and all this information is irrelevant. The rest of this
        // logic has been for skipping past this conversion properly to avoid
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
          case (LengthModifier::L): // This isn't in the standard, but is in
                                    // other libc implementations.
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

#endif // LIBC_COPT_PRINTF_DISABLE_INDEX_MODE
};

} // namespace printf_core
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_PARSER_H
