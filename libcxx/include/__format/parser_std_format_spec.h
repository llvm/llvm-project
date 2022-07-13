// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_H
#define _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_H

/// \file Contains the std-format-spec parser.
///
/// Most of the code can be reused in the chrono-format-spec.
/// This header has some support for the chrono-format-spec since it doesn't
/// affect the std-format-spec.

#include <__algorithm/find_if.h>
#include <__algorithm/min.h>
#include <__assert>
#include <__config>
#include <__debug>
#include <__format/format_arg.h>
#include <__format/format_error.h>
#include <__format/format_parse_context.h>
#include <__format/format_string.h>
#include <__variant/monostate.h>
#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

namespace __format_spec {

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __format::__parse_number_result< _CharT>
__parse_arg_id(const _CharT* __begin, const _CharT* __end, auto& __parse_ctx) {
  // This function is a wrapper to call the real parser. But it does the
  // validation for the pre-conditions and post-conditions.
  if (__begin == __end)
    __throw_format_error("End of input while parsing format-spec arg-id");

  __format::__parse_number_result __r =
      __format::__parse_arg_id(__begin, __end, __parse_ctx);

  if (__r.__ptr == __end || *__r.__ptr != _CharT('}'))
    __throw_format_error("Invalid arg-id");

  ++__r.__ptr;
  return __r;
}

template <class _Context>
_LIBCPP_HIDE_FROM_ABI constexpr uint32_t
__substitute_arg_id(basic_format_arg<_Context> __format_arg) {
  return visit_format_arg(
      [](auto __arg) -> uint32_t {
        using _Type = decltype(__arg);
        if constexpr (integral<_Type>) {
          if constexpr (signed_integral<_Type>) {
            if (__arg < 0)
              __throw_format_error("A format-spec arg-id replacement shouldn't "
                                   "have a negative value");
          }

          using _CT = common_type_t<_Type, decltype(__format::__number_max)>;
          if (static_cast<_CT>(__arg) >
              static_cast<_CT>(__format::__number_max))
            __throw_format_error("A format-spec arg-id replacement exceeds "
                                 "the maximum supported value");

          return __arg;
        } else if constexpr (same_as<_Type, monostate>)
          __throw_format_error("Argument index out of bounds");
        else
          __throw_format_error("A format-spec arg-id replacement argument "
                               "isn't an integral type");
      },
      __format_arg);
}

/** Helper struct returned from @ref __get_string_alignment. */
template <class _CharT>
struct _LIBCPP_TEMPLATE_VIS __string_alignment {
  /** Points beyond the last character to write to the output. */
  const _CharT* __last;
  /**
   * The estimated number of columns in the output or 0.
   *
   * Only when the output needs to be aligned it's required to know the exact
   * number of columns in the output. So if the formatted output has only a
   * minimum width the exact size isn't important. It's only important to know
   * the minimum has been reached. The minimum width is the width specified in
   * the format-spec.
   *
   * For example in this code @code std::format("{:10}", MyString); @endcode
   * the width estimation can stop once the algorithm has determined the output
   * width is 10 columns.
   *
   * So if:
   * * @ref __align == @c true the @ref __size is the estimated number of
   *   columns required.
   * * @ref __align == @c false the @ref __size is the estimated number of
   *   columns required or 0 when the estimation algorithm stopped prematurely.
   */
  ptrdiff_t __size;
  /**
   * Does the output need to be aligned.
   *
   * When alignment is needed the output algorithm needs to add the proper
   * padding. Else the output algorithm just needs to copy the input up to
   * @ref __last.
   */
  bool __align;
};

#ifndef _LIBCPP_HAS_NO_UNICODE
namespace __detail {

/**
 * Unicode column width estimates.
 *
 * Unicode can be stored in several formats: UTF-8, UTF-16, and UTF-32.
 * Depending on format the relation between the number of code units stored and
 * the number of output columns differs. The first relation is the number of
 * code units forming a code point. (The text assumes the code units are
 * unsigned.)
 * - UTF-8 The number of code units is between one and four. The first 127
 *   Unicode code points match the ASCII character set. When the highest bit is
 *   set it means the code point has more than one code unit.
 * - UTF-16: The number of code units is between 1 and 2. When the first
 *   code unit is in the range [0xd800,0xdfff) it means the code point uses two
 *   code units.
 * - UTF-32: The number of code units is always one.
 *
 * The code point to the number of columns isn't well defined. The code uses the
 * estimations defined in [format.string.std]/11. This list might change in the
 * future.
 *
 * The algorithm of @ref __get_string_alignment uses two different scanners:
 * - The simple scanner @ref __estimate_column_width_fast. This scanner assumes
 *   1 code unit is 1 column. This scanner stops when it can't be sure the
 *   assumption is valid:
 *   - UTF-8 when the code point is encoded in more than 1 code unit.
 *   - UTF-16 and UTF-32 when the first multi-column code point is encountered.
 *     (The code unit's value is lower than 0xd800 so the 2 code unit encoding
 *     is irrelevant for this scanner.)
 *   Due to these assumptions the scanner is faster than the full scanner. It
 *   can process all text only containing ASCII. For UTF-16/32 it can process
 *   most (all?) European languages. (Note the set it can process might be
 *   reduced in the future, due to updates in the scanning rules.)
 * - The full scanner @ref __estimate_column_width. This scanner, if needed,
 *   converts multiple code units into one code point then converts the code
 *   point to a column width.
 *
 * See also:
 * - [format.string.general]/11
 * - https://en.wikipedia.org/wiki/UTF-8#Encoding
 * - https://en.wikipedia.org/wiki/UTF-16#U+D800_to_U+DFFF
 */

/**
 * The first 2 column code point.
 *
 * This is the point where the fast UTF-16/32 scanner needs to stop processing.
 */
inline constexpr uint32_t __two_column_code_point = 0x1100;

/** Helper concept for an UTF-8 character type. */
template <class _CharT>
concept __utf8_character = same_as<_CharT, char> || same_as<_CharT, char8_t>;

/** Helper concept for an UTF-16 character type. */
template <class _CharT>
concept __utf16_character = (same_as<_CharT, wchar_t> && sizeof(wchar_t) == 2) || same_as<_CharT, char16_t>;

/** Helper concept for an UTF-32 character type. */
template <class _CharT>
concept __utf32_character = (same_as<_CharT, wchar_t> && sizeof(wchar_t) == 4) || same_as<_CharT, char32_t>;

/** Helper concept for an UTF-16 or UTF-32 character type. */
template <class _CharT>
concept __utf16_or_32_character = __utf16_character<_CharT> || __utf32_character<_CharT>;

/**
 * Converts a code point to the column width.
 *
 * The estimations are conforming to [format.string.general]/11
 *
 * This version expects a value less than 0x1'0000, which is a 3-byte UTF-8
 * character.
 */
_LIBCPP_HIDE_FROM_ABI inline constexpr int __column_width_3(uint32_t __c) noexcept {
  _LIBCPP_ASSERT(__c < 0x10000,
                 "Use __column_width_4 or __column_width for larger values");

  // clang-format off
  return 1 + (__c >= 0x1100 && (__c <= 0x115f ||
             (__c >= 0x2329 && (__c <= 0x232a ||
             (__c >= 0x2e80 && (__c <= 0x303e ||
             (__c >= 0x3040 && (__c <= 0xa4cf ||
             (__c >= 0xac00 && (__c <= 0xd7a3 ||
             (__c >= 0xf900 && (__c <= 0xfaff ||
             (__c >= 0xfe10 && (__c <= 0xfe19 ||
             (__c >= 0xfe30 && (__c <= 0xfe6f ||
             (__c >= 0xff00 && (__c <= 0xff60 ||
             (__c >= 0xffe0 && (__c <= 0xffe6
             ))))))))))))))))))));
  // clang-format on
}

/**
 * @overload
 *
 * This version expects a value greater than or equal to 0x1'0000, which is a
 * 4-byte UTF-8 character.
 */
_LIBCPP_HIDE_FROM_ABI inline constexpr int __column_width_4(uint32_t __c) noexcept {
  _LIBCPP_ASSERT(__c >= 0x10000,
                 "Use __column_width_3 or __column_width for smaller values");

  // clang-format off
  return 1 + (__c >= 0x1'f300 && (__c <= 0x1'f64f ||
             (__c >= 0x1'f900 && (__c <= 0x1'f9ff ||
             (__c >= 0x2'0000 && (__c <= 0x2'fffd ||
             (__c >= 0x3'0000 && (__c <= 0x3'fffd
             ))))))));
  // clang-format on
}

/**
 * @overload
 *
 * The general case, accepting all values.
 */
_LIBCPP_HIDE_FROM_ABI inline constexpr int __column_width(uint32_t __c) noexcept {
  if (__c < 0x10000)
    return __column_width_3(__c);

  return __column_width_4(__c);
}

/**
 * Estimate the column width for the UTF-8 sequence using the fast algorithm.
 */
template <__utf8_character _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__estimate_column_width_fast(const _CharT* __first,
                             const _CharT* __last) noexcept {
  return _VSTD::find_if(__first, __last,
                        [](unsigned char __c) { return __c & 0x80; });
}

/**
 * @overload
 *
 * The implementation for UTF-16/32.
 */
template <__utf16_or_32_character _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__estimate_column_width_fast(const _CharT* __first,
                             const _CharT* __last) noexcept {
  return _VSTD::find_if(__first, __last,
                        [](uint32_t __c) { return __c >= 0x1100; });
}

template <class _CharT>
struct _LIBCPP_TEMPLATE_VIS __column_width_result {
  /** The number of output columns. */
  size_t __width;
  /**
   * The last parsed element.
   *
   * This limits the original output to fit in the wanted number of columns.
   */
  const _CharT* __ptr;
};

/**
 * Small helper to determine the width of malformed Unicode.
 *
 * @note This function's only needed for UTF-8. During scanning UTF-8 there
 * are multiple place where it can be detected that the Unicode is malformed.
 * UTF-16 only requires 1 test and UTF-32 requires no testing.
 */
template <__utf8_character _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __column_width_result<_CharT>
__estimate_column_width_malformed(const _CharT* __first, const _CharT* __last,
                                  size_t __maximum, size_t __result) noexcept {
  size_t __size = __last - __first;
  size_t __n = _VSTD::min(__size, __maximum);
  return {__result + __n, __first + __n};
}

/**
 * Determines the number of output columns needed to render the input.
 *
 * @note When the scanner encounters malformed Unicode it acts as-if every code
 * unit at the end of the input is one output column. It's expected the output
 * terminal will replace these malformed code units with a one column
 * replacement characters.
 *
 * @param __first   Points to the first element of the input range.
 * @param __last    Points beyond the last element of the input range.
 * @param __maximum The maximum number of output columns. The returned number
 *                  of estimated output columns will not exceed this value.
 */
template <__utf8_character _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __column_width_result<_CharT>
__estimate_column_width(const _CharT* __first, const _CharT* __last,
                        size_t __maximum) noexcept {
  size_t __result = 0;

  while (__first != __last) {
    // Based on the number of leading 1 bits the number of code units in the
    // code point can be determined. See
    // https://en.wikipedia.org/wiki/UTF-8#Encoding
    switch (_VSTD::countl_one(static_cast<unsigned char>(*__first))) {
    case 0: // 1-code unit encoding: all 1 column
      ++__result;
      ++__first;
      break;

    case 2: // 2-code unit encoding: all 1 column
      // Malformed Unicode.
      if (__last - __first < 2) [[unlikely]]
        return __estimate_column_width_malformed(__first, __last, __maximum,
                                                 __result);
      __first += 2;
      ++__result;
      break;

    case 3: // 3-code unit encoding: either 1 or 2 columns
      // Malformed Unicode.
      if (__last - __first < 3) [[unlikely]]
        return __estimate_column_width_malformed(__first, __last, __maximum,
                                                 __result);
      {
        uint32_t __c = static_cast<unsigned char>(*__first++) & 0x0f;
        __c <<= 6;
        __c |= static_cast<unsigned char>(*__first++) & 0x3f;
        __c <<= 6;
        __c |= static_cast<unsigned char>(*__first++) & 0x3f;
        __result += __column_width_3(__c);
        if (__result > __maximum)
          return {__result - 2, __first - 3};
      }
      break;
    case 4: // 4-code unit encoding: either 1 or 2 columns
      // Malformed Unicode.
      if (__last - __first < 4) [[unlikely]]
        return __estimate_column_width_malformed(__first, __last, __maximum,
                                                 __result);
      {
        uint32_t __c = static_cast<unsigned char>(*__first++) & 0x07;
        __c <<= 6;
        __c |= static_cast<unsigned char>(*__first++) & 0x3f;
        __c <<= 6;
        __c |= static_cast<unsigned char>(*__first++) & 0x3f;
        __c <<= 6;
        __c |= static_cast<unsigned char>(*__first++) & 0x3f;
        __result += __column_width_4(__c);
        if (__result > __maximum)
          return {__result - 2, __first - 4};
      }
      break;
    default:
      // Malformed Unicode.
      return __estimate_column_width_malformed(__first, __last, __maximum,
                                               __result);
    }

    if (__result >= __maximum)
      return {__result, __first};
  }
  return {__result, __first};
}

template <__utf16_character _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __column_width_result<_CharT>
__estimate_column_width(const _CharT* __first, const _CharT* __last,
                        size_t __maximum) noexcept {
  size_t __result = 0;

  while (__first != __last) {
    uint32_t __c = *__first;
    // Is the code unit part of a surrogate pair? See
    // https://en.wikipedia.org/wiki/UTF-16#U+D800_to_U+DFFF
    if (__c >= 0xd800 && __c <= 0xDfff) {
      // Malformed Unicode.
      if (__last - __first < 2) [[unlikely]]
        return {__result + 1, __first + 1};

      __c -= 0xd800;
      __c <<= 10;
      __c += (*(__first + 1) - 0xdc00);
      __c += 0x10000;

      __result += __column_width_4(__c);
      if (__result > __maximum)
        return {__result - 2, __first};
      __first += 2;
    } else {
      __result += __column_width_3(__c);
      if (__result > __maximum)
        return {__result - 2, __first};
      ++__first;
    }

    if (__result >= __maximum)
      return {__result, __first};
  }

  return {__result, __first};
}

template <__utf32_character _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __column_width_result<_CharT>
__estimate_column_width(const _CharT* __first, const _CharT* __last,
                        size_t __maximum) noexcept {
  size_t __result = 0;

  while (__first != __last) {
    uint32_t __c = *__first;
    __result += __column_width(__c);

    if (__result > __maximum)
      return {__result - 2, __first};

    ++__first;
    if (__result >= __maximum)
      return {__result, __first};
  }

  return {__result, __first};
}

} // namespace __detail

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __string_alignment<_CharT>
__get_string_alignment(const _CharT* __first, const _CharT* __last,
                       ptrdiff_t __width, ptrdiff_t __precision) noexcept {
  _LIBCPP_ASSERT(__width != 0 || __precision != -1,
                 "The function has no effect and shouldn't be used");

  // TODO FMT There might be more optimizations possible:
  // If __precision == __format::__number_max and the encoding is:
  // * UTF-8  : 4 * (__last - __first) >= __width
  // * UTF-16 : 2 * (__last - __first) >= __width
  // * UTF-32 : (__last - __first) >= __width
  // In these cases it's certain the output is at least the requested width.
  // It's unknown how often this happens in practice. For now the improvement
  // isn't implemented.

  /*
   * First assume there are no special Unicode code units in the input.
   * - Apply the precision (this may reduce the size of the input). When
   *   __precison == -1 this step is omitted.
   * - Scan for special code units in the input.
   * If our assumption was correct the __pos will be at the end of the input.
   */
  const ptrdiff_t __length = __last - __first;
  const _CharT* __limit =
      __first +
      (__precision == -1 ? __length : _VSTD::min(__length, __precision));
  ptrdiff_t __size = __limit - __first;
  const _CharT* __pos =
      __detail::__estimate_column_width_fast(__first, __limit);

  if (__pos == __limit)
    return {__limit, __size, __size < __width};

  /*
   * Our assumption was wrong, there are special Unicode code units.
   * The range [__first, __pos) contains a set of code units with the
   * following property:
   *      Every _CharT in the range will be rendered in 1 column.
   *
   * If there's no maximum width and the parsed size already exceeds the
   *   minimum required width. The real size isn't important. So bail out.
   */
  if (__precision == -1 && (__pos - __first) >= __width)
    return {__last, 0, false};

  /* If there's a __precision, truncate the output to that width. */
  ptrdiff_t __prefix = __pos - __first;
  if (__precision != -1) {
    _LIBCPP_ASSERT(__precision > __prefix, "Logic error.");
    auto __lengh_info = __detail::__estimate_column_width(
        __pos, __last, __precision - __prefix);
    __size = __lengh_info.__width + __prefix;
    return {__lengh_info.__ptr, __size, __size < __width};
  }

  /* Else use __width to determine the number of required padding characters. */
  _LIBCPP_ASSERT(__width > __prefix, "Logic error.");
  /*
   * The column width is always one or two columns. For the precision the wanted
   * column width is the maximum, for the width it's the minimum. Using the
   * width estimation with its truncating behavior will result in the wrong
   * result in the following case:
   * - The last code unit processed requires two columns and exceeds the
   *   maximum column width.
   * By increasing the __maximum by one avoids this issue. (It means it may
   * pass one code point more than required to determine the proper result;
   * that however isn't a problem for the algorithm.)
   */
  size_t __maximum = 1 + __width - __prefix;
  auto __lengh_info =
      __detail::__estimate_column_width(__pos, __last, __maximum);
  if (__lengh_info.__ptr != __last) {
    // Consumed the width number of code units. The exact size of the string
    // is unknown. We only know we don't need to align the output.
    _LIBCPP_ASSERT(static_cast<ptrdiff_t>(__lengh_info.__width + __prefix) >=
                       __width,
                   "Logic error");
    return {__last, 0, false};
  }

  __size = __lengh_info.__width + __prefix;
  return {__last, __size, __size < __width};
}
#else  // _LIBCPP_HAS_NO_UNICODE
template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __string_alignment<_CharT>
__get_string_alignment(const _CharT* __first, const _CharT* __last,
                       ptrdiff_t __width, ptrdiff_t __precision) noexcept {
  const ptrdiff_t __length = __last - __first;
  const _CharT* __limit =
      __first +
      (__precision == -1 ? __length : _VSTD::min(__length, __precision));
  ptrdiff_t __size = __limit - __first;
  return {__limit, __size, __size < __width};
}
#endif // _LIBCPP_HAS_NO_UNICODE

/// These fields are a filter for which elements to parse.
///
/// They default to false so when a new field is added it needs to be opted in
/// explicitly.
struct __fields {
  uint8_t __sign_ : 1 {false};
  uint8_t __alternate_form_ : 1 {false};
  uint8_t __zero_padding_ : 1 {false};
  uint8_t __precision_ : 1 {false};
  uint8_t __locale_specific_form_ : 1 {false};
  uint8_t __type_ : 1 {false};
};

// By not placing this constant in the formatter class it's not duplicated for
// char and wchar_t.
inline constexpr __fields __fields_integral{
    .__sign_                 = true,
    .__alternate_form_       = true,
    .__zero_padding_         = true,
    .__locale_specific_form_ = true,
    .__type_                 = true};
inline constexpr __fields __fields_floating_point{
    .__sign_                 = true,
    .__alternate_form_       = true,
    .__zero_padding_         = true,
    .__precision_            = true,
    .__locale_specific_form_ = true,
    .__type_                 = true};
inline constexpr __fields __fields_string{.__precision_ = true, .__type_ = true};
inline constexpr __fields __fields_pointer{.__type_ = true};

enum class _LIBCPP_ENUM_VIS __alignment : uint8_t {
  /// No alignment is set in the format string.
  __default,
  __left,
  __center,
  __right,
  __zero_padding
};

enum class _LIBCPP_ENUM_VIS __sign : uint8_t {
  /// No sign is set in the format string.
  ///
  /// The sign isn't allowed for certain format-types. By using this value
  /// it's possible to detect whether or not the user explicitly set the sign
  /// flag. For formatting purposes it behaves the same as \ref __minus.
  __default,
  __minus,
  __plus,
  __space
};

enum class _LIBCPP_ENUM_VIS __type : uint8_t {
  __default,
  __string,
  __binary_lower_case,
  __binary_upper_case,
  __octal,
  __decimal,
  __hexadecimal_lower_case,
  __hexadecimal_upper_case,
  __pointer,
  __char,
  __hexfloat_lower_case,
  __hexfloat_upper_case,
  __scientific_lower_case,
  __scientific_upper_case,
  __fixed_lower_case,
  __fixed_upper_case,
  __general_lower_case,
  __general_upper_case
};

struct __std {
  __alignment __alignment_ : 3;
  __sign __sign_ : 2;
  bool __alternate_form_ : 1;
  bool __locale_specific_form_ : 1;
  __type __type_;
};

struct __chrono {
  __alignment __alignment_ : 3;
  bool __weekday_name_ : 1;
  bool __month_name_ : 1;
};

/// Contains the parsed formatting specifications.
///
/// This contains information for both the std-format-spec and the
/// chrono-format-spec. This results in some unused members for both
/// specifications. However these unused members don't increase the size
/// of the structure.
///
/// This struct doesn't cross ABI boundaries so its layout doesn't need to be
/// kept stable.
template <class _CharT>
struct __parsed_specifications {
  union {
    // The field __alignment_ is the first element in __std_ and __chrono_.
    // This allows the code to always inspect this value regards which member
    // of the union is the active member [class.union.general]/2.
    //
    // This is needed since the generic output routines handle the alignment of
    // the output.
    __alignment __alignment_ : 3;
    __std __std_;
    __chrono __chrono_;
  };

  /// The requested width.
  ///
  /// When the format-spec used an arg-id for this field it has already been
  /// replaced with the value of that arg-id.
  int32_t __width_;

  /// The requested precision.
  ///
  /// When the format-spec used an arg-id for this field it has already been
  /// replaced with the value of that arg-id.
  int32_t __precision_;

  _CharT __fill_;

  _LIBCPP_HIDE_FROM_ABI constexpr bool __has_width() const { return __width_ > 0; }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __has_precision() const { return __precision_ >= 0; }
};

// Validate the struct is small and cheap to copy since the struct is passed by
// value in formatting functions.
static_assert(sizeof(__parsed_specifications<char>) == 16);
static_assert(is_trivially_copyable_v<__parsed_specifications<char>>);
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
static_assert(sizeof(__parsed_specifications<wchar_t>) == 16);
static_assert(is_trivially_copyable_v<__parsed_specifications<wchar_t>>);
#  endif

/// The parser for the std-format-spec.
///
/// Note this class is a member of std::formatter specializations. It's
/// expected developers will create their own formatter specializations that
/// inherit from the std::formatter specializations. This means this class
/// must be ABI stable. To aid the stability the unused bits in the class are
/// set to zero. That way they can be repurposed if a future revision of the
/// Standards adds new fields to std-format-spec.
template <class _CharT>
class _LIBCPP_TEMPLATE_VIS __parser {
public:
  _LIBCPP_HIDE_FROM_ABI constexpr auto __parse(basic_format_parse_context<_CharT>& __parse_ctx, __fields __fields)
      -> decltype(__parse_ctx.begin()) {

    const _CharT* __begin = __parse_ctx.begin();
    const _CharT* __end = __parse_ctx.end();
    if (__begin == __end)
      return __begin;

    if (__parse_fill_align(__begin, __end) && __begin == __end)
      return __begin;

    if (__fields.__sign_ && __parse_sign(__begin) && __begin == __end)
      return __begin;

    if (__fields.__alternate_form_ && __parse_alternate_form(__begin) && __begin == __end)
      return __begin;

    if (__fields.__zero_padding_ && __parse_zero_padding(__begin) && __begin == __end)
      return __begin;

    if (__parse_width(__begin, __end, __parse_ctx) && __begin == __end)
      return __begin;

    if (__fields.__precision_ && __parse_precision(__begin, __end, __parse_ctx) && __begin == __end)
      return __begin;

    if (__fields.__locale_specific_form_ && __parse_locale_specific_form(__begin) && __begin == __end)
      return __begin;

    if (__fields.__type_) {
      __parse_type(__begin);

      // When __type_ is false the calling parser is expected to do additional
      // parsing. In that case that parser should do the end of format string
      // validation.
      if (__begin != __end && *__begin != _CharT('}'))
        __throw_format_error("The format-spec should consume the input or end with a '}'");
    }

    return __begin;
  }

  /// \returns the `__parsed_specifications` with the resolved dynamic sizes..
  _LIBCPP_HIDE_FROM_ABI
  __parsed_specifications<_CharT> __get_parsed_std_specifications(auto& __ctx) const {
    return __parsed_specifications<_CharT>{
        .__std_ =
            __std{.__alignment_            = __alignment_,
                  .__sign_                 = __sign_,
                  .__alternate_form_       = __alternate_form_,
                  .__locale_specific_form_ = __locale_specific_form_,
                  .__type_                 = __type_},
        .__width_{__get_width(__ctx)},
        .__precision_{__get_precision(__ctx)},
        .__fill_{__fill_}};
  }

  __alignment __alignment_ : 3 {__alignment::__default};
  __sign __sign_ : 2 {__sign::__default};
  bool __alternate_form_ : 1 {false};
  bool __locale_specific_form_ : 1 {false};
  bool __reserved_0_ : 1 {false};
  __type __type_{__type::__default};

  // These two flags are used for formatting chrono. Since the struct has
  // padding space left it's added to this structure.
  bool __weekday_name_ : 1 {false};
  bool __month_name_ : 1 {false};

  uint8_t __reserved_1_ : 6 {0};
  uint8_t __reserved_2_ : 6 {0};
  // These two flags are only used internally and not part of the
  // __parsed_specifications. Therefore put them at the end.
  bool __width_as_arg_ : 1 {false};
  bool __precision_as_arg_ : 1 {false};

  /// The requested width, either the value or the arg-id.
  int32_t __width_{0};

  /// The requested precision, either the value or the arg-id.
  int32_t __precision_{-1};

  // LWG 3576 will probably change this to always accept a Unicode code point
  // To avoid changing the size with that change align the field so when it
  // becomes 32-bit its alignment will remain the same. That also means the
  // size will remain the same. (D2572 addresses the solution for LWG 3576.)
  _CharT __fill_{_CharT(' ')};

private:
  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_alignment(_CharT __c) {
    switch (__c) {
    case _CharT('<'):
      __alignment_ = __alignment::__left;
      return true;

    case _CharT('^'):
      __alignment_ = __alignment::__center;
      return true;

    case _CharT('>'):
      __alignment_ = __alignment::__right;
      return true;
    }
    return false;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_fill_align(const _CharT*& __begin, const _CharT* __end) {
    _LIBCPP_ASSERT(__begin != __end, "when called with an empty input the function will cause "
                                     "undefined behavior by evaluating data not in the input");
    if (__begin + 1 != __end) {
      if (__parse_alignment(*(__begin + 1))) {
        if (*__begin == _CharT('{') || *__begin == _CharT('}'))
          __throw_format_error("The format-spec fill field contains an invalid character");

        __fill_ = *__begin;
        __begin += 2;
        return true;
      }
    }

    if (!__parse_alignment(*__begin))
      return false;

    ++__begin;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_sign(const _CharT*& __begin) {
    switch (*__begin) {
    case _CharT('-'):
      __sign_ = __sign::__minus;
      break;
    case _CharT('+'):
      __sign_ = __sign::__plus;
      break;
    case _CharT(' '):
      __sign_ = __sign::__space;
      break;
    default:
      return false;
    }
    ++__begin;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_alternate_form(const _CharT*& __begin) {
    if (*__begin != _CharT('#'))
      return false;

    __alternate_form_ = true;
    ++__begin;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_zero_padding(const _CharT*& __begin) {
    if (*__begin != _CharT('0'))
      return false;

    if (__alignment_ == __alignment::__default)
      __alignment_ = __alignment::__zero_padding;
    ++__begin;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_width(const _CharT*& __begin, const _CharT* __end, auto& __parse_ctx) {
    if (*__begin == _CharT('0'))
      __throw_format_error("A format-spec width field shouldn't have a leading zero");

    if (*__begin == _CharT('{')) {
      __format::__parse_number_result __r = __format_spec::__parse_arg_id(++__begin, __end, __parse_ctx);
      __width_as_arg_ = true;
      __width_ = __r.__value;
      __begin = __r.__ptr;
      return true;
    }

    if (*__begin < _CharT('0') || *__begin > _CharT('9'))
      return false;

    __format::__parse_number_result __r = __format::__parse_number(__begin, __end);
    __width_ = __r.__value;
    _LIBCPP_ASSERT(__width_ != 0, "A zero value isn't allowed and should be impossible, "
                                  "due to validations in this function");
    __begin = __r.__ptr;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_precision(const _CharT*& __begin, const _CharT* __end,
                                                         auto& __parse_ctx) {
    if (*__begin != _CharT('.'))
      return false;

    ++__begin;
    if (__begin == __end)
      __throw_format_error("End of input while parsing format-spec precision");

    if (*__begin == _CharT('{')) {
      __format::__parse_number_result __arg_id = __format_spec::__parse_arg_id(++__begin, __end, __parse_ctx);
      __precision_as_arg_ = true;
      __precision_ = __arg_id.__value;
      __begin = __arg_id.__ptr;
      return true;
    }

    if (*__begin < _CharT('0') || *__begin > _CharT('9'))
      __throw_format_error("The format-spec precision field doesn't contain a value or arg-id");

    __format::__parse_number_result __r = __format::__parse_number(__begin, __end);
    __precision_ = __r.__value;
    __precision_as_arg_ = false;
    __begin = __r.__ptr;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __parse_locale_specific_form(const _CharT*& __begin) {
    if (*__begin != _CharT('L'))
      return false;

    __locale_specific_form_ = true;
    ++__begin;
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void __parse_type(const _CharT*& __begin) {
    // Determines the type. It does not validate whether the selected type is
    // valid. Most formatters have optional fields that are only allowed for
    // certain types. These parsers need to do validation after the type has
    // been parsed. So its easier to implement the validation for all types in
    // the specific parse function.
    switch (*__begin) {
    case 'A':
      __type_ = __type::__hexfloat_upper_case;
      break;
    case 'B':
      __type_ = __type::__binary_upper_case;
      break;
    case 'E':
      __type_ = __type::__scientific_upper_case;
      break;
    case 'F':
      __type_ = __type::__fixed_upper_case;
      break;
    case 'G':
      __type_ = __type::__general_upper_case;
      break;
    case 'X':
      __type_ = __type::__hexadecimal_upper_case;
      break;
    case 'a':
      __type_ = __type::__hexfloat_lower_case;
      break;
    case 'b':
      __type_ = __type::__binary_lower_case;
      break;
    case 'c':
      __type_ = __type::__char;
      break;
    case 'd':
      __type_ = __type::__decimal;
      break;
    case 'e':
      __type_ = __type::__scientific_lower_case;
      break;
    case 'f':
      __type_ = __type::__fixed_lower_case;
      break;
    case 'g':
      __type_ = __type::__general_lower_case;
      break;
    case 'o':
      __type_ = __type::__octal;
      break;
    case 'p':
      __type_ = __type::__pointer;
      break;
    case 's':
      __type_ = __type::__string;
      break;
    case 'x':
      __type_ = __type::__hexadecimal_lower_case;
      break;
    default:
      return;
    }
    ++__begin;
  }

  _LIBCPP_HIDE_FROM_ABI
  int32_t __get_width(auto& __ctx) const {
    if (!__width_as_arg_)
      return __width_;

    int32_t __result = __format_spec::__substitute_arg_id(__ctx.arg(__width_));
    if (__result == 0)
      __throw_format_error("A format-spec width field replacement should have a positive value");
    return __result;
  }

  _LIBCPP_HIDE_FROM_ABI
  int32_t __get_precision(auto& __ctx) const {
    if (!__precision_as_arg_)
      return __precision_;

    return __format_spec::__substitute_arg_id(__ctx.arg(__precision_));
  }
};

// Validates whether the reserved bitfields don't change the size.
static_assert(sizeof(__parser<char>) == 16);
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
static_assert(sizeof(__parser<wchar_t>) == 16);
#  endif

_LIBCPP_HIDE_FROM_ABI constexpr void __process_display_type_string(__format_spec::__type __type) {
  switch (__type) {
  case __format_spec::__type::__default:
  case __format_spec::__type::__string:
    break;

  default:
    std::__throw_format_error("The format-spec type has a type not supported for a string argument");
  }
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr void __process_display_type_bool_string(__parser<_CharT>& __parser) {
  if (__parser.__sign_ != __sign::__default)
    std::__throw_format_error("A sign field isn't allowed in this format-spec");

  if (__parser.__alternate_form_)
    std::__throw_format_error("An alternate form field isn't allowed in this format-spec");

  if (__parser.__alignment_ == __alignment::__zero_padding)
    std::__throw_format_error("A zero-padding field isn't allowed in this format-spec");

  if (__parser.__alignment_ == __alignment::__default)
    __parser.__alignment_ = __alignment::__left;
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr void __process_display_type_char(__parser<_CharT>& __parser) {
  __format_spec::__process_display_type_bool_string(__parser);
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr void __process_parsed_bool(__parser<_CharT>& __parser) {
  switch (__parser.__type_) {
  case __format_spec::__type::__default:
  case __format_spec::__type::__string:
    __format_spec::__process_display_type_bool_string(__parser);
    break;

  case __format_spec::__type::__binary_lower_case:
  case __format_spec::__type::__binary_upper_case:
  case __format_spec::__type::__octal:
  case __format_spec::__type::__decimal:
  case __format_spec::__type::__hexadecimal_lower_case:
  case __format_spec::__type::__hexadecimal_upper_case:
    break;

  default:
    std::__throw_format_error("The format-spec type has a type not supported for a bool argument");
  }
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr void __process_parsed_char(__parser<_CharT>& __parser) {
  switch (__parser.__type_) {
  case __format_spec::__type::__default:
  case __format_spec::__type::__char:
    __format_spec::__process_display_type_char(__parser);
    break;

  case __format_spec::__type::__binary_lower_case:
  case __format_spec::__type::__binary_upper_case:
  case __format_spec::__type::__octal:
  case __format_spec::__type::__decimal:
  case __format_spec::__type::__hexadecimal_lower_case:
  case __format_spec::__type::__hexadecimal_upper_case:
    break;

  default:
    std::__throw_format_error("The format-spec type has a type not supported for a char argument");
  }
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr void __process_parsed_integer(__parser<_CharT>& __parser) {
  switch (__parser.__type_) {
  case __format_spec::__type::__default:
  case __format_spec::__type::__binary_lower_case:
  case __format_spec::__type::__binary_upper_case:
  case __format_spec::__type::__octal:
  case __format_spec::__type::__decimal:
  case __format_spec::__type::__hexadecimal_lower_case:
  case __format_spec::__type::__hexadecimal_upper_case:
    break;

  case __format_spec::__type::__char:
    __format_spec::__process_display_type_char(__parser);
    break;

  default:
    std::__throw_format_error("The format-spec type has a type not supported for an integer argument");
  }
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr void __process_parsed_floating_point(__parser<_CharT>& __parser) {
  switch (__parser.__type_) {
  case __format_spec::__type::__default:
    // When no precision specified then it keeps default since that
    // formatting differs from the other types.
    if (__parser.__precision_as_arg_ || __parser.__precision_ != -1)
      __parser.__type_ = __format_spec::__type::__general_lower_case;
    break;
  case __format_spec::__type::__hexfloat_lower_case:
  case __format_spec::__type::__hexfloat_upper_case:
    // Precision specific behavior will be handled later.
    break;
  case __format_spec::__type::__scientific_lower_case:
  case __format_spec::__type::__scientific_upper_case:
  case __format_spec::__type::__fixed_lower_case:
  case __format_spec::__type::__fixed_upper_case:
  case __format_spec::__type::__general_lower_case:
  case __format_spec::__type::__general_upper_case:
    if (!__parser.__precision_as_arg_ && __parser.__precision_ == -1)
      // Set the default precision for the call to to_chars.
      __parser.__precision_ = 6;
    break;

  default:
    std::__throw_format_error("The format-spec type has a type not supported for a floating-point argument");
  }
}

_LIBCPP_HIDE_FROM_ABI constexpr void __process_display_type_pointer(__format_spec::__type __type) {
  switch (__type) {
  case __format_spec::__type::__default:
  case __format_spec::__type::__pointer:
    break;

  default:
    std::__throw_format_error("The format-spec type has a type not supported for a pointer argument");
  }
}

} // namespace __format_spec

#endif //_LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_H
