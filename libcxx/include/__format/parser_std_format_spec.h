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

#include <__config>
#include <__debug>
#include <__format/format_arg.h>
#include <__format/format_error.h>
#include <__format/format_string.h>
#include <__variant/monostate.h>
#include <concepts>
#include <cstdint>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
# pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

// TODO FMT Remove this once we require compilers with proper C++20 support.
// If the compiler has no concepts support, the format header will be disabled.
// Without concepts support enable_if needs to be used and that too much effort
// to support compilers with partial C++20 support.
# if !defined(_LIBCPP_HAS_NO_CONCEPTS)

namespace __format_spec {

/**
 * Contains the flags for the std-format-spec.
 *
 * Some format-options can only be used for specific C++types and may depend on
 * the selected format-type.
 * * The C++type filtering can be done using the proper policies for
 *   @ref __parser_std.
 * * The format-type filtering needs to be done post parsing in the parser
 *   derived from @ref __parser_std.
 */
class _LIBCPP_TYPE_VIS _Flags {
public:
  enum class _LIBCPP_ENUM_VIS _Alignment : uint8_t {
    /**
     * No alignment is set in the format string.
     *
     * Zero-padding is ignored when an alignment is selected.
     * The default alignment depends on the selected format-type.
     */
    __default,
    __left,
    __center,
    __right
  };
  enum class _LIBCPP_ENUM_VIS _Sign : uint8_t {
    /**
     * No sign is set in the format string.
     *
     * The sign isn't allowed for certain format-types. By using this value
     * it's possible to detect whether or not the user explicitly set the sign
     * flag. For formatting purposes it behaves the same as @ref __minus.
     */
    __default,
    __minus,
    __plus,
    __space
  };

  _Alignment __alignment : 2 {_Alignment::__default};
  _Sign __sign : 2 {_Sign::__default};
  uint8_t __alternate_form : 1 {false};
  uint8_t __zero_padding : 1 {false};
  uint8_t __locale_specific_form : 1 {false};

  enum class _LIBCPP_ENUM_VIS _Type : uint8_t {
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
    __float_hexadecimal_lower_case,
    __float_hexadecimal_upper_case,
    __scientific_lower_case,
    __scientific_upper_case,
    __fixed_lower_case,
    __fixed_upper_case,
    __general_lower_case,
    __general_upper_case
  };

  _Type __type{_Type::__default};
};

namespace __detail {
template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr bool
__parse_alignment(_CharT __c, _Flags& __flags) noexcept {
  switch (__c) {
  case _CharT('<'):
    __flags.__alignment = _Flags::_Alignment::__left;
    return true;

  case _CharT('^'):
    __flags.__alignment = _Flags::_Alignment::__center;
    return true;

  case _CharT('>'):
    __flags.__alignment = _Flags::_Alignment::__right;
    return true;
  }
  return false;
}
} // namespace __detail

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS __parser_fill_align {
public:
  // TODO FMT The standard doesn't specify this character is a Unicode
  // character. Validate what fmt and MSVC have implemented.
  _CharT __fill{_CharT(' ')};

protected:
  _LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
  __parse(const _CharT* __begin, const _CharT* __end, _Flags& __flags) {
    _LIBCPP_ASSERT(__begin != __end,
                   "When called with an empty input the function will cause "
                   "undefined behavior by evaluating data not in the input");
    if (__begin + 1 != __end) {
      if (__detail::__parse_alignment(*(__begin + 1), __flags)) {
        if (*__begin == _CharT('{') || *__begin == _CharT('}'))
          __throw_format_error(
              "The format-spec fill field contains an invalid character");
        __fill = *__begin;
        return __begin + 2;
      }
    }

    if (__detail::__parse_alignment(*__begin, __flags))
      return __begin + 1;

    return __begin;
  }
};

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__parse_sign(const _CharT* __begin, _Flags& __flags) noexcept {
  switch (*__begin) {
  case _CharT('-'):
    __flags.__sign = _Flags::_Sign::__minus;
    break;
  case _CharT('+'):
    __flags.__sign = _Flags::_Sign::__plus;
    break;
  case _CharT(' '):
    __flags.__sign = _Flags::_Sign::__space;
    break;
  default:
    return __begin;
  }
  return __begin + 1;
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__parse_alternate_form(const _CharT* __begin, _Flags& __flags) noexcept {
  if (*__begin == _CharT('#')) {
    __flags.__alternate_form = true;
    ++__begin;
  }

  return __begin;
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__parse_zero_padding(const _CharT* __begin, _Flags& __flags) noexcept {
  if (*__begin == _CharT('0')) {
    __flags.__zero_padding = true;
    ++__begin;
  }

  return __begin;
}

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
    __throw_format_error("A format-spec arg-id should terminate at a '}'");

  ++__r.__ptr;
  return __r;
}

template <class _Context>
_LIBCPP_HIDE_FROM_ABI constexpr uint32_t
__substitute_arg_id(basic_format_arg<_Context> __arg) {
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
      __arg);
}

class _LIBCPP_TYPE_VIS __parser_width {
public:
  /** Contains a width or an arg-id. */
  uint32_t __width : 31 {0};
  /** Determines whether the value stored is a width or an arg-id. */
  uint32_t __width_as_arg : 1 {0};

protected:
  /**
   * Does the supplied std-format-spec contain a width field?
   *
   * When the field isn't present there's no padding required. This can be used
   * to optimize the formatting.
   */
  constexpr bool __has_width_field() const noexcept {
    return __width_as_arg || __width;
  }

  /**
   * Does the supplied width field contain an arg-id?
   *
   * If @c true the formatter needs to call @ref __substitute_width_arg_id.
   */
  constexpr bool __width_needs_substitution() const noexcept {
    return __width_as_arg;
  }

  template <class _CharT>
  _LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
  __parse(const _CharT* __begin, const _CharT* __end, auto& __parse_ctx) {
    if (*__begin == _CharT('0'))
      __throw_format_error(
          "A format-spec width field shouldn't have a leading zero");

    if (*__begin == _CharT('{')) {
      __format::__parse_number_result __r =
          __parse_arg_id(++__begin, __end, __parse_ctx);
      __width = __r.__value;
      __width_as_arg = 1;
      return __r.__ptr;
    }

    if (*__begin < _CharT('0') || *__begin > _CharT('9'))
      return __begin;

    __format::__parse_number_result __r =
        __format::__parse_number(__begin, __end);
    __width = __r.__value;
    _LIBCPP_ASSERT(__width != 0,
                   "A zero value isn't allowed and should be impossible, "
                   "due to validations in this function");
    return __r.__ptr;
  }

  void _LIBCPP_HIDE_FROM_ABI constexpr __substitute_width_arg_id(auto __arg) {
    _LIBCPP_ASSERT(__width_as_arg == 1,
                   "Substitute width called when no substitution is required");

    // The clearing of the flag isn't required but looks better when debugging
    // the code.
    __width_as_arg = 0;
    __width = __substitute_arg_id(__arg);
    if (__width == 0)
      __throw_format_error(
          "A format-spec width field replacement should have a positive value");
  }
};

class _LIBCPP_TYPE_VIS __parser_precision {
public:
  /** Contains a precision or an arg-id. */
  uint32_t __precision : 31 {__format::__number_max};
  /**
   * Determines whether the value stored is a precision or an arg-id.
   *
   * @note Since @ref __precision == @ref __format::__number_max is a valid
   * value, the default value contains an arg-id of INT32_MAX. (This number of
   * arguments isn't supported by compilers.)  This is used to detect whether
   * the std-format-spec contains a precision field.
   */
  uint32_t __precision_as_arg : 1 {1};

protected:
  /**
   * Does the supplied std-format-spec contain a precision field?
   *
   * When the field isn't present there's no truncating required. This can be
   * used to optimize the formatting.
   */
  constexpr bool __has_precision_field() const noexcept {

    return __precision_as_arg == 0 ||             // Contains a value?
           __precision != __format::__number_max; // The arg-id is valid?
  }

  /**
   * Does the supplied precision field contain an arg-id?
   *
   * If @c true the formatter needs to call @ref __substitute_precision_arg_id.
   */
  constexpr bool __precision_needs_substitution() const noexcept {
    return __precision_as_arg && __precision != __format::__number_max;
  }

  template <class _CharT>
  _LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
  __parse(const _CharT* __begin, const _CharT* __end, auto& __parse_ctx) {
    if (*__begin != _CharT('.'))
      return __begin;

    ++__begin;
    if (__begin == __end)
      __throw_format_error("End of input while parsing format-spec precision");

    if (*__begin == _CharT('0')) {
      ++__begin;
      if (__begin != __end && *__begin >= '0' && *__begin <= '9')
        __throw_format_error(
            "A format-spec precision field shouldn't have a leading zero");

      __precision = 0;
      __precision_as_arg = 0;
      return __begin;
    }

    if (*__begin == _CharT('{')) {
      __format::__parse_number_result __arg_id =
          __parse_arg_id(++__begin, __end, __parse_ctx);
      _LIBCPP_ASSERT(__arg_id.__value != __format::__number_max,
                     "Unsupported number of arguments, since this number of "
                     "arguments is used a special value");
      __precision = __arg_id.__value;
      return __arg_id.__ptr;
    }

    if (*__begin < _CharT('0') || *__begin > _CharT('9'))
      __throw_format_error(
          "The format-spec precision field doesn't contain a value or arg-id");

    __format::__parse_number_result __r =
        __format::__parse_number(__begin, __end);
    __precision = __r.__value;
    __precision_as_arg = 0;
    return __r.__ptr;
  }

  void _LIBCPP_HIDE_FROM_ABI constexpr __substitute_precision_arg_id(
      auto __arg) {
    _LIBCPP_ASSERT(
        __precision_as_arg == 1 && __precision != __format::__number_max,
        "Substitute precision called when no substitution is required");

    // The clearing of the flag isn't required but looks better when debugging
    // the code.
    __precision_as_arg = 0;
    __precision = __substitute_arg_id(__arg);
  }
};

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__parse_locale_specific_form(const _CharT* __begin, _Flags& __flags) noexcept {
  if (*__begin == _CharT('L')) {
    __flags.__locale_specific_form = true;
    ++__begin;
  }

  return __begin;
}

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr const _CharT*
__parse_type(const _CharT* __begin, _Flags& __flags) {

  // Determines the type. It does not validate whether the selected type is
  // valid. Most formatters have optional fields that are only allowed for
  // certain types. These parsers need to do validation after the type has
  // been parsed. So its easier to implement the validation for all types in
  // the specific parse function.
  switch (*__begin) {
  case 'A':
    __flags.__type = _Flags::_Type::__float_hexadecimal_upper_case;
    break;
  case 'B':
    __flags.__type = _Flags::_Type::__binary_upper_case;
    break;
  case 'E':
    __flags.__type = _Flags::_Type::__scientific_upper_case;
    break;
  case 'F':
    __flags.__type = _Flags::_Type::__fixed_upper_case;
    break;
  case 'G':
    __flags.__type = _Flags::_Type::__general_upper_case;
    break;
  case 'X':
    __flags.__type = _Flags::_Type::__hexadecimal_upper_case;
    break;
  case 'a':
    __flags.__type = _Flags::_Type::__float_hexadecimal_lower_case;
    break;
  case 'b':
    __flags.__type = _Flags::_Type::__binary_lower_case;
    break;
  case 'c':
    __flags.__type = _Flags::_Type::__char;
    break;
  case 'd':
    __flags.__type = _Flags::_Type::__decimal;
    break;
  case 'e':
    __flags.__type = _Flags::_Type::__scientific_lower_case;
    break;
  case 'f':
    __flags.__type = _Flags::_Type::__fixed_lower_case;
    break;
  case 'g':
    __flags.__type = _Flags::_Type::__general_lower_case;
    break;
  case 'o':
    __flags.__type = _Flags::_Type::__octal;
    break;
  case 'p':
    __flags.__type = _Flags::_Type::__pointer;
    break;
  case 's':
    __flags.__type = _Flags::_Type::__string;
    break;
  case 'x':
    __flags.__type = _Flags::_Type::__hexadecimal_lower_case;
    break;
  default:
    return __begin;
  }
  return ++__begin;
}

/**
 * The parser for the std-format-spec.
 *
 * [format.string.std]/1 specifies the std-format-spec:
 *   fill-and-align sign # 0 width precision L type
 *
 * All these fields are optional. Whether these fields can be used depend on:
 * - The type supplied to the format string.
 *   E.g. A string never uses the sign field so the field may not be set.
 *   This constrain is validated by the parsers in this file.
 * - The supplied value for the optional type field.
 *   E.g. A int formatted as decimal uses the sign field.
 *   When formatted as a char the sign field may no longer be set.
 *   This constrain isn't validated by the parsers in this file.
 *
 * The base classes are ordered to minimize the amount of padding.
 *
 * This implements the parser for the string types.
 */
template <class _CharT>
class _LIBCPP_TEMPLATE_VIS __parser_string
    : public __parser_width,              // provides __width(|as_arg)
      public __parser_precision,          // provides __precision(|as_arg)
      public __parser_fill_align<_CharT>, // provides __fill and uses __flags
      public _Flags                       // provides __flags
{
public:
  using char_type = _CharT;

  _LIBCPP_HIDE_FROM_ABI constexpr __parser_string() {
    this->__alignment = _Flags::_Alignment::__left;
  }

  /**
   * The low-level std-format-spec parse function.
   *
   * @pre __begin points at the beginning of the std-format-spec. This means
   * directly after the ':'.
   * @pre The std-format-spec parses the entire input, or the first unmatched
   * character is a '}'.
   *
   * @returns The iterator pointing at the last parsed character.
   */
  _LIBCPP_HIDE_FROM_ABI constexpr auto parse(auto& __parse_ctx)
      -> decltype(__parse_ctx.begin()) {
    auto __it = __parse(__parse_ctx);
    __process_display_type();
    return __it;
  }

private:
  /**
   * Parses the std-format-spec.
   *
   * @throws __throw_format_error When @a __parse_ctx contains an ill-formed
   *                               std-format-spec.
   *
   * @returns An iterator to the end of input or point at the closing '}'.
   */
  _LIBCPP_HIDE_FROM_ABI constexpr auto __parse(auto& __parse_ctx)
      -> decltype(__parse_ctx.begin()) {

    auto __begin = __parse_ctx.begin();
    auto __end = __parse_ctx.end();
    if (__begin == __end)
      return __begin;

    __begin = __parser_fill_align<_CharT>::__parse(__begin, __end,
                                                   static_cast<_Flags&>(*this));
    if (__begin == __end)
      return __begin;

    __begin = __parser_width::__parse(__begin, __end, __parse_ctx);
    if (__begin == __end)
      return __begin;

    __begin = __parser_precision::__parse(__begin, __end, __parse_ctx);
    if (__begin == __end)
      return __begin;

    __begin = __parse_type(__begin, static_cast<_Flags&>(*this));

    if (__begin != __end && *__begin != _CharT('}'))
      __throw_format_error(
          "The format-spec should consume the input or end with a '}'");

    return __begin;
  }

  /** Processes the parsed std-format-spec based on the parsed display type. */
  void _LIBCPP_HIDE_FROM_ABI constexpr __process_display_type() {
    switch (this->__type) {
    case _Flags::_Type::__default:
    case _Flags::_Type::__string:
      break;

    default:
      __throw_format_error("The format-spec type has a type not supported for "
                           "a string argument");
    }
  }
};

/**
 * The parser for the std-format-spec.
 *
 * This implements the parser for the integral types. This includes the
 * character type and boolean type.
 *
 * See @ref __parser_string.
 */
template <class _CharT>
class _LIBCPP_TEMPLATE_VIS __parser_integral
    : public __parser_width,              // provides __width(|as_arg)
      public __parser_fill_align<_CharT>, // provides __fill and uses __flags
      public _Flags                       // provides __flags
{
public:
  using char_type = _CharT;

  // TODO FMT This class probably doesn't need public member functions after
  // format.string.std/std_format_spec_integral.pass.cpp has been retired.

  /**
   * The low-level std-format-spec parse function.
   *
   * @pre __begin points at the beginning of the std-format-spec. This means
   * directly after the ':'.
   * @pre The std-format-spec parses the entire input, or the first unmatched
   * character is a '}'.
   *
   * @returns The iterator pointing at the last parsed character.
   */
  _LIBCPP_HIDE_FROM_ABI constexpr auto parse(auto& __parse_ctx)
      -> decltype(__parse_ctx.begin()) {
    auto __begin = __parse_ctx.begin();
    auto __end = __parse_ctx.end();
    if (__begin == __end)
      return __begin;

    __begin = __parser_fill_align<_CharT>::__parse(__begin, __end,
                                                   static_cast<_Flags&>(*this));
    if (__begin == __end)
      return __begin;

    __begin = __parse_sign(__begin, static_cast<_Flags&>(*this));
    if (__begin == __end)
      return __begin;

    __begin = __parse_alternate_form(__begin, static_cast<_Flags&>(*this));
    if (__begin == __end)
      return __begin;

    __begin = __parse_zero_padding(__begin, static_cast<_Flags&>(*this));
    if (__begin == __end)
      return __begin;

    __begin = __parser_width::__parse(__begin, __end, __parse_ctx);
    if (__begin == __end)
      return __begin;

    __begin =
        __parse_locale_specific_form(__begin, static_cast<_Flags&>(*this));
    if (__begin == __end)
      return __begin;

    __begin = __parse_type(__begin, static_cast<_Flags&>(*this));

    if (__begin != __end && *__begin != _CharT('}'))
      __throw_format_error(
          "The format-spec should consume the input or end with a '}'");

    return __begin;
  }

protected:
  /**
   * Handles the post-parsing updates for the integer types.
   *
   * Updates the zero-padding and alignment for integer types.
   *
   * [format.string.std]/13
   *   If the 0 character and an align option both appear, the 0 character is
   *   ignored.
   *
   * For the formatter a @ref __default alignment means zero-padding. Update
   * the alignment based on parsed format string.
   */
  _LIBCPP_HIDE_FROM_ABI constexpr void __handle_integer() noexcept {
    this->__zero_padding &= this->__alignment == _Flags::_Alignment::__default;
    if (!this->__zero_padding &&
        this->__alignment == _Flags::_Alignment::__default)
      this->__alignment = _Flags::_Alignment::__right;
  }

  /**
   * Handles the post-parsing updates for the character types.
   *
   * Sets the alignment and validates the format flags set for a character type.
   *
   * At the moment the validation for a character and a Boolean behave the
   * same, but this may change in the future.
   * Specifically at the moment the locale-specific form is allowed for the
   * char output type, but it has no effect on the output.
   */
  _LIBCPP_HIDE_FROM_ABI constexpr void __handle_char() { __handle_bool(); }

  /**
   * Handles the post-parsing updates for the Boolean types.
   *
   * Sets the alignment and validates the format flags set for a Boolean type.
   */
  _LIBCPP_HIDE_FROM_ABI constexpr void __handle_bool() {
    if (this->__sign != _Flags::_Sign::__default)
      __throw_format_error("A sign field isn't allowed in this format-spec");

    if (this->__alternate_form)
      __throw_format_error(
          "An alternate form field isn't allowed in this format-spec");

    if (this->__zero_padding)
      __throw_format_error(
          "A zero-padding field isn't allowed in this format-spec");

    if (this->__alignment == _Flags::_Alignment::__default)
      this->__alignment = _Flags::_Alignment::__left;
  }
};

// TODO FMT Add a parser for floating-point values.
// TODO FMT Add a parser for pointer values.

} // namespace __format_spec

# endif // !defined(_LIBCPP_HAS_NO_CONCEPTS)

#endif //_LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FORMAT_PARSER_STD_FORMAT_SPEC_H
