// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FORMAT_ARG_STORE_H
#define _LIBCPP___FORMAT_FORMAT_ARG_STORE_H

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#include <__concepts/arithmetic.h>
#include <__concepts/same_as.h>
#include <__concepts/semiregular.h>
#include <__config>
#include <__format/concepts.h>
#include <__format/format_arg.h>
#include <__format/format_parse_context.h>
#include <__type_traits/conditional.h>
#include <__type_traits/extent.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/remove_const.h>
#include <cstdint>
#include <string>
#include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

namespace __format {

/// \returns The @c __arg_t based on the type of the formatting argument.
///
/// \pre \c __formattable<_Tp, typename _Context::char_type>
template <class _Context, class _Tp>
consteval __arg_t __determine_arg_t();

// Boolean
template <class, same_as<bool> _Tp>
consteval __arg_t __determine_arg_t() {
  return __arg_t::__boolean;
}

// Char
template <class _Context, same_as<typename _Context::char_type> _Tp>
consteval __arg_t __determine_arg_t() {
  return __arg_t::__char_type;
}
#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <class _Context, class _CharT>
  requires(same_as<typename _Context::char_type, wchar_t> && same_as<_CharT, char>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__char_type;
}
#  endif

// Signed integers
template <class, __libcpp_signed_integer _Tp>
consteval __arg_t __determine_arg_t() {
  if constexpr (sizeof(_Tp) <= sizeof(int))
    return __arg_t::__int;
  else if constexpr (sizeof(_Tp) <= sizeof(long long))
    return __arg_t::__long_long;
#  if _LIBCPP_HAS_INT128
  else if constexpr (sizeof(_Tp) == sizeof(__int128_t))
    return __arg_t::__i128;
#  endif
  else
    static_assert(sizeof(_Tp) == 0, "an unsupported signed integer was used");
}

// Unsigned integers
template <class, __libcpp_unsigned_integer _Tp>
consteval __arg_t __determine_arg_t() {
  if constexpr (sizeof(_Tp) <= sizeof(unsigned))
    return __arg_t::__unsigned;
  else if constexpr (sizeof(_Tp) <= sizeof(unsigned long long))
    return __arg_t::__unsigned_long_long;
#  if _LIBCPP_HAS_INT128
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
    return __arg_t::__u128;
#  endif
  else
    static_assert(sizeof(_Tp) == 0, "an unsupported unsigned integer was used");
}

// Floating-point
template <class, same_as<float> _Tp>
consteval __arg_t __determine_arg_t() {
  return __arg_t::__float;
}
template <class, same_as<double> _Tp>
consteval __arg_t __determine_arg_t() {
  return __arg_t::__double;
}
template <class, same_as<long double> _Tp>
consteval __arg_t __determine_arg_t() {
  return __arg_t::__long_double;
}

// Char pointer
template <class _Context, class _Tp>
  requires(same_as<typename _Context::char_type*, _Tp> || same_as<const typename _Context::char_type*, _Tp>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__const_char_type_ptr;
}

// Char array
template <class _Context, class _Tp>
  requires(is_array_v<_Tp> && same_as<_Tp, typename _Context::char_type[extent_v<_Tp>]>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__string_view;
}

// String view
template <class _Context, class _Tp>
  requires(same_as<typename _Context::char_type, typename _Tp::value_type> &&
           same_as<_Tp, basic_string_view<typename _Tp::value_type, typename _Tp::traits_type>>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__string_view;
}

// String
template <class _Context, class _Tp>
  requires(
      same_as<typename _Context::char_type, typename _Tp::value_type> &&
      same_as<_Tp, basic_string<typename _Tp::value_type, typename _Tp::traits_type, typename _Tp::allocator_type>>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__string_view;
}

// Pointers
template <class, class _Ptr>
  requires(same_as<_Ptr, void*> || same_as<_Ptr, const void*> || same_as<_Ptr, nullptr_t>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__ptr;
}

// Handle
//
// Note this version can't be constrained avoiding ambiguous overloads.
// That means it can be instantiated by disabled formatters. To solve this, a
// constrained version for not formattable formatters is added.
template <class _Context, class _Tp>
consteval __arg_t __determine_arg_t() {
  return __arg_t::__handle;
}

// The overload for not formattable types allows triggering the static
// assertion below.
template <class _Context, class _Tp>
  requires(!__formattable_with<_Tp, _Context>)
consteval __arg_t __determine_arg_t() {
  return __arg_t::__none;
}

// Pseudo constuctor for basic_format_arg
//
// Modeled after template<class T> explicit basic_format_arg(T& v) noexcept;
// [format.arg]/4-6
template <class _Context, __formattable_with<_Context> _Tp>
_LIBCPP_HIDE_FROM_ABI basic_format_arg<_Context> __create_format_arg(_Tp& __value) noexcept {
  using _Dp               = remove_const_t<_Tp>;
  constexpr __arg_t __arg = __format::__determine_arg_t<_Context, _Dp>();
  static_assert(__arg != __arg_t::__none, "the supplied type is not formattable");

  // Not all types can be used to directly initialize the
  // __basic_format_arg_value.  First handle all types needing adjustment, the
  // final else requires no adjustment.
  if constexpr (__arg == __arg_t::__char_type)

#  if _LIBCPP_HAS_WIDE_CHARACTERS
    if constexpr (same_as<typename _Context::char_type, wchar_t> && same_as<_Dp, char>)
      return basic_format_arg<_Context>{__arg, static_cast<wchar_t>(static_cast<unsigned char>(__value))};
    else
#  endif
      return basic_format_arg<_Context>{__arg, __value};
  else if constexpr (__arg == __arg_t::__int)
    return basic_format_arg<_Context>{__arg, static_cast<int>(__value)};
  else if constexpr (__arg == __arg_t::__long_long)
    return basic_format_arg<_Context>{__arg, static_cast<long long>(__value)};
  else if constexpr (__arg == __arg_t::__unsigned)
    return basic_format_arg<_Context>{__arg, static_cast<unsigned>(__value)};
  else if constexpr (__arg == __arg_t::__unsigned_long_long)
    return basic_format_arg<_Context>{__arg, static_cast<unsigned long long>(__value)};
  else if constexpr (__arg == __arg_t::__string_view)
    // Using std::size on a character array will add the NUL-terminator to the size.
    if constexpr (is_array_v<_Dp>)
      return basic_format_arg<_Context>{
          __arg, basic_string_view<typename _Context::char_type>{__value, extent_v<_Dp> - 1}};
    else
      // When the _Traits or _Allocator are different an implicit conversion will
      // fail.
      return basic_format_arg<_Context>{
          __arg, basic_string_view<typename _Context::char_type>{__value.data(), __value.size()}};
  else if constexpr (__arg == __arg_t::__ptr)
    return basic_format_arg<_Context>{__arg, static_cast<const void*>(__value)};
  else if constexpr (__arg == __arg_t::__handle)
    return basic_format_arg<_Context>{__arg, typename __basic_format_arg_value<_Context>::__handle{__value}};
  else
    return basic_format_arg<_Context>{__arg, __value};
}

// Helper function that issues a diagnostic when __formattable_with<_Tp, _Context> is false.
//
// Since it's quite easy to make a mistake writing a formatter specialization
// this function tries to give a better explanation. This should improve the
// diagnostics when trying to format type that has no properly specialized
// formatter.
template <class _Context, class _Tp>
[[noreturn]] _LIBCPP_HIDE_FROM_ABI constexpr void __diagnose_invalid_formatter() {
  using _Formatter = typename _Context::template formatter_type<remove_const_t<_Tp>>;
  constexpr bool __is_disabled =
      !is_default_constructible_v<_Formatter> && !is_copy_constructible_v<_Formatter> &&
      !is_move_constructible_v<_Formatter> && !is_copy_assignable_v<_Formatter> && !is_move_assignable_v<_Formatter>;
  constexpr bool __is_semiregular = semiregular<_Formatter>;

  constexpr bool __has_parse_function =
      requires(_Formatter& __f, basic_format_parse_context<typename _Context::char_type> __pc) {
        { __f.parse(__pc) };
      };
  constexpr bool __correct_parse_function_return_type =
      requires(_Formatter& __f, basic_format_parse_context<typename _Context::char_type> __pc) {
        { __f.parse(__pc) } -> same_as<typename decltype(__pc)::iterator>;
      };

  // The reason these static_asserts are placed in an if-constexpr-chain is to
  // only show one error. For example, when the formatter is not specialized it
  // would show all static_assert messages. With this chain the compiler only
  // evaluates one static_assert.

  if constexpr (__is_disabled)
    static_assert(false, "The required formatter specialization has not been provided.");
  else if constexpr (!__is_semiregular)
    static_assert(false, "The required formatter specialization is not semiregular.");

  else if constexpr (!__has_parse_function)
    static_assert(
        false, "The required formatter specialization does not have a parse function taking the proper arguments.");
  else if constexpr (!__correct_parse_function_return_type)
    static_assert(false, "The required formatter specialization's parse function does not return the required type.");

  else {
    // During constant evaluation this function is called, but the format
    // member function has not been evaluated. This means these functions
    // can't be evaluated at that time.
    //
    // Note this else branch should never been taken during constant
    // eveluation, the static_asserts in one of the branches above should
    // trigger.
    constexpr bool __has_format_function = requires(_Formatter& __f, _Tp&& __t, _Context __fc) {
      { __f.format(__t, __fc) };
    };
    constexpr bool __is_format_function_const_qualified = requires(const _Formatter& __cf, _Tp&& __t, _Context __fc) {
      { __cf.format(__t, __fc) };
    };
    constexpr bool __correct_format_function_return_type = requires(_Formatter& __f, _Tp&& __t, _Context __fc) {
      { __f.format(__t, __fc)->same_as<typename _Context::iterator> };
    };

    if constexpr (!__has_format_function)
      static_assert(
          false, "The required formatter specialization does not have a format function taking the proper arguments.");
    else if constexpr (!__is_format_function_const_qualified)
      static_assert(false, "The required formatter specialization's format function is not const qualified.");
    else if constexpr (!__correct_format_function_return_type)
      static_assert(
          false, "The required formatter specialization's format function does not return the required type.");

    else
      // This should not happen; it makes sure the code is ill-formed.
      static_assert(false, "The required formatter specialization is not formattable with its context.");
  }
}

// The Psuedo constructor is constrained per [format.arg]/4.
// The only way for a user to call __create_format_arg is from std::make_format_args.
// Per [format.arg.store]/3
//   template<class Context = format_context, class... Args>
//     format-arg-store<Context, Args...> make_format_args(Args&... fmt_args);
//
//   Preconditions: The type typename Context::template formatter_type<remove_const_t<Ti>>
//   meets the BasicFormatter requirements ([formatter.requirements]) for each Ti in Args.
//
// This function is only called when the precondition is violated.
// Without this function the compilation would fail since the overload is
// SFINAE'ed away. This function is used to provide better diagnostics.
template <class _Context, class _Tp>
_LIBCPP_HIDE_FROM_ABI basic_format_arg<_Context> __create_format_arg(_Tp&) noexcept {
  __format::__diagnose_invalid_formatter<_Context, _Tp>();
}

template <class _Context, class... _Args>
_LIBCPP_HIDE_FROM_ABI void
__create_packed_storage(uint64_t& __types, __basic_format_arg_value<_Context>* __values, _Args&... __args) noexcept {
  int __shift = 0;
  (
      [&] {
        basic_format_arg<_Context> __arg = __format::__create_format_arg<_Context>(__args);
        if (__shift != 0)
          __types |= static_cast<uint64_t>(__arg.__type_) << __shift;
        else
          // Assigns the initial value.
          __types = static_cast<uint64_t>(__arg.__type_);
        __shift += __packed_arg_t_bits;
        *__values++ = __arg.__value_;
      }(),
      ...);
}

template <class _Context, class... _Args>
_LIBCPP_HIDE_FROM_ABI void __store_basic_format_arg(basic_format_arg<_Context>* __data, _Args&... __args) noexcept {
  ([&] { *__data++ = __format::__create_format_arg<_Context>(__args); }(), ...);
}

template <class _Context, size_t _Np>
struct __packed_format_arg_store {
  __basic_format_arg_value<_Context> __values_[_Np];
  uint64_t __types_ = 0;
};

template <class _Context>
struct __packed_format_arg_store<_Context, 0> {
  uint64_t __types_ = 0;
};

template <class _Context, size_t _Np>
struct __unpacked_format_arg_store {
  basic_format_arg<_Context> __args_[_Np];
};

} // namespace __format

template <class _Context, class... _Args>
struct _LIBCPP_TEMPLATE_VIS __format_arg_store {
  _LIBCPP_HIDE_FROM_ABI __format_arg_store(_Args&... __args) noexcept {
    if constexpr (sizeof...(_Args) != 0) {
      if constexpr (__format::__use_packed_format_arg_store(sizeof...(_Args)))
        __format::__create_packed_storage(__storage.__types_, __storage.__values_, __args...);
      else
        __format::__store_basic_format_arg<_Context>(__storage.__args_, __args...);
    }
  }

  using _Storage _LIBCPP_NODEBUG =
      conditional_t<__format::__use_packed_format_arg_store(sizeof...(_Args)),
                    __format::__packed_format_arg_store<_Context, sizeof...(_Args)>,
                    __format::__unpacked_format_arg_store<_Context, sizeof...(_Args)>>;

  _Storage __storage;
};

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FORMAT_FORMAT_ARG_STORE_H
