// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FORMAT_FUNCTIONS
#define _LIBCPP___FORMAT_FORMAT_FUNCTIONS

#include <__algorithm/clamp.h>
#include <__algorithm/ranges_find_first_of.h>
#include <__chrono/statically_widen.h>
#include <__concepts/convertible_to.h>
#include <__concepts/same_as.h>
#include <__config>
#include <__format/buffer.h>
#include <__format/format_arg.h>
#include <__format/format_arg_store.h>
#include <__format/format_args.h>
#include <__format/format_context.h>
#include <__format/format_error.h>
#include <__format/format_parse_context.h>
#include <__format/format_string.h>
#include <__format/format_to_n_result.h>
#include <__format/formatter.h>
#include <__format/formatter_bool.h>
#include <__format/formatter_char.h>
#include <__format/formatter_floating_point.h>
#include <__format/formatter_integer.h>
#include <__format/formatter_pointer.h>
#include <__format/formatter_string.h>
#include <__format/parser_std_format_spec.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h> // iter_value_t
#include <__variant/monostate.h>
#include <array>
#include <optional>
#include <string>
#include <string_view>

#if _LIBCPP_HAS_LOCALIZATION
#  include <__locale>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

// TODO FMT Evaluate which templates should be external templates. This
// improves the efficiency of the header. However since the header is still
// under heavy development and not all classes are stable it makes no sense
// to do this optimization now.

using format_args = basic_format_args<format_context>;
#  if _LIBCPP_HAS_WIDE_CHARACTERS
using wformat_args = basic_format_args<wformat_context>;
#  endif

template <class _Context = format_context, class... _Args>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI __format_arg_store<_Context, _Args...> make_format_args(_Args&... __args) {
  return std::__format_arg_store<_Context, _Args...>(__args...);
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <class... _Args>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI __format_arg_store<wformat_context, _Args...> make_wformat_args(_Args&... __args) {
  return std::__format_arg_store<wformat_context, _Args...>(__args...);
}
#  endif

template <output_iterator<const char&> _OutIt>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt vformat_to(_OutIt __out_it, string_view __fmt, format_args __args);

namespace __format {

/// Helper class parse and handle argument.
///
/// When parsing a handle which is not enabled the code is ill-formed.
/// This helper uses the parser of the appropriate formatter for the stored type.
template <class _CharT>
class __compile_time_handle {
public:
  template <class _ParseContext>
  _LIBCPP_HIDE_FROM_ABI constexpr void __parse(_ParseContext& __ctx) const {
    __parse_(__ctx);
  }

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI constexpr void __enable() {
    __parse_ = [](basic_format_parse_context<_CharT>& __ctx) {
      formatter<_Tp, _CharT> __f;
      __ctx.advance_to(__f.parse(__ctx));
    };
  }

  // Before calling __parse the proper handler needs to be set with __enable.
  // The default handler isn't a core constant expression.
  _LIBCPP_HIDE_FROM_ABI constexpr __compile_time_handle()
      : __parse_([](basic_format_parse_context<_CharT>&) { std::__throw_format_error("Not a handle"); }) {}

private:
  void (*__parse_)(basic_format_parse_context<_CharT>&);
};

// Dummy format_context only providing the parts used during constant
// validation of the basic_format_string.
template <class _CharT>
struct __compile_time_basic_format_context {
public:
  using char_type = _CharT;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __compile_time_basic_format_context(
      const __arg_t* __args, const __compile_time_handle<_CharT>* __handles, size_t __size)
      : __args_(__args), __handles_(__handles), __size_(__size) {}

  // During the compile-time validation nothing needs to be written.
  // Therefore all operations of this iterator are a NOP.
  struct iterator {
    _LIBCPP_HIDE_FROM_ABI constexpr iterator& operator=(_CharT) { return *this; }
    _LIBCPP_HIDE_FROM_ABI constexpr iterator& operator*() { return *this; }
    _LIBCPP_HIDE_FROM_ABI constexpr iterator operator++(int) { return *this; }
  };

  _LIBCPP_HIDE_FROM_ABI constexpr __arg_t arg(size_t __id) const {
    if (__id >= __size_)
      std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    return __args_[__id];
  }

  _LIBCPP_HIDE_FROM_ABI constexpr const __compile_time_handle<_CharT>& __handle(size_t __id) const {
    if (__id >= __size_)
      std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    return __handles_[__id];
  }

  _LIBCPP_HIDE_FROM_ABI constexpr iterator out() { return {}; }
  _LIBCPP_HIDE_FROM_ABI constexpr void advance_to(iterator) {}

private:
  const __arg_t* __args_;
  const __compile_time_handle<_CharT>* __handles_;
  size_t __size_;
};

template <class _CharT>
union __any_std_formatter {
#  define _LIBCCP_IMPLICIT_UNION(__type, __member_name)                                                                \
    constexpr __any_std_formatter(formatter<__type, _CharT> __formatter) : __member_name(__formatter) {}               \
    explicit constexpr operator formatter<__type, _CharT>() const { return __member_name; }                            \
    formatter<__type, _CharT> __member_name

  _LIBCCP_IMPLICIT_UNION(bool, __b);
  _LIBCCP_IMPLICIT_UNION(_CharT, __c);

  _LIBCCP_IMPLICIT_UNION(int, __i);
  _LIBCCP_IMPLICIT_UNION(long long, __ll);
#  if _LIBCPP_HAS_INT128
  _LIBCCP_IMPLICIT_UNION(__int128_t, __i128);
#  endif

  _LIBCCP_IMPLICIT_UNION(unsigned, __u);
  _LIBCCP_IMPLICIT_UNION(unsigned long long, __ull);
#  if _LIBCPP_HAS_INT128
  _LIBCCP_IMPLICIT_UNION(__uint128_t, __u128);
#  endif

  _LIBCCP_IMPLICIT_UNION(float, __f);
  _LIBCCP_IMPLICIT_UNION(double, __d);
  _LIBCCP_IMPLICIT_UNION(long double, __ld);

  _LIBCCP_IMPLICIT_UNION(const _CharT*, __cs);
  _LIBCCP_IMPLICIT_UNION(basic_string_view<_CharT>, __sv);

  _LIBCCP_IMPLICIT_UNION(const void*, __p);

#  undef _LIBCCP_IMPLICIT_UNION
};

// [format.string.std]/8
// If { arg-idopt } is used in a width or precision, the value of the
// corresponding formatting argument is used in its place. If the
// corresponding formatting argument is not of standard signed or unsigned
// integer type, or its value is negative for precision or non-positive for
// width, an exception of type format_error is thrown.
//
// _HasPrecision does the formatter have a precision?
template <class _CharT, class _Tp, bool _HasPrecision = false>
_LIBCPP_HIDE_FROM_ABI constexpr __any_std_formatter<_CharT> __compile_time_validate_argument(
    basic_format_parse_context<_CharT>& __parse_ctx, __compile_time_basic_format_context<_CharT>& __ctx) {
  auto __validate_type = [](__arg_t __type) {
    // LWG3720 originally allowed "signed or unsigned integer types", however
    // the final version explicitly changed it to "*standard* signed or unsigned
    // integer types". It's trivial to use 128-bit integrals in libc++'s
    // implementation, but other implementations may not implement it.
    // (Using a width or precision, that does not fit in 64-bits, sounds very
    // unlikely in real world code.)
    switch (__type) {
    case __arg_t::__int:
    case __arg_t::__long_long:
    case __arg_t::__unsigned:
    case __arg_t::__unsigned_long_long:
      return;

    default:
      std::__throw_format_error("Replacement argument isn't a standard signed or unsigned integer type");
    }
  };

  formatter<_Tp, _CharT> __formatter;
  __parse_ctx.advance_to(__formatter.parse(__parse_ctx));
  if (__formatter.__parser_.__width_as_arg_)
    __validate_type(__ctx.arg(__formatter.__parser_.__width_));

  if constexpr (_HasPrecision)
    if (__formatter.__parser_.__precision_as_arg_)
      __validate_type(__ctx.arg(__formatter.__parser_.__precision_));

  return __formatter;
}

// This function is not user facing, so it can directly use the non-standard types of the "variant".
template <class _CharT>
_LIBCPP_HIDE_FROM_ABI constexpr __any_std_formatter<_CharT> __compile_time_visit_format_arg(
    basic_format_parse_context<_CharT>& __parse_ctx,
    __compile_time_basic_format_context<_CharT>& __ctx,
    __arg_t __type) {
  switch (__type) {
  case __arg_t::__none:
    std::__throw_format_error("Invalid argument");
  case __arg_t::__boolean:
    return __format::__compile_time_validate_argument<_CharT, bool>(__parse_ctx, __ctx);
  case __arg_t::__char_type:
    return __format::__compile_time_validate_argument<_CharT, _CharT>(__parse_ctx, __ctx);
  case __arg_t::__int:
    return __format::__compile_time_validate_argument<_CharT, int>(__parse_ctx, __ctx);
  case __arg_t::__long_long:
    return __format::__compile_time_validate_argument<_CharT, long long>(__parse_ctx, __ctx);
  case __arg_t::__i128:
#  if _LIBCPP_HAS_INT128
    return __format::__compile_time_validate_argument<_CharT, __int128_t>(__parse_ctx, __ctx);
#  else
    std::__throw_format_error("Invalid argument");
#  endif
    break;
  case __arg_t::__unsigned:
    return __format::__compile_time_validate_argument<_CharT, unsigned>(__parse_ctx, __ctx);
  case __arg_t::__unsigned_long_long:
    return __format::__compile_time_validate_argument<_CharT, unsigned long long>(__parse_ctx, __ctx);
  case __arg_t::__u128:
#  if _LIBCPP_HAS_INT128
    return __format::__compile_time_validate_argument<_CharT, __uint128_t>(__parse_ctx, __ctx);
#  else
    std::__throw_format_error("Invalid argument");
#  endif
    break;
  case __arg_t::__float:
    return __format::__compile_time_validate_argument<_CharT, float, true>(__parse_ctx, __ctx);
  case __arg_t::__double:
    return __format::__compile_time_validate_argument<_CharT, double, true>(__parse_ctx, __ctx);
  case __arg_t::__long_double:
    return __format::__compile_time_validate_argument<_CharT, long double, true>(__parse_ctx, __ctx);
  case __arg_t::__const_char_type_ptr:
    return __format::__compile_time_validate_argument<_CharT, const _CharT*, true>(__parse_ctx, __ctx);
  case __arg_t::__string_view:
    return __format::__compile_time_validate_argument<_CharT, basic_string_view<_CharT>, true>(__parse_ctx, __ctx);
  case __arg_t::__ptr:
    return __format::__compile_time_validate_argument<_CharT, const void*>(__parse_ctx, __ctx);
  case __arg_t::__handle:
    std::__throw_format_error("Handle should use __compile_time_validate_handle_argument");
  }
  std::__throw_format_error("Invalid argument");
}

template <contiguous_iterator _Iterator, class _ParseCtx, class _Ctx>
_LIBCPP_HIDE_FROM_ABI constexpr _Iterator
__handle_replacement_field(_Iterator __begin, _Iterator __end, _ParseCtx& __parse_ctx, _Ctx& __ctx) {
  using _CharT                        = iter_value_t<_Iterator>;
  __format::__parse_number_result __r = __format::__parse_arg_id(__begin, __end, __parse_ctx);

  if (__r.__last == __end)
    std::__throw_format_error("The argument index should end with a ':' or a '}'");

  bool __parse = *__r.__last == _CharT(':');
  switch (*__r.__last) {
  case _CharT(':'):
    // The arg-id has a format-specifier, advance the input to the format-spec.
    __parse_ctx.advance_to(__r.__last + 1);
    break;
  case _CharT('}'):
    // The arg-id has no format-specifier.
    __parse_ctx.advance_to(__r.__last);
    break;
  default:
    std::__throw_format_error("The argument index should end with a ':' or a '}'");
  }

  if constexpr (same_as<_Ctx, __compile_time_basic_format_context<_CharT>>) {
    __arg_t __type = __ctx.arg(__r.__value);
    if (__type == __arg_t::__none)
      std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
    else if (__type == __arg_t::__handle)
      __ctx.__handle(__r.__value).__parse(__parse_ctx);
    else if (__parse)
      __format::__compile_time_visit_format_arg(__parse_ctx, __ctx, __type);
  } else
    std::__visit_format_arg(
        [&](auto __arg) {
          if constexpr (same_as<decltype(__arg), monostate>)
            std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
          else if constexpr (same_as<decltype(__arg), typename basic_format_arg<_Ctx>::handle>)
            __arg.format(__parse_ctx, __ctx);
          else {
            formatter<decltype(__arg), _CharT> __formatter;
            if (__parse)
              __parse_ctx.advance_to(__formatter.parse(__parse_ctx));
            __ctx.advance_to(__formatter.format(__arg, __ctx));
          }
        },
        __ctx.arg(__r.__value));

  __begin = __parse_ctx.begin();
  if (__begin == __end || *__begin != _CharT('}'))
    std::__throw_format_error("The replacement field misses a terminating '}'");

  return ++__begin;
}

template <class _ParseCtx, class _Ctx>
_LIBCPP_HIDE_FROM_ABI constexpr typename _Ctx::iterator __vformat_to(_ParseCtx&& __parse_ctx, _Ctx&& __ctx) {
  using _CharT = typename _ParseCtx::char_type;
  static_assert(same_as<typename _Ctx::char_type, _CharT>);

  auto __begin                     = __parse_ctx.begin();
  auto __end                       = __parse_ctx.end();
  typename _Ctx::iterator __out_it = __ctx.out();
  while (__begin != __end) {
    switch (*__begin) {
    case _CharT('{'):
      ++__begin;
      if (__begin == __end)
        std::__throw_format_error("The format string terminates at a '{'");

      if (*__begin != _CharT('{')) [[likely]] {
        __ctx.advance_to(std::move(__out_it));
        __begin  = __format::__handle_replacement_field(__begin, __end, __parse_ctx, __ctx);
        __out_it = __ctx.out();

        // The output is written and __begin points to the next character. So
        // start the next iteration.
        continue;
      }
      // The string is an escape character.
      break;

    case _CharT('}'):
      ++__begin;
      if (__begin == __end || *__begin != _CharT('}'))
        std::__throw_format_error("The format string contains an invalid escape sequence");

      break;
    }

    // Copy the character to the output verbatim.
    *__out_it++ = *__begin++;
  }
  return __out_it;
}

} // namespace __format

#  if _LIBCPP_STD_VER >= 26
template <class _CharT>
struct __runtime_format_string {
private:
  basic_string_view<_CharT> __str_;

  template <class _Cp, class... _Args>
  friend struct basic_format_string;

public:
  _LIBCPP_HIDE_FROM_ABI __runtime_format_string(basic_string_view<_CharT> __s) noexcept : __str_(__s) {}

  __runtime_format_string(const __runtime_format_string&)            = delete;
  __runtime_format_string& operator=(const __runtime_format_string&) = delete;
};

_LIBCPP_HIDE_FROM_ABI inline __runtime_format_string<char> runtime_format(string_view __fmt) noexcept { return __fmt; }
#    if _LIBCPP_HAS_WIDE_CHARACTERS
_LIBCPP_HIDE_FROM_ABI inline __runtime_format_string<wchar_t> runtime_format(wstring_view __fmt) noexcept {
  return __fmt;
}
#    endif
#  endif // _LIBCPP_STD_VER >= 26

template <class _CharT, class... _Args>
struct basic_format_string {
  template <class _Tp>
    requires convertible_to<const _Tp&, basic_string_view<_CharT>>
  consteval basic_format_string(const _Tp& __str) : __str_{__str} {
    __format::__vformat_to(basic_format_parse_context<_CharT>{__str_, sizeof...(_Args)},
                           _Context{__types_.data(), __handles_.data(), sizeof...(_Args)});

    basic_format_parse_context<_CharT> __parse_ctx{__str_, sizeof...(_Args)};
    _Context __ctx{__types_.data(), __handles_.data(), sizeof...(_Args)};

    __state_ = [&] {
      auto __begin         = __str_.begin();
      auto __end           = __str_.end();
      auto __element_begin = __str_.begin();

      while (__begin != __end) {
        switch (*__begin) {
        case _CharT('{'):
          if (__begin + 1 == __end)
            std::__throw_format_error("The format string terminates at a '{'");

          if (*(__begin + 1) == _CharT('{')) {
            if (!__add_curly_brace(__element_begin, __begin, __element_type::__curly_brace_open))
              return __element_state::__parsing_partial;

          } else {
            if (__element_begin != __begin) {
              // Add the parsed string before the replaacemet
              if (!__add_string(__element_begin, __begin))
                return __element_state::__parsing_partial;
            }

            __format::__parse_number_result __r = __process_arg_id(__begin + 1, __end, __parse_ctx);
            __format::__arg_t __type            = __ctx.arg(__r.__value);
            if (__type == __format::__arg_t::__none)
              std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
            else if (__type == __format::__arg_t::__handle)
              //
              // TODO ADD Handle support
              //
              return __element_state::__do_not_use;
            else {
              if (!__add_std_formatter(__element_begin, __begin, __end, __r.__value, __type, __parse_ctx, __ctx))
                return __element_state::__parsing_partial;
            }
          }
          break;

        case _CharT('}'):
          if (__begin + 1 == __end || *(__begin + 1) != _CharT('}'))
            std::__throw_format_error("The format string contains an invalid escape sequence");

          if (!__add_curly_brace(__element_begin, __begin, __element_type::__curly_brace_close))
            return __element_state::__parsing_partial;
          break;

        default:
          ++__begin;
        }
      }
      // Handle last element. This should always be a string element, else there is a logic errror.
      if (__element_count_ == 0 || __element_begin != __end) {
        if (!__add_string(__element_begin, __begin))
          return __element_state::__parsing_partial;
      }
      return __element_state::__parsing_completed;
    }();
  }

  _LIBCPP_HIDE_FROM_ABI constexpr basic_string_view<_CharT> get() const noexcept { return __str_; }
#  if _LIBCPP_STD_VER >= 26
  _LIBCPP_HIDE_FROM_ABI basic_format_string(__runtime_format_string<_CharT> __s) noexcept : __str_(__s.__str_) {}
#  endif

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI string __vformat(format_args __args) {
    __format::__allocating_buffer<char> __buffer;
    switch (__state_) {
    case __element_state::__parsing_partial:
#  ifdef _LIBCPP_DEBUG_BASIC_FORMAT_STRING_CACHE
      __buffer.__copy(std::string_view{"PART:"});
#  endif
      [[fallthrough]];
    case __element_state::__do_not_use:
      std::vformat_to(__buffer.__make_output_iterator(), __str_, __args);
      break;

    case __element_state::__parsing_completed:
      __format(__buffer, __args);
      break;
    }

    return string{__buffer.__view()};
  }

private:
  basic_string_view<_CharT> __str_;

  using _Context _LIBCPP_NODEBUG = __format::__compile_time_basic_format_context<_CharT>;

  static constexpr array<__format::__arg_t, sizeof...(_Args)> __types_{
      __format::__determine_arg_t<_Context, remove_cvref_t<_Args>>()...};

  static constexpr array<__format::__compile_time_handle<_CharT>, sizeof...(_Args)> __handles_{[] {
    using _Tp = remove_cvref_t<_Args>;
    __format::__compile_time_handle<_CharT> __handle;
    if (__format::__determine_arg_t<_Context, _Tp>() == __format::__arg_t::__handle)
      __handle.template __enable<_Tp>();

    return __handle;
  }()...};

  using __iterator = basic_string_view<_CharT>::iterator;

  enum class __element_state {
    // The __elements_ should not be used.
    //
    // This is intened to be removed once everything works
    __do_not_use,

    // The entire __str_ has been parsed and stored in __elements_
    __parsing_completed,
    // The input __str_ has been partially parsed and stored in __elements_.
    //
    // This happens when the __str_ needed more entries than available in __elemenst_,
    // so the output can first process __elements_ and then do run-time parsing for
    // the remaining part of __str_.
    //
    //
    // TODO test autotic numbering keeps the right elements
    //
    __parsing_partial,

  };

  __element_state __state_;

  enum class __element_type {
    // This is a formatter for one of the non-handle types.
    //
    // All these formatters use the same state for parsing this state is stored in the array.
    // This means at run-time there is no need to parse this replacement-field.
    __formatter_std,
    // A handle formatter
    //
    // This replacement-field is validated at compile-time, but its state is not stored.
    // This means it needs to be parsed again at run-time.
    __formatter_handle,

    // This part of the format-string contains a wstring.
    //
    // This can be copied to the output verbatim.
    __wstring,
    // This part of the format-string contains a string with valid UTF-8.
    //
    // This can be copied to the output verbatim.
    __string_valid_utf8,
    // This part of the format-string contains a string with invalid UTF-8.
    //
    // Since UTF-8 is self syncing the end of the string can be determined
    //
    // The distiction between wstring, string (|in)valid UTF-8 is needed for
    // std::string. When the string contains invalid Unicode the recommended
    // practice is to diagnose this. std::print only takes `char` as input
    // character type. So when using std::print the algorithm can use this
    // information to diagnose things. (Diagnose means using Unicode
    // replacement characters.)
    __string_invalid_utf8,
    // Found {{ in the input, unescape to {.
    __curly_brace_open,
    // Found }} in the input, unescape to }.
    //
    // Note it would be possible to use __char and then store '{' or '}' in
    // the __parser_state_'s fill field. However that feels clumsy and there
    // are only 2 escape characters.
    __curly_brace_close,
    // The field is empty.
    //
    // The number of fields is hard-coded so this is a sentinal value.
    // Once an element is empty, all remaining elements are empty too.
    __empty,
  };

  struct __element_formatter {
    uint32_t __id = 0;
    __format::__any_std_formatter<_CharT> __formatter;
  };

  struct __element {
    __element_type __type;
    union {
      __element_formatter __formatter;
      basic_string_view<_CharT> __string;
      int __dummy;
    } __data_ = {.__dummy = 0};
  };

  size_t __element_count_ = 0;

  // The elements found in __str_
  static constexpr size_t __n_elements_       = 32;
  array<__element, __n_elements_> __elements_ = [] {
    array<__element, __n_elements_> __result;
    for (size_t __i = 0; __i < __result.size(); ++__i) {
      __result[__i] = {__element_type::__empty};
    }
    return __result;
  }();

  // TODO add UTF-8 validation to this function.
  [[nodiscard]] consteval bool __add_string(__iterator __begin, __iterator __end) {
    if (__element_count_ == __elements_.size())
      return false;

    __elements_[__element_count_++] = {
        same_as<_CharT, char> ? __element_type::__string_valid_utf8 : __element_type::__wstring,
        {.__string = {__begin, __end}}};

    return true;
  }

  // Adds an escaped curly brace, with an optional string prefix, to the parsed elements.
  //
  // When the input is like
  //   abc{{
  //   ^  ^ ^
  //   1  2 3
  //
  // 1. input value of __element_begin
  // 2. input value of __begin
  // 3. output value of __element_begin and __begin
  //
  // pre __begin + 1 == __begin + 2 == ('{' || '}')
  // pre __type = *(__begin + 1) == '{' ? __curly_brace_open
  //                                    : *(__begin + 1) == '}' ? __curly_brace_close
  //                                                            : pre-condition failure
  //
  // __begin and __element_begin may point to the same element, in that case
  // there is no parsed string. For example, "{}{{" or "{{" at the start of the
  // input.
  //
  // When there is a string prefix the curly brace is stored in the string.
  // This saves 1 entry.
  [[nodiscard]] consteval bool
  __add_curly_brace(__iterator& __element_begin, __iterator& __begin, __element_type __type) {
    if (__element_begin == __begin) {
      // No string before.
      if (__element_count_ == __elements_.size())
        return false;
      __elements_[__element_count_++] = {__type};
    } else {
      // Merge with string.
      if (!__add_string(__element_begin, __begin + 1))
        return false;
    }

    __begin += 2;
    __element_begin = __begin;
    return true;
  }

  [[nodiscard]] consteval bool __add_std_formatter(
      __iterator& __element_begin,
      __iterator& __begin,
      __iterator __end,
      uint32_t __arg_id,
      __format::__arg_t __type,
      basic_format_parse_context<_CharT>& __parse_ctx,
      _Context& __ctx) {
    if (__element_count_ == __elements_.size())
      return false;

    __format::__any_std_formatter<_CharT> __formatter = __compile_time_visit_format_arg(__parse_ctx, __ctx, __type);
    __begin                                           = __parse_ctx.begin();
    if (__begin == __end || *__begin != _CharT('}'))
      std::__throw_format_error("The replacement field misses a terminating '}'");

    ++__begin;
    __element_begin = __begin;

    __elements_[__element_count_++] = {__element_type::__formatter_std, {.__formatter{__arg_id, __formatter}}

    };

    return true;
  }

  [[nodiscard]] consteval __format::__parse_number_result<__iterator>
  __process_arg_id(__iterator __begin, __iterator __end, basic_format_parse_context<_CharT>& __parse_ctx) {
    __format::__parse_number_result __result = __format::__parse_arg_id(__begin, __end, __parse_ctx);
    if (__result.__last == __end)
      std::__throw_format_error("The argument index should end with a ':' or a '}'");

    //
    // Note we always need to parse to get the proper default parser settings,
    // this differs from where we copy pasted the data
    //
    // bool __parse = *__result.__last == _CharT(':');
    switch (*__result.__last) {
    case _CharT(':'):
      // The arg-id has a format-specifier, advance the input to the format-spec.
      __parse_ctx.advance_to(__result.__last + 1);
      break;
    case _CharT('}'):
      // The arg-id has no format-specifier.
      __parse_ctx.advance_to(__result.__last);
      break;
    default:
      std::__throw_format_error("The argument index should end with a ':' or a '}'");
    }

    return __result;
  }

  void __format(__format::__allocating_buffer<char>& __buffer, format_args __args) {
#  ifdef _LIBCPP_DEBUG_BASIC_FORMAT_STRING_CACHE
    __buffer.__copy(std::string_view{"FULL:"});
#  endif
    auto __ctx = std::__format_context_create(__buffer.__make_output_iterator(), __args);

    for (auto& __element : __elements_) {
      if (__element.__type == __element_type::__empty)
        break;

      switch (__element.__type) {
      case __element_type::__formatter_std:
      {
        std::__visit_format_arg(
            [&](auto __arg) {
              if constexpr (same_as<decltype(__arg), monostate>)
                std::__throw_format_error("The argument index value is too large for the number of arguments supplied");
              else if constexpr (same_as<decltype(__arg), typename basic_format_arg<decltype(__ctx)>::handle>)
                std::__throw_format_error("Not implemented");
              else {
                // TODO __formatter.__formatter looks odd, maybe the first should
                // be __replacement_field.
                const auto& __formatter =
                    static_cast<formatter<decltype(__arg), _CharT>>(__element.__data_.__formatter.__formatter);

                __ctx.advance_to(__formatter.format(__arg, __ctx));
              }
            },
            __ctx.arg(__element.__data_.__formatter.__id));

      } break;

      case __element_type::__formatter_handle:
        break;
      case __element_type::__string_invalid_utf8:
        if (false) {
          break;
        }
        [[fallthrough]];
      case __element_type::__string_valid_utf8:
        [[fallthrough]];
      case __element_type::__wstring:
        __buffer.__copy(__element.__data_.__string);
        break;
      case __element_type::__curly_brace_open:
        __buffer.push_back(_CharT('{'));
        break;
      case __element_type::__curly_brace_close:
        __buffer.push_back(_CharT('}'));
        break;
      case __element_type::__empty:
        break;
      }
    }
  }
};

template <class... _Args>
using format_string = basic_format_string<char, type_identity_t<_Args>...>;

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <class... _Args>
using wformat_string = basic_format_string<wchar_t, type_identity_t<_Args>...>;
#  endif

template <class _OutIt, class _CharT, class _FormatOutIt>
  requires(output_iterator<_OutIt, const _CharT&>)
_LIBCPP_HIDE_FROM_ABI _OutIt __vformat_to(_OutIt __out_it,
                                          basic_string_view<_CharT> __fmt,
                                          basic_format_args<basic_format_context<_FormatOutIt, _CharT>> __args) {
  if constexpr (same_as<_OutIt, _FormatOutIt>)
    return std::__format::__vformat_to(
        basic_format_parse_context{__fmt, __args.__size()}, std::__format_context_create(std::move(__out_it), __args));
  else {
    typename __format::__buffer_selector<_OutIt, _CharT>::type __buffer{std::move(__out_it)};
    std::__format::__vformat_to(basic_format_parse_context{__fmt, __args.__size()},
                                std::__format_context_create(__buffer.__make_output_iterator(), __args));
    return std::move(__buffer).__out_it();
  }
}

// The function is _LIBCPP_ALWAYS_INLINE since the compiler is bad at inlining
// https://reviews.llvm.org/D110499#inline-1180704
// TODO FMT Evaluate whether we want to file a Clang bug report regarding this.
template <output_iterator<const char&> _OutIt>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt vformat_to(_OutIt __out_it, string_view __fmt, format_args __args) {
  return std::__vformat_to(std::move(__out_it), __fmt, __args);
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <output_iterator<const wchar_t&> _OutIt>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
vformat_to(_OutIt __out_it, wstring_view __fmt, wformat_args __args) {
  return std::__vformat_to(std::move(__out_it), __fmt, __args);
}
#  endif

template <output_iterator<const char&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
format_to(_OutIt __out_it, format_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat_to(std::move(__out_it), __fmt.get(), std::make_format_args(__args...));
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <output_iterator<const wchar_t&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
format_to(_OutIt __out_it, wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat_to(std::move(__out_it), __fmt.get(), std::make_wformat_args(__args...));
}
#  endif

// Try constant folding the format string instead of going through the whole formatting machinery. If there is no
// constant folding no extra code should be emitted (with optimizations enabled) and the function returns nullopt. When
// constant folding is successful, the formatting is performed and the resulting string is returned.
namespace __format {
template <class _CharT>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI optional<basic_string<_CharT>> __try_constant_folding(
    basic_string_view<_CharT> __fmt,
    basic_format_args<basic_format_context<back_insert_iterator<__format::__output_buffer<_CharT>>, _CharT>> __args) {
  // Fold strings not containing '{' or '}' to just return the string
  if (bool __is_identity = [&] [[__gnu__::__pure__]] // Make sure the compiler knows this call can be eliminated
      { return std::ranges::find_first_of(__fmt, array{'{', '}'}) == __fmt.end(); }();
      __builtin_constant_p(__is_identity) && __is_identity)
    return basic_string<_CharT>{__fmt};

  // Fold '{}' to the appropriate conversion function
  if (auto __only_first_arg = __fmt == _LIBCPP_STATICALLY_WIDEN(_CharT, "{}");
      __builtin_constant_p(__only_first_arg) && __only_first_arg) {
    if (auto __arg = __args.get(0); __builtin_constant_p(__arg.__type_)) {
      return std::__visit_format_arg(
          []<class _Tp>(_Tp&& __argument) -> optional<basic_string<_CharT>> {
            if constexpr (is_same_v<remove_cvref_t<_Tp>, basic_string_view<_CharT>>) {
              return basic_string<_CharT>{__argument};
            } else {
              return nullopt;
            }
          },
          __arg);
    }
  }

  return nullopt;
}
} // namespace __format

// TODO FMT This needs to be a template or std::to_chars(floating-point) availability markup
// fires too eagerly, see http://llvm.org/PR61563.
template <class = void>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE inline _LIBCPP_HIDE_FROM_ABI string vformat(string_view __fmt, format_args __args) {
  auto __result = __format::__try_constant_folding(__fmt, __args);
  if (__result.has_value())
    return *std::move(__result);
  __format::__allocating_buffer<char> __buffer;
  std::vformat_to(__buffer.__make_output_iterator(), __fmt, __args);
  return string{__buffer.__view()};
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
// TODO FMT This needs to be a template or std::to_chars(floating-point) availability markup
// fires too eagerly, see http://llvm.org/PR61563.
template <class = void>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE inline _LIBCPP_HIDE_FROM_ABI wstring
vformat(wstring_view __fmt, wformat_args __args) {
  auto __result = __format::__try_constant_folding(__fmt, __args);
  if (__result.has_value())
    return *std::move(__result);
  __format::__allocating_buffer<wchar_t> __buffer;
  std::vformat_to(__buffer.__make_output_iterator(), __fmt, __args);
  return wstring{__buffer.__view()};
}
#  endif

template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI string
format(format_string<_Args...> __fmt, _Args&&... __args) {
  return __fmt.__vformat(std::make_format_args(__args...));
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI wstring
format(wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat(__fmt.get(), std::make_wformat_args(__args...));
}
#  endif

template <class _Context, class _OutIt, class _CharT>
_LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt>
__vformat_to_n(_OutIt __out_it,
               iter_difference_t<_OutIt> __n,
               basic_string_view<_CharT> __fmt,
               basic_format_args<_Context> __args) {
  __format::__format_to_n_buffer<_OutIt, _CharT> __buffer{std::move(__out_it), __n};
  std::__format::__vformat_to(basic_format_parse_context{__fmt, __args.__size()},
                              std::__format_context_create(__buffer.__make_output_iterator(), __args));
  return std::move(__buffer).__result();
}

template <output_iterator<const char&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt>
format_to_n(_OutIt __out_it, iter_difference_t<_OutIt> __n, format_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformat_to_n<format_context>(std::move(__out_it), __n, __fmt.get(), std::make_format_args(__args...));
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <output_iterator<const wchar_t&> _OutIt, class... _Args>
_LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt>
format_to_n(_OutIt __out_it, iter_difference_t<_OutIt> __n, wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformat_to_n<wformat_context>(std::move(__out_it), __n, __fmt.get(), std::make_wformat_args(__args...));
}
#  endif

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI size_t __vformatted_size(basic_string_view<_CharT> __fmt, auto __args) {
  __format::__formatted_size_buffer<_CharT> __buffer;
  std::__format::__vformat_to(basic_format_parse_context{__fmt, __args.__size()},
                              std::__format_context_create(__buffer.__make_output_iterator(), __args));
  return std::move(__buffer).__result();
}

template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI size_t
formatted_size(format_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformatted_size(__fmt.get(), basic_format_args{std::make_format_args(__args...)});
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI size_t
formatted_size(wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformatted_size(__fmt.get(), basic_format_args{std::make_wformat_args(__args...)});
}
#  endif

#  if _LIBCPP_HAS_LOCALIZATION

template <class _OutIt, class _CharT, class _FormatOutIt>
  requires(output_iterator<_OutIt, const _CharT&>)
_LIBCPP_HIDE_FROM_ABI _OutIt __vformat_to(
    _OutIt __out_it,
    locale __loc,
    basic_string_view<_CharT> __fmt,
    basic_format_args<basic_format_context<_FormatOutIt, _CharT>> __args) {
  if constexpr (same_as<_OutIt, _FormatOutIt>)
    return std::__format::__vformat_to(basic_format_parse_context{__fmt, __args.__size()},
                                       std::__format_context_create(std::move(__out_it), __args, std::move(__loc)));
  else {
    typename __format::__buffer_selector<_OutIt, _CharT>::type __buffer{std::move(__out_it)};
    std::__format::__vformat_to(
        basic_format_parse_context{__fmt, __args.__size()},
        std::__format_context_create(__buffer.__make_output_iterator(), __args, std::move(__loc)));
    return std::move(__buffer).__out_it();
  }
}

template <output_iterator<const char&> _OutIt>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
vformat_to(_OutIt __out_it, locale __loc, string_view __fmt, format_args __args) {
  return std::__vformat_to(std::move(__out_it), std::move(__loc), __fmt, __args);
}

#    if _LIBCPP_HAS_WIDE_CHARACTERS
template <output_iterator<const wchar_t&> _OutIt>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
vformat_to(_OutIt __out_it, locale __loc, wstring_view __fmt, wformat_args __args) {
  return std::__vformat_to(std::move(__out_it), std::move(__loc), __fmt, __args);
}
#    endif

template <output_iterator<const char&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
format_to(_OutIt __out_it, locale __loc, format_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat_to(std::move(__out_it), std::move(__loc), __fmt.get(), std::make_format_args(__args...));
}

#    if _LIBCPP_HAS_WIDE_CHARACTERS
template <output_iterator<const wchar_t&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI _OutIt
format_to(_OutIt __out_it, locale __loc, wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat_to(std::move(__out_it), std::move(__loc), __fmt.get(), std::make_wformat_args(__args...));
}
#    endif

// TODO FMT This needs to be a template or std::to_chars(floating-point) availability markup
// fires too eagerly, see http://llvm.org/PR61563.
template <class = void>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE inline _LIBCPP_HIDE_FROM_ABI string
vformat(locale __loc, string_view __fmt, format_args __args) {
  __format::__allocating_buffer<char> __buffer;
  std::vformat_to(__buffer.__make_output_iterator(), std::move(__loc), __fmt, __args);
  return string{__buffer.__view()};
}

#    if _LIBCPP_HAS_WIDE_CHARACTERS
// TODO FMT This needs to be a template or std::to_chars(floating-point) availability markup
// fires too eagerly, see http://llvm.org/PR61563.
template <class = void>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE inline _LIBCPP_HIDE_FROM_ABI wstring
vformat(locale __loc, wstring_view __fmt, wformat_args __args) {
  __format::__allocating_buffer<wchar_t> __buffer;
  std::vformat_to(__buffer.__make_output_iterator(), std::move(__loc), __fmt, __args);
  return wstring{__buffer.__view()};
}
#    endif

template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI string
format(locale __loc, format_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat(std::move(__loc), __fmt.get(), std::make_format_args(__args...));
}

#    if _LIBCPP_HAS_WIDE_CHARACTERS
template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI wstring
format(locale __loc, wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::vformat(std::move(__loc), __fmt.get(), std::make_wformat_args(__args...));
}
#    endif

template <class _Context, class _OutIt, class _CharT>
_LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt> __vformat_to_n(
    _OutIt __out_it,
    iter_difference_t<_OutIt> __n,
    locale __loc,
    basic_string_view<_CharT> __fmt,
    basic_format_args<_Context> __args) {
  __format::__format_to_n_buffer<_OutIt, _CharT> __buffer{std::move(__out_it), __n};
  std::__format::__vformat_to(
      basic_format_parse_context{__fmt, __args.__size()},
      std::__format_context_create(__buffer.__make_output_iterator(), __args, std::move(__loc)));
  return std::move(__buffer).__result();
}

template <output_iterator<const char&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt> format_to_n(
    _OutIt __out_it, iter_difference_t<_OutIt> __n, locale __loc, format_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformat_to_n<format_context>(
      std::move(__out_it), __n, std::move(__loc), __fmt.get(), std::make_format_args(__args...));
}

#    if _LIBCPP_HAS_WIDE_CHARACTERS
template <output_iterator<const wchar_t&> _OutIt, class... _Args>
_LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt> format_to_n(
    _OutIt __out_it, iter_difference_t<_OutIt> __n, locale __loc, wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformat_to_n<wformat_context>(
      std::move(__out_it), __n, std::move(__loc), __fmt.get(), std::make_wformat_args(__args...));
}
#    endif

template <class _CharT>
_LIBCPP_HIDE_FROM_ABI size_t __vformatted_size(locale __loc, basic_string_view<_CharT> __fmt, auto __args) {
  __format::__formatted_size_buffer<_CharT> __buffer;
  std::__format::__vformat_to(
      basic_format_parse_context{__fmt, __args.__size()},
      std::__format_context_create(__buffer.__make_output_iterator(), __args, std::move(__loc)));
  return std::move(__buffer).__result();
}

template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI size_t
formatted_size(locale __loc, format_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformatted_size(std::move(__loc), __fmt.get(), basic_format_args{std::make_format_args(__args...)});
}

#    if _LIBCPP_HAS_WIDE_CHARACTERS
template <class... _Args>
[[nodiscard]] _LIBCPP_ALWAYS_INLINE _LIBCPP_HIDE_FROM_ABI size_t
formatted_size(locale __loc, wformat_string<_Args...> __fmt, _Args&&... __args) {
  return std::__vformatted_size(std::move(__loc), __fmt.get(), basic_format_args{std::make_wformat_args(__args...)});
}
#    endif

#  endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FORMAT_FORMAT_FUNCTIONS
