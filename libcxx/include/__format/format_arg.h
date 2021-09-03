// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FORMAT_ARG_H
#define _LIBCPP___FORMAT_FORMAT_ARG_H

#include <__concepts/arithmetic.h>
#include <__config>
#include <__format/format_error.h>
#include <__format/format_fwd.h>
#include <__functional_base>
#include <__variant/monostate.h>
#include <string>
#include <string_view>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

// TODO FMT Remove this once we require compilers with proper C++20 support.
// If the compiler has no concepts support, the format header will be disabled.
// Without concepts support enable_if needs to be used and that too much effort
// to support compilers with partial C++20 support.
#if !defined(_LIBCPP_HAS_NO_CONCEPTS)

namespace __format {
/** The type stored in @ref basic_format_arg. */
enum class _LIBCPP_ENUM_VIS __arg_t : uint8_t {
  __none,
  __boolean,
  __char_type,
  __int,
  __long_long,
#ifndef _LIBCPP_HAS_NO_INT128
  __i128,
#endif
  __unsigned,
  __unsigned_long_long,
#ifndef _LIBCPP_HAS_NO_INT128
  __u128,
#endif
  __float,
  __double,
  __long_double,
  __const_char_type_ptr,
  __string_view,
  __ptr
};
} // namespace __format

template <class _Visitor, class _Context>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_AVAILABILITY_FORMAT decltype(auto)
visit_format_arg(_Visitor&& __vis, basic_format_arg<_Context> __arg) {
  switch (__arg.__type_) {
  case __format::__arg_t::__none:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), monostate{});
  case __format::__arg_t::__boolean:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__boolean);
  case __format::__arg_t::__char_type:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__char_type);
  case __format::__arg_t::__int:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__int);
  case __format::__arg_t::__long_long:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__long_long);
#ifndef _LIBCPP_HAS_NO_INT128
  case __format::__arg_t::__i128:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__i128);
#endif
  case __format::__arg_t::__unsigned:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__unsigned);
  case __format::__arg_t::__unsigned_long_long:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis),
                         __arg.__unsigned_long_long);
#ifndef _LIBCPP_HAS_NO_INT128
  case __format::__arg_t::__u128:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__u128);
#endif
  case __format::__arg_t::__float:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__float);
  case __format::__arg_t::__double:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__double);
  case __format::__arg_t::__long_double:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__long_double);
  case __format::__arg_t::__const_char_type_ptr:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis),
                         __arg.__const_char_type_ptr);
  case __format::__arg_t::__string_view:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__string_view);
  case __format::__arg_t::__ptr:
    return _VSTD::invoke(_VSTD::forward<_Visitor>(__vis), __arg.__ptr);
  }
  _LIBCPP_UNREACHABLE();
}

template <class _Context>
class _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT basic_format_arg {
public:
  // TODO FMT Define the handle class.
  class handle;

  _LIBCPP_HIDE_FROM_ABI basic_format_arg() noexcept
      : __type_{__format::__arg_t::__none} {}

  _LIBCPP_HIDE_FROM_ABI explicit operator bool() const noexcept {
    return __type_ != __format::__arg_t::__none;
  }

private:
  using char_type = typename _Context::char_type;

  // TODO FMT Implement constrain [format.arg]/4
  // Constraints: The template specialization
  //   typename Context::template formatter_type<T>
  // meets the Formatter requirements ([formatter.requirements]).  The extent
  // to which an implementation determines that the specialization meets the
  // Formatter requirements is unspecified, except that as a minimum the
  // expression
  //   typename Context::template formatter_type<T>()
  //    .format(declval<const T&>(), declval<Context&>())
  // shall be well-formed when treated as an unevaluated operand.

  template <class _Ctx, class... _Args>
  _LIBCPP_HIDE_FROM_ABI
      _LIBCPP_AVAILABILITY_FORMAT friend __format_arg_store<_Ctx, _Args...>
      _VSTD::make_format_args(const _Args&...);

  template <class _Visitor, class _Ctx>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_AVAILABILITY_FORMAT friend decltype(auto)
  _VSTD::visit_format_arg(_Visitor&& __vis, basic_format_arg<_Ctx> __arg);

  union {
    bool __boolean;
    char_type __char_type;
    int __int;
    unsigned __unsigned;
    long long __long_long;
    unsigned long long __unsigned_long_long;
#ifndef _LIBCPP_HAS_NO_INT128
    __int128_t __i128;
    __uint128_t __u128;
#endif
    float __float;
    double __double;
    long double __long_double;
    const char_type* __const_char_type_ptr;
    basic_string_view<char_type> __string_view;
    const void* __ptr;
    // TODO FMT Add the handle.
  };
  __format::__arg_t __type_;

  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(bool __v) noexcept
      : __boolean(__v), __type_(__format::__arg_t::__boolean) {}

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(_Tp __v) noexcept
      requires(same_as<_Tp, char_type> ||
               (same_as<_Tp, char> && same_as<char_type, wchar_t>))
      : __char_type(__v), __type_(__format::__arg_t::__char_type) {}

  template <__libcpp_signed_integer _Tp>
  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(_Tp __v) noexcept {
    if constexpr (sizeof(_Tp) <= sizeof(int)) {
      __int = static_cast<int>(__v);
      __type_ = __format::__arg_t::__int;
    } else if constexpr (sizeof(_Tp) <= sizeof(long long)) {
      __long_long = static_cast<long long>(__v);
      __type_ = __format::__arg_t::__long_long;
    }
#ifndef _LIBCPP_HAS_NO_INT128
    else if constexpr (sizeof(_Tp) == sizeof(__int128_t)) {
      __i128 = __v;
      __type_ = __format::__arg_t::__i128;
    }
#endif
    else
      static_assert(sizeof(_Tp) == 0, "An unsupported signed integer was used");
  }

  template <__libcpp_unsigned_integer _Tp>
  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(_Tp __v) noexcept {
    if constexpr (sizeof(_Tp) <= sizeof(unsigned)) {
      __unsigned = static_cast<unsigned>(__v);
      __type_ = __format::__arg_t::__unsigned;
    } else if constexpr (sizeof(_Tp) <= sizeof(unsigned long long)) {
      __unsigned_long_long = static_cast<unsigned long long>(__v);
      __type_ = __format::__arg_t::__unsigned_long_long;
    }
#ifndef _LIBCPP_HAS_NO_INT128
    else if constexpr (sizeof(_Tp) == sizeof(__int128_t)) {
      __u128 = __v;
      __type_ = __format::__arg_t::__u128;
    }
#endif
    else
      static_assert(sizeof(_Tp) == 0,
                    "An unsupported unsigned integer was used");
  }

  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(float __v) noexcept
      : __float(__v), __type_(__format::__arg_t::__float) {}

  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(double __v) noexcept
      : __double(__v), __type_(__format::__arg_t::__double) {}

  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(long double __v) noexcept
      : __long_double(__v), __type_(__format::__arg_t::__long_double) {}

  // Note not a 'noexcept' function.
  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(const char_type* __s)
      : __const_char_type_ptr(__s),
        __type_(__format::__arg_t::__const_char_type_ptr) {
    _LIBCPP_ASSERT(__s, "Used a nullptr argument to initialize a C-string");
  }

  template <class _Traits>
  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(
      basic_string_view<char_type, _Traits> __s) noexcept
      : __string_view{__s.data(), __s.size()},
        __type_(__format::__arg_t::__string_view) {}

  template <class _Traits, class _Allocator>
  _LIBCPP_HIDE_FROM_ABI explicit basic_format_arg(
      const basic_string<char_type, _Traits, _Allocator>& __s) noexcept
      : __string_view{__s.data(), __s.size()},
        __type_(__format::__arg_t::__string_view) {}

  _LIBCPP_HIDE_FROM_ABI
  explicit basic_format_arg(nullptr_t) noexcept
      : __ptr(nullptr), __type_(__format::__arg_t::__ptr) {}

  // TODO FMT Implement the _Tp* constructor.
};

#endif // !defined(_LIBCPP_HAS_NO_CONCEPTS)

#endif //_LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FORMAT_FORMAT_ARG_H
