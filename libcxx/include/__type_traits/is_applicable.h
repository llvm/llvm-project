//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_APPLICABLE_H
#define _LIBCPP___TYPE_TRAITS_IS_APPLICABLE_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__fwd/get.h>
#include <__tuple/tuple_like.h>
#include <__tuple/tuple_size.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/invoke.h>
#include <__type_traits/remove_reference.h>
#include <__utility/declval.h>
#include <__utility/integer_sequence.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <class _Fn, class _Tuple>
struct __apply_result_disabled_base {};

template <class _Fn, class _Tuple, class _Tp>
struct __apply_result_enabled_base {
  using type _LIBCPP_NODEBUG = _Tp;
};

template <bool _Applicable, bool _Nothrow, class _Tp>
struct __applicability_traits {
  static constexpr bool __applicable         = true;
  static constexpr bool __nothrow_applicable = _Nothrow;

  template <class _Fn, class _Tuple>
  using __base_type _LIBCPP_NODEBUG = __apply_result_enabled_base<_Fn, _Tuple, _Tp>;
};

template <bool _Nothrow, class _Tp>
struct __applicability_traits<false, _Nothrow, _Tp> {
  static_assert(!_Nothrow, "misspecified [_Applicable = false, _Nothrow = true]");
  static constexpr bool __applicable         = false;
  static constexpr bool __nothrow_applicable = false;

  template <class _Fn, class _Tuple>
  using __base_type _LIBCPP_NODEBUG = __apply_result_disabled_base<_Fn, _Tuple>;
};

template <class _Fn, class _Tuple, size_t... _Is>
concept __tuple_applicable_impl = requires(_Tuple&& __tuple) {
  [](auto&&...) {}(std::get<_Is>(static_cast<_Tuple&&>(__tuple))...);
} && __is_invocable_v<_Fn, decltype(std::get<_Is>(std::declval<_Tuple>()))...>;

template <class _Fn, class _Tuple, size_t... _Is>
concept __tuple_nothrow_applicable_impl = requires(_Tuple&& __tuple) {
  {
    [](auto&&...) noexcept {}(std::get<_Is>(static_cast<_Tuple&&>(__tuple))...)
  } noexcept;
} && __is_nothrow_invocable_v<_Fn, decltype(std::get<_Is>(std::declval<_Tuple>()))...>;

template <class _Fn, class _Tuple>
consteval auto __applicability_traits_of() {
  if constexpr (__tuple_like<_Tuple>)
    return []<size_t... _Is>(index_sequence<_Is...>) {
      if constexpr (__tuple_applicable_impl<_Fn, _Tuple, _Is...>) {
        return __applicability_traits<true,
                                      __tuple_nothrow_applicable_impl<_Fn, _Tuple, _Is...>,
                                      __invoke_result_t<_Fn, decltype(std::get<_Is>(std::declval<_Tuple>()))...>>{};
      } else
        return __applicability_traits<false, false, void>{};
    }(make_index_sequence<tuple_size_v<remove_reference_t<_Tuple>>>{});
  else
    return __applicability_traits<false, false, void>{};
}

template <class _Fn, class _Tuple>
struct _LIBCPP_NO_SPECIALIZATIONS is_applicable
    : bool_constant<decltype(std::__applicability_traits_of<_Fn, _Tuple>())::__applicable> {};

template <class _Fn, class _Tuple>
struct _LIBCPP_NO_SPECIALIZATIONS is_nothrow_applicable
    : bool_constant<decltype(std::__applicability_traits_of<_Fn, _Tuple>())::__nothrow_applicable> {};

template <class _Fn, class _Tuple>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_applicable_v =
    decltype(std::__applicability_traits_of<_Fn, _Tuple>())::__applicable;

template <class _Fn, class _Tuple>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_nothrow_applicable_v =
    decltype(std::__applicability_traits_of<_Fn, _Tuple>())::__nothrow_applicable;

template <class _Fn, class _Tuple>
struct _LIBCPP_NO_SPECIALIZATIONS apply_result
    : decltype(std::__applicability_traits_of<_Fn, _Tuple>())::template __base_type<_Fn, _Tuple> {};

template <class _Fn, class _Tuple>
using apply_result_t = apply_result<_Fn, _Tuple>::type;

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_APPLICABLE_H
