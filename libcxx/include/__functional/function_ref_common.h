//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FUNCTIONAL_FUNCTION_REF_COMMON_H
#define _LIBCPP___FUNCTIONAL_FUNCTION_REF_COMMON_H

#include <__config>
#include <__type_traits/invoke.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_object.h>
#include <__type_traits/remove_pointer.h>
#include <__utility/nontype.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <class...>
class function_ref;

template <class>
inline constexpr bool __is_function_ref = false;

template <class _Rp, class... _ArgTypes>
inline constexpr bool __is_function_ref<function_ref<_Rp, _ArgTypes...>> = true;

template <class _Fp, class _Tp>
struct __function_ref_bind {};

template <class _Tp, class _Rp, class _Gp, class... _ArgTypes>
struct __function_ref_bind<_Rp (*)(_Gp, _ArgTypes...), _Tp> {
  using type = _Rp(_ArgTypes...);
};

template <class _Tp, class _Rp, class _Gp, class... _ArgTypes>
struct __function_ref_bind<_Rp (*)(_Gp, _ArgTypes...) noexcept, _Tp> {
  using type = _Rp(_ArgTypes...) noexcept;
};

template <class _Tp, class _Mp, class _Gp>
  requires is_object_v<_Mp>
struct __function_ref_bind<_Mp _Gp::*, _Tp> {
  using type = invoke_result_t<_Mp _Gp::*, _Tp&>();
};

template <class _Fp, class _Tp>
using __function_ref_bind_t = __function_ref_bind<_Fp, _Tp>::type;

template <class _Fp>
  requires is_function_v<_Fp>
function_ref(_Fp*) -> function_ref<_Fp>;

template <auto _Fn>
  requires is_function_v<remove_pointer_t<decltype(_Fn)>>
function_ref(nontype_t<_Fn>) -> function_ref<remove_pointer_t<decltype(_Fn)>>;

template <auto _Fn, class _Tp>
function_ref(nontype_t<_Fn>, _Tp&&) -> function_ref<__function_ref_bind_t<decltype(_Fn), _Tp&>>;

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FUNCTIONAL_FUNCTION_REF_COMMON_H
