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
#include <__type_traits/is_const.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_trivially_constructible.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_pointer.h>
#include <__utility/constant_wrapper.h>
#include <__utility/declval.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <class...>
class function_ref;

template <class _Fn1, class _Fn2>
struct __is_convertible_from_specialization : false_type {};

union __function_ref_storage {
  void* __obj_ptr_;
  void (*__fn_ptr_)();

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __function_ref_storage() noexcept : __obj_ptr_(nullptr) {}

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit __function_ref_storage(_Tp* __ptr) noexcept {
    if constexpr (is_object_v<_Tp>) {
      __obj_ptr_ = const_cast<remove_cv_t<_Tp>*>(__ptr);
    } else {
      static_assert(is_function_v<_Tp>);
      __fn_ptr_ = reinterpret_cast<void (*)()>(__ptr);
    }
  }

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __get(__function_ref_storage __storage) {
    if constexpr (is_object_v<_Tp>) {
      return static_cast<_Tp*>(__storage.__obj_ptr_);
    } else {
      static_assert(is_function_v<_Tp>);
      return reinterpret_cast<_Tp*>(__storage.__fn_ptr_);
    }
  }
};

template <class _Fp, class _Tp>
struct __function_ref_bind {};

// F is of the form R(*)(G, A...) noexcept(E) for a type G.
template <bool _Noexcept, class _Tp, class _Rp, class _Gp, class... _ArgTypes>
struct __function_ref_bind<_Rp (*)(_Gp, _ArgTypes...) noexcept(_Noexcept), _Tp> {
  using type _LIBCPP_NODEBUG = _Rp(_ArgTypes...) noexcept(_Noexcept);
};

template <class _Tp, class _Mp, class _Gp>
  requires is_object_v<_Mp>
struct __function_ref_bind<_Mp _Gp::*, _Tp> {
  using type _LIBCPP_NODEBUG = invoke_result_t<_Mp _Gp::*, _Tp&>() noexcept;
};

template <class _Fp, class _Tp>
using __function_ref_bind_t _LIBCPP_NODEBUG = __function_ref_bind<_Fp, _Tp>::type;

template <class _Fp>
  requires is_function_v<_Fp>
function_ref(_Fp*) -> function_ref<_Fp>;

template <auto _Cw, class _Fn>
  requires is_function_v<remove_pointer_t<_Fn>>
function_ref(constant_wrapper<_Cw, _Fn>) -> function_ref<remove_pointer_t<_Fn>>;

template <auto _Cw, class _Fn, class _Tp>
function_ref(constant_wrapper<_Cw, _Fn>, _Tp&&) -> function_ref<__function_ref_bind_t<_Fn, _Tp&>>;

template <class>
constexpr bool __is_constant_wrapper = false;

template <auto _Value>
constexpr bool __is_constant_wrapper<constant_wrapper<_Value>> = true;

template <class _Arg>
concept __itanium_trivial_for_calls =
    is_trivially_destructible_v<_Arg> && is_trivially_copy_constructible_v<_Arg> &&
    is_trivially_move_constructible_v<_Arg>;

template <class _Arg>
concept __register_passable =
    !is_reference_v<_Arg> && sizeof(_Arg) <= 2 * sizeof(void*) && __itanium_trivial_for_calls<_Arg>;

template <class _Fn, class... _Args>
concept __statically_callable = requires { _Fn::operator()(std::declval<_Args>()...); };

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FUNCTIONAL_FUNCTION_REF_COMMON_H
