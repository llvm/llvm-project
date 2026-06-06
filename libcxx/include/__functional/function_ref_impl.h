//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header is unguarded on purpose. This header is an implementation detail of function_ref.h
// and generates multiple versions of std::function_ref

#include <__assert>
#include <__config>
#include <__functional/function_ref_common.h>
#include <__functional/invoke.h>
#include <__memory/addressof.h>
#include <__type_traits/conditional.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_void.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/remove_pointer.h>
#include <__type_traits/remove_reference.h>
#include <__utility/constant_wrapper.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifndef _LIBCPP___FUNCTIONAL_FUNCTION_REF_H
#  error This header should only be included from function_ref.h
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <class...>
class function_ref;

template <bool _NoExcept1, bool _NoExcept2, class _Rp, class... _ArgTypes>
struct __is_convertible_from_specialization<
    function_ref<_Rp(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV noexcept(_NoExcept1)>,
    function_ref<_Rp(_ArgTypes...) const noexcept(_NoExcept2)> >
    : is_convertible<_Rp (&)(_ArgTypes...) noexcept(_NoExcept2), _Rp (&)(_ArgTypes...) noexcept(_NoExcept1)> {};

template <bool _NoExcept1, bool _NoExcept2, class _Rp, class... _ArgTypes>
struct __is_convertible_from_specialization<
    function_ref<_Rp(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV noexcept(_NoExcept1)>,
    function_ref<_Rp(_ArgTypes...) noexcept(_NoExcept2)> >
    : _And<is_convertible<_Rp (&)(_ArgTypes...) noexcept(_NoExcept2), _Rp (&)(_ArgTypes...) noexcept(_NoExcept1)>,
           is_convertible<_LIBCPP_FUNCTION_REF_CV int&, int&>> {};

template <class _Rp, class... _ArgTypes, bool __is_noexcept>
class function_ref<_Rp(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV noexcept(__is_noexcept)> {
private:
  template <class... _Tp>
  static constexpr bool __is_invocable_using =
      _If<__is_noexcept, is_nothrow_invocable_r<_Rp, _Tp..., _ArgTypes...>, is_invocable_r<_Rp, _Tp..., _ArgTypes...>>::
          value;

  template <class _Fn2>
  static constexpr bool __is_convertible_from_specialization_v =
      __is_convertible_from_specialization<function_ref, _Fn2>::value;

  template <class... _Tp>
  friend class function_ref;

  template <class _Arg>
  using __arg_t _LIBCPP_NODEBUG = _If<__register_passable<_Arg>, _Arg, _Arg&&>;

  using __storage_t _LIBCPP_NODEBUG = __function_ref_storage;

  using __call_t _LIBCPP_NODEBUG = _Rp (*)(__storage_t, __arg_t<_ArgTypes>...) noexcept(__is_noexcept);

  __storage_t __storage_;
  __call_t __call_;

public:
  template <class _Fp>
    requires is_function_v<_Fp> && __is_invocable_using<_Fp>
  _LIBCPP_HIDE_FROM_ABI function_ref(_Fp* __fn_ptr) noexcept
      : __storage_(__fn_ptr),
        __call_([](__storage_t __storage, __arg_t<_ArgTypes>... __args) static noexcept(__is_noexcept) -> _Rp {
          return __storage_t::template __get<_Fp>(__storage)(std::forward<__arg_t<_ArgTypes>>(__args)...);
        }) {
    _LIBCPP_ASSERT_NON_NULL(__fn_ptr != nullptr, "the function pointer should not be a nullptr");
  }

  template <class _Fn, class _Tp = remove_reference_t<_Fn>>
    requires(!is_same_v<remove_cvref_t<_Fn>, function_ref> && !is_member_pointer_v<_Tp> &&
             __is_invocable_using<_LIBCPP_FUNCTION_REF_CV _Tp&> && !__is_convertible_from_specialization_v<_Tp>)
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(_Fn&& __obj) noexcept {
    using _Dn = remove_cv_t<_Tp>;
    if constexpr (__statically_callable<_Dn, __arg_t<_ArgTypes>...>) {
      __call_ = [](__storage_t, __arg_t<_ArgTypes>... __args) static noexcept(__is_noexcept) -> _Rp {
        return _Dn::operator()(std::forward<__arg_t<_ArgTypes>>(__args)...);
      };
    } else {
      __storage_ = __storage_t(std::addressof(__obj)),
      __call_    = [](__storage_t __storage, __arg_t<_ArgTypes>... __args) static noexcept(__is_noexcept) -> _Rp {
        _LIBCPP_FUNCTION_REF_CV _Tp& __obj1 = *__storage_t::template __get<_Tp>(__storage);
        return __obj1(std::forward<__arg_t<_ArgTypes>>(__args)...);
      };
    }
  }

  template <class _Fn, class _Tp = remove_reference_t<_Fn>>
    requires(!is_same_v<remove_cvref_t<_Fn>, function_ref> && !is_member_pointer_v<_Tp> &&
             __is_invocable_using<_LIBCPP_FUNCTION_REF_CV _Tp&> && __is_convertible_from_specialization_v<_Tp>)
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(_Fn&& __obj) noexcept
      : __storage_(__obj.__storage_), __call_(__obj.__call_) {}

  template <auto _Cw, class _Fn>
    requires __is_invocable_using<const _Fn&>
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(constant_wrapper<_Cw, _Fn> __f) noexcept
      : __call_([](__storage_t, __arg_t<_ArgTypes>... __args) static noexcept(__is_noexcept) -> _Rp {
          return std::invoke_r<_Rp>(decltype(__f)::value, std::forward<__arg_t<_ArgTypes>>(__args)...);
        }) {
    if constexpr (is_pointer_v<_Fn> || is_member_pointer_v<_Fn>) {
      static_assert(__f.value != nullptr, "the function pointer should not be a nullptr");
    }
    if constexpr (sizeof...(_ArgTypes) > 0 && (__constexpr_param<remove_cvref_t<_ArgTypes>> && ...)) {
      static_assert(
          !requires {
            typename constant_wrapper<std::invoke(decltype(__f)::value, remove_cvref_t<_ArgTypes>::value...)>;
          },
          "cw(args...) should be equivalent to fn(args...), otherwise the intended behavior for a function_ref "
          "constructed from cw would be ambiguous");
    }
  }

  template <auto _Cw, class _Fn, class _Up, class _Tp = remove_reference_t<_Up>>
    requires(!is_rvalue_reference_v<_Up &&>) && __is_invocable_using<const _Fn&, _LIBCPP_FUNCTION_REF_CV _Tp&>
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(constant_wrapper<_Cw, _Fn> __f, _Up&& __obj) noexcept
      : __storage_(std::addressof(__obj)),
        __call_([](__storage_t __storage, __arg_t<_ArgTypes>... __args) static noexcept(__is_noexcept) -> _Rp {
          _LIBCPP_FUNCTION_REF_CV _Tp& __obj1 = *__storage_t::template __get<_Tp>(__storage);
          return std::invoke_r<_Rp>(decltype(__f)::value, __obj1, std::forward<__arg_t<_ArgTypes>>(__args)...);
        }) {
    if constexpr (is_pointer_v<_Fn> || is_member_pointer_v<_Fn>) {
      static_assert(__f.value != nullptr, "the function pointer should not be a nullptr");
    }
  }

  template <auto _Cw, class _Fn, class _Tp>
    requires __is_invocable_using<const _Fn&, _LIBCPP_FUNCTION_REF_CV _Tp*>
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(constant_wrapper<_Cw, _Fn> __f,
                                               _LIBCPP_FUNCTION_REF_CV _Tp* __obj_ptr_) noexcept
      : __storage_(__obj_ptr_),
        __call_([](__storage_t __storage, __arg_t<_ArgTypes>... __args) static noexcept(__is_noexcept) -> _Rp {
          auto* __obj = __storage_t::template __get<_LIBCPP_FUNCTION_REF_CV _Tp>(__storage);
          return std::invoke_r<_Rp>(decltype(__f)::value, __obj, std::forward<__arg_t<_ArgTypes>>(__args)...);
        }) {
    if constexpr (is_pointer_v<_Fn> || is_member_pointer_v<_Fn>) {
      static_assert(__f.value != nullptr, "the function pointer should not be a nullptr");
    }

    if constexpr (is_member_pointer_v<_Fn>) {
      _LIBCPP_ASSERT_NON_NULL(__obj_ptr_ != nullptr, "the object pointer should not be a nullptr");
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(const function_ref&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI constexpr function_ref& operator=(const function_ref&) noexcept = default;

  template <class _Tp>
    requires(!__is_convertible_from_specialization_v<_Tp>) && (!is_pointer_v<_Tp>) && (!__is_constant_wrapper<_Tp>)
  _LIBCPP_HIDE_FROM_ABI function_ref& operator=(_Tp) = delete;

  _LIBCPP_HIDE_FROM_ABI _Rp operator()(_ArgTypes... __args) const noexcept(__is_noexcept) {
    return __call_(__storage_, std::forward<_ArgTypes>(__args)...);
  }
};

template <class _Tp, class _Rp, class _Gp, class... _ArgTypes, bool __is_noexcept>
struct __function_ref_bind<_Rp (_Gp::*)(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV noexcept(__is_noexcept), _Tp> {
  using type _LIBCPP_NODEBUG = _Rp(_ArgTypes...) noexcept(__is_noexcept);
};

template <class _Tp, class _Rp, class _Gp, class... _ArgTypes, bool __is_noexcept>
struct __function_ref_bind<_Rp (_Gp::*)(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV & noexcept(__is_noexcept), _Tp> {
  using type _LIBCPP_NODEBUG = _Rp(_ArgTypes...) noexcept(__is_noexcept);
};

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD
