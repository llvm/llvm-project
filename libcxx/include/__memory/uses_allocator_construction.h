//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_USES_ALLOCATOR_CONSTRUCTION_H
#define _LIBCPP___MEMORY_USES_ALLOCATOR_CONSTRUCTION_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/reference_wrapper.h>
#include <__memory/construct_at.h>
#include <__memory/uses_allocator.h>
#include <__tuple/tuple_like_no_subrange.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/declval.h>
#include <__utility/integer_sequence.h>
#include <__utility/pair.h>
#include <__utility/piecewise_construct.h>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_CXX03_LANG) && _LIBCPP_STD_VER < 14

template <class _Alloc, class... _Args, size_t... _Is>
_LIBCPP_HIDE_FROM_ABI void __transform_tuple_using_allocator_impl(
    integral_constant<int, -1>, const _Alloc&, tuple<_Args...>&&, __index_sequence<_Is...>) {
  static_assert(false, "If uses_allocator_v<T, A> is true, T has to be allocator-constructible");
}

template <class _Alloc, class... _Args, size_t... _Is>
_LIBCPP_HIDE_FROM_ABI tuple<_Args&&...> __transform_tuple_using_allocator_impl(
    integral_constant<int, 0>, const _Alloc&, tuple<_Args...>&& __t, __index_sequence<_Is...>) {
  return tuple<_Args&&...>(std::move(__t));
}

template <class _Alloc, class... _Args, size_t... _Is>
_LIBCPP_HIDE_FROM_ABI tuple<allocator_arg_t, const _Alloc&, _Args&&...> __transform_tuple_using_allocator_impl(
    integral_constant<int, 1>, const _Alloc& __a, tuple<_Args...>&& __t, __index_sequence<_Is...>) {
  return tuple<allocator_arg_t, const _Alloc&, _Args&&...>(allocator_arg, __a, std::get<_Is>(std::move(__t))...);
}

template <class _Alloc, class... _Args, size_t... _Is>
_LIBCPP_HIDE_FROM_ABI tuple<_Args&&..., const _Alloc&> __transform_tuple_using_allocator_impl(
    integral_constant<int, 2>, const _Alloc& __a, tuple<_Args...>&& __t, __index_sequence<_Is...>) {
  return tuple<_Args&&..., const _Alloc&>(std::get<_Is>(std::move(__t))..., __a);
}

template <class _Tp, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI auto __transform_tuple_using_allocator(const _Alloc& __a, tuple<_Args...>&& __t)
    -> decltype(std::__transform_tuple_using_allocator_impl(
        __uses_alloc_ctor<_Tp, _Alloc, _Args...>{}, __a, std::move(__t), __make_index_sequence<sizeof...(_Args)>{})) {
  return std::__transform_tuple_using_allocator_impl(
      __uses_alloc_ctor<_Tp, _Alloc, _Args...>{}, __a, std::move(__t), __make_index_sequence<sizeof...(_Args)>{});
}

#endif // !defined(_LIBCPP_CXX03_LANG) && _LIBCPP_STD_VER < 14

#if _LIBCPP_STD_VER >= 14

template <class _Tp>
inline constexpr bool __is_cv_std_pair = __is_pair_v<remove_cv_t<_Tp>>;

template <class _Tp, class = void>
struct __uses_allocator_construction_args;

namespace __uses_allocator_detail {

template <class _Ap, class _Bp>
void __pair_taker(const pair<_Ap, _Bp>&);

template <class, class = void>
inline constexpr bool __convertible_to_const_pair_ref = false;
template <class _Tp>
inline constexpr bool
    __convertible_to_const_pair_ref<_Tp, decltype(__uses_allocator_detail::__pair_taker(std::declval<_Tp>()))> = true;

#  if _LIBCPP_STD_VER >= 23
template <class _Tp, class _Up>
inline constexpr bool __uses_allocator_constraints =
    __is_cv_std_pair<_Tp> && !__pair_like_no_subrange<_Up> && !__convertible_to_const_pair_ref<_Up>;
#  else
template <class _Tp, class _Up>
inline constexpr bool __uses_allocator_constraints = __is_cv_std_pair<_Tp> && !__convertible_to_const_pair_ref<_Up>;
#  endif

#  if _LIBCPP_STD_VER < 17
template <class _Tp>
struct __construction_fn {
  template <class... _Args>
  static _LIBCPP_HIDE_FROM_ABI constexpr _Tp operator()(_Args&&... __args) {
    static_assert(is_constructible<_Tp, _Args...>::value, "undesired C-style cast used");
    return _Tp(std::forward<_Args&&>(__args)...);
  }
};

template <class _Fn, class _Tuple, size_t... _Ip>
_LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) __apply_impl(_Fn&& __fn, _Tuple&& __t, index_sequence<_Ip...>) {
  return std::__invoke(std::forward<_Fn>(__fn), std::get<_Ip>(std::forward<_Tuple>(__t))...);
}
#  endif // _LIBCPP_STD_VER < 17

template <class _Fn, class _Tuple>
_LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) __apply(_Fn&& __fn, _Tuple&& __t) {
#  if _LIBCPP_STD_VER >= 17
  return std::apply(std::forward<_Fn>(__fn), std::forward<_Tuple>(__t));
#  else
  return __uses_allocator_detail::__apply_impl(
      std::forward<_Fn>(__fn),
      std::forward<_Tuple>(__t),
      std::make_index_sequence<tuple_size<__remove_cvref_t<_Tuple>>::value>{});
#  endif
}

template <class _Tp, class _Tuple>
_LIBCPP_HIDE_FROM_ABI constexpr _Tp __make_from_tuple(_Tuple&& __t) {
#  if _LIBCPP_STD_VER >= 17
  return std::make_from_tuple<_Tp>(std::forward<_Tuple>(__t));
#  else
  return __uses_allocator_detail::__apply_impl(
      __construction_fn<_Tp>{},
      std::forward<_Tuple>(__t),
      std::make_index_sequence<tuple_size<__remove_cvref_t<_Tuple>>::value>{});
#  endif
}

template <class _Tp, class... _Args, class = decltype(::new (std::declval<void*>()) _Tp(std::declval<_Args>()...))>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp* __construct_at_nocv(_Tp* __location, _Args&&... __args) {
  return std::__construct_at(const_cast<remove_cv_t<_Tp>*>(__location), std::forward<_Args>(__args)...);
}

} // namespace __uses_allocator_detail

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 _Type
__make_obj_using_allocator(const _Alloc& __alloc, _Args&&... __args);

template <class _Pair>
struct __uses_allocator_construction_args<_Pair, __enable_if_t<__is_cv_std_pair<_Pair>>> {
  template <class _Alloc, class _Tuple1, class _Tuple2>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 auto
  __apply(const _Alloc& __alloc, piecewise_construct_t, _Tuple1&& __x, _Tuple2&& __y) noexcept {
    return std::make_tuple(
#  if _LIBCPP_STD_VER >= 20
        piecewise_construct,
#  else  // _LIBCPP_STD_VER >= 20
        std::ref(piecewise_construct),
#  endif // _LIBCPP_STD_VER >= 20
        __uses_allocator_detail::__apply(
            [&__alloc](auto&&... __args1) {
              return __uses_allocator_construction_args<typename _Pair::first_type>::__apply(
                  __alloc, std::forward<decltype(__args1)>(__args1)...);
            },
            std::forward<_Tuple1>(__x)),
        __uses_allocator_detail::__apply(
            [&__alloc](auto&&... __args2) {
              return __uses_allocator_construction_args<typename _Pair::second_type>::__apply(
                  __alloc, std::forward<decltype(__args2)>(__args2)...);
            },
            std::forward<_Tuple2>(__y)));
  }

  template <class _Alloc>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 auto __apply(const _Alloc& __alloc) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(__alloc, piecewise_construct, tuple<>{}, tuple<>{});
  }

  template <class _Alloc, class _Up, class _Vp>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 auto
  __apply(const _Alloc& __alloc, _Up&& __u, _Vp&& __v) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(
        __alloc,
        piecewise_construct,
        std::forward_as_tuple(std::forward<_Up>(__u)),
        std::forward_as_tuple(std::forward<_Vp>(__v)));
  }

#  if _LIBCPP_STD_VER >= 23
  template <class _Alloc, class _Up, class _Vp>
  static _LIBCPP_HIDE_FROM_ABI constexpr auto __apply(const _Alloc& __alloc, pair<_Up, _Vp>& __pair) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(
        __alloc, piecewise_construct, std::forward_as_tuple(__pair.first), std::forward_as_tuple(__pair.second));
  }
#  endif

  template <class _Alloc, class _Up, class _Vp>
  static _LIBCPP_HIDE_FROM_ABI
  _LIBCPP_CONSTEXPR_SINCE_CXX17 auto __apply(const _Alloc& __alloc, const pair<_Up, _Vp>& __pair) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(
        __alloc, piecewise_construct, std::forward_as_tuple(__pair.first), std::forward_as_tuple(__pair.second));
  }

  template <class _Alloc, class _Up, class _Vp>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 auto
  __apply(const _Alloc& __alloc, pair<_Up, _Vp>&& __pair) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(
        __alloc,
        piecewise_construct,
        std::forward_as_tuple(std::get<0>(std::move(__pair))),
        std::forward_as_tuple(std::get<1>(std::move(__pair))));
  }

#  if _LIBCPP_STD_VER >= 23
  template <class _Alloc, class _Up, class _Vp>
  static _LIBCPP_HIDE_FROM_ABI constexpr auto __apply(const _Alloc& __alloc, const pair<_Up, _Vp>&& __pair) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(
        __alloc,
        piecewise_construct,
        std::forward_as_tuple(std::get<0>(std::move(__pair))),
        std::forward_as_tuple(std::get<1>(std::move(__pair))));
  }

  template < class _Alloc, __pair_like_no_subrange _PairLike>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 auto
  __apply(const _Alloc& __alloc, _PairLike&& __p) noexcept {
    return __uses_allocator_construction_args<_Pair>::__apply(
        __alloc,
        piecewise_construct,
        std::forward_as_tuple(std::get<0>(std::forward<_PairLike>(__p))),
        std::forward_as_tuple(std::get<1>(std::forward<_PairLike>(__p))));
  }
#  endif

  template <class _Alloc,
            class _Type,
            __enable_if_t<__uses_allocator_detail::__uses_allocator_constraints<_Pair, _Type>, int> = 0>
  static _LIBCPP_HIDE_FROM_ABI
  _LIBCPP_CONSTEXPR_SINCE_CXX17 auto __apply(const _Alloc& __alloc, _Type&& __value) noexcept {
    struct __pair_constructor {
      using _PairMutable = remove_cv_t<_Pair>;

      _LIBCPP_HIDDEN _LIBCPP_CONSTEXPR_SINCE_CXX17 auto __do_construct(const _PairMutable& __pair) const {
        return std::__make_obj_using_allocator<_PairMutable>(__alloc_, __pair);
      }

      _LIBCPP_HIDDEN _LIBCPP_CONSTEXPR_SINCE_CXX17 auto __do_construct(_PairMutable&& __pair) const {
        return std::__make_obj_using_allocator<_PairMutable>(__alloc_, std::move(__pair));
      }

      const _Alloc& __alloc_;
      _Type& __value_;

      _LIBCPP_HIDDEN _LIBCPP_CONSTEXPR_SINCE_CXX17 operator _PairMutable() const {
        return __do_construct(std::forward<_Type>(__value_));
      }
    };

    return std::make_tuple(__pair_constructor{__alloc, __value});
  }
};

template <class _Type>
struct __uses_allocator_construction_args<_Type, __enable_if_t<!__is_cv_std_pair<_Type>>> {
  template <class _Alloc, class... _Args>
  static _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 auto
  __apply(const _Alloc& __alloc, _Args&&... __args) noexcept {
    if constexpr (!uses_allocator<remove_cv_t<_Type>, _Alloc>::value && is_constructible<_Type, _Args...>::value) {
      return std::forward_as_tuple(std::forward<_Args>(__args)...);
    } else if constexpr (uses_allocator<remove_cv_t<_Type>, _Alloc>::value &&
                         is_constructible<_Type, allocator_arg_t, const _Alloc&, _Args...>::value) {
      return tuple<allocator_arg_t, const _Alloc&, _Args&&...>(allocator_arg, __alloc, std::forward<_Args>(__args)...);
    } else if constexpr (uses_allocator<remove_cv_t<_Type>, _Alloc>::value &&
                         is_constructible<_Type, _Args..., const _Alloc&>::value) {
      return std::forward_as_tuple(std::forward<_Args>(__args)..., __alloc);
    } else {
      static_assert(
          sizeof(_Type) + 1 == 0, "If uses_allocator_v<Type> is true, the type has to be allocator-constructible");
    }
  }
};

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 _Type
__make_obj_using_allocator(const _Alloc& __alloc, _Args&&... __args) {
  return __uses_allocator_detail::__make_from_tuple<_Type>(
      __uses_allocator_construction_args<_Type>::__apply(__alloc, std::forward<_Args>(__args)...));
}

#endif // _LIBCPP_STD_VER >= 14

#if _LIBCPP_STD_VER >= 17

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Type*
__uninitialized_construct_using_allocator(_Type* __ptr, const _Alloc& __alloc, _Args&&... __args) {
  return std::apply(
      [&__ptr](auto&&... __xs) { return std::__construct_at(__ptr, std::forward<decltype(__xs)>(__xs)...); },
      __uses_allocator_construction_args<_Type>::__apply(__alloc, std::forward<_Args>(__args)...));
}

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Type*
__uninitialized_construct_using_allocator_nocv(_Type* __ptr, const _Alloc& __alloc, _Args&&... __args) {
  return std::apply(
      [&__ptr](auto&&... __xs) {
        return __uses_allocator_detail::__construct_at_nocv(__ptr, std::forward<decltype(__xs)>(__xs)...);
      },
      __uses_allocator_construction_args<_Type>::__apply(__alloc, std::forward<_Args>(__args)...));
}

#endif // _LIBCPP_STD_VER >= 17

#if _LIBCPP_STD_VER >= 20

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI constexpr auto uses_allocator_construction_args(const _Alloc& __alloc, _Args&&... __args) noexcept
    -> decltype(__uses_allocator_construction_args<_Type>::__apply(__alloc, std::forward<_Args>(__args)...)) {
  return /*--*/ __uses_allocator_construction_args<_Type>::__apply(__alloc, std::forward<_Args>(__args)...);
}

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI constexpr auto make_obj_using_allocator(const _Alloc& __alloc, _Args&&... __args)
    -> decltype(std::__make_obj_using_allocator<_Type>(__alloc, std::forward<_Args>(__args)...)) {
  return /*--*/ std::__make_obj_using_allocator<_Type>(__alloc, std::forward<_Args>(__args)...);
}

template <class _Type, class _Alloc, class... _Args>
_LIBCPP_HIDE_FROM_ABI constexpr auto
uninitialized_construct_using_allocator(_Type* __ptr, const _Alloc& __alloc, _Args&&... __args)
    -> decltype(std::__uninitialized_construct_using_allocator(__ptr, __alloc, std::forward<_Args>(__args)...)) {
  return /*--*/ std::__uninitialized_construct_using_allocator(__ptr, __alloc, std::forward<_Args>(__args)...);
}

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_USES_ALLOCATOR_CONSTRUCTION_H
