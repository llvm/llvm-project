//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::optional<T>

#ifndef _LIBCPP_OPTIONAL_OPTIONAL_H
#define _LIBCPP_OPTIONAL_OPTIONAL_H

#include <__concepts/invocable.h>
#include <__config>
#include <__functional/invoke.h>
#include <__optional/common.h>
#include <__optional/nullopt_t.h>
#include <__tuple/sfinae_helpers.h>
#include <__type_traits/add_pointer.h>
#include <__type_traits/conditional.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/decay.h>
#include <__type_traits/disjunction.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/is_destructible.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_scalar.h>
#include <__type_traits/is_swappable.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__type_traits/negation.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/remove_reference.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>

#include <initializer_list>

#include <__optional/comparison.h>
#include <__optional/hash.h>
#include <__optional/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class _LIBCPP_DECLSPEC_EMPTY_BASES optional
    : public __optional_iterator_base<_Tp>,
      private __optional_sfinae_ctor_base_t<_Tp>,
      private __optional_sfinae_assign_base_t<_Tp> {
  using __base _LIBCPP_NODEBUG = __optional_iterator_base<_Tp>;

public:
  using value_type = __libcpp_remove_reference_t<_Tp>;

  using __trivially_relocatable _LIBCPP_NODEBUG =
      conditional_t<__libcpp_is_trivially_relocatable<_Tp>::value, optional, void>;

private:
  static_assert(!is_same_v<remove_cv_t<_Tp>, in_place_t>, "instantiation of optional with in_place_t is ill-formed");
  static_assert(!is_same_v<remove_cv_t<_Tp>, nullopt_t>, "instantiation of optional with nullopt_t is ill-formed");
#  if _LIBCPP_STD_VER >= 26
  static_assert(!is_rvalue_reference_v<_Tp>, "instantiation of optional with an rvalue reference type is ill-formed");
#  else
  static_assert(!is_reference_v<_Tp>, "instantiation of optional with a reference type is ill-formed");
#  endif
  static_assert(is_destructible_v<_Tp>, "instantiation of optional with a non-destructible type is ill-formed");
  static_assert(!is_array_v<_Tp>, "instantiation of optional with an array type is ill-formed");

  // LWG2756: conditionally explicit conversion from _Up
  struct _CheckOptionalArgsConstructor {
    template <class _Up>
    _LIBCPP_HIDE_FROM_ABI static constexpr bool __enable_implicit() {
      return is_constructible_v<_Tp, _Up&&> && is_convertible_v<_Up&&, _Tp>;
    }

    template <class _Up>
    _LIBCPP_HIDE_FROM_ABI static constexpr bool __enable_explicit() {
      return is_constructible_v<_Tp, _Up&&> && !is_convertible_v<_Up&&, _Tp>;
    }
  };
  template <class _Up>
  using _CheckOptionalArgsCtor _LIBCPP_NODEBUG =
      _If< _IsNotSame<__remove_cvref_t<_Up>, in_place_t>::value && _IsNotSame<__remove_cvref_t<_Up>, optional>::value &&
               (!is_same_v<remove_cv_t<_Tp>, bool> || !__is_std_optional<__remove_cvref_t<_Up>>::value),
           _CheckOptionalArgsConstructor,
           __check_tuple_constructor_fail >;
  template <class _QualUp>
  struct _CheckOptionalLikeConstructor {
    template <class _Up, class _Opt = optional<_Up>>
    using __check_constructible_from_opt _LIBCPP_NODEBUG =
        _Or< is_constructible<_Tp, _Opt&>,
             is_constructible<_Tp, _Opt const&>,
             is_constructible<_Tp, _Opt&&>,
             is_constructible<_Tp, _Opt const&&>,
             is_convertible<_Opt&, _Tp>,
             is_convertible<_Opt const&, _Tp>,
             is_convertible<_Opt&&, _Tp>,
             is_convertible<_Opt const&&, _Tp> >;
    template <class _Up, class _Opt = optional<_Up>>
    using __check_assignable_from_opt _LIBCPP_NODEBUG =
        _Or< is_assignable<_Tp&, _Opt&>,
             is_assignable<_Tp&, _Opt const&>,
             is_assignable<_Tp&, _Opt&&>,
             is_assignable<_Tp&, _Opt const&&> >;
    template <class _Up, class _QUp = _QualUp>
    _LIBCPP_HIDE_FROM_ABI static constexpr bool __enable_implicit() {
      return is_convertible<_QUp, _Tp>::value &&
             (is_same_v<remove_cv_t<_Tp>, bool> || !__check_constructible_from_opt<_Up>::value);
    }
    template <class _Up, class _QUp = _QualUp>
    _LIBCPP_HIDE_FROM_ABI static constexpr bool __enable_explicit() {
      return !is_convertible<_QUp, _Tp>::value &&
             (is_same_v<remove_cv_t<_Tp>, bool> || !__check_constructible_from_opt<_Up>::value);
    }
    template <class _Up, class _QUp = _QualUp>
    _LIBCPP_HIDE_FROM_ABI static constexpr bool __enable_assign() {
      // Construction and assignability of _QUp to _Tp has already been
      // checked.
      return !__check_constructible_from_opt<_Up>::value && !__check_assignable_from_opt<_Up>::value;
    }
  };

  template <class _Up, class _QualUp>
  using _CheckOptionalLikeCtor _LIBCPP_NODEBUG =
      _If< _And< _IsNotSame<_Up, _Tp>, is_constructible<_Tp, _QualUp> >::value,
           _CheckOptionalLikeConstructor<_QualUp>,
           __check_tuple_constructor_fail >;
  template <class _Up, class _QualUp>
  using _CheckOptionalLikeAssign _LIBCPP_NODEBUG =
      _If< _And< _IsNotSame<_Up, _Tp>, is_constructible<_Tp, _QualUp>, is_assignable<_Tp&, _QualUp> >::value,
           _CheckOptionalLikeConstructor<_QualUp>,
           __check_tuple_constructor_fail >;

public:
  _LIBCPP_HIDE_FROM_ABI constexpr optional() noexcept {}
  _LIBCPP_HIDE_FROM_ABI constexpr optional(const optional&) = default;
  _LIBCPP_HIDE_FROM_ABI constexpr optional(optional&&)      = default;
  _LIBCPP_HIDE_FROM_ABI constexpr optional(nullopt_t) noexcept {}

  template <
      class _InPlaceT,
      class... _Args,
      enable_if_t<_And<_IsSame<_InPlaceT, in_place_t>, __is_constructible_for_optional<_Tp, _Args...>>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit optional(_InPlaceT, _Args&&... __args)
      : __base(in_place, std::forward<_Args>(__args)...) {}

  template <class _Up,
            class... _Args,
            enable_if_t<__is_constructible_for_optional_initializer_list_v<_Tp, _Up, _Args...>, int> = 0>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit optional(in_place_t, initializer_list<_Up> __il, _Args&&... __args)
      : __base(in_place, __il, std::forward<_Args>(__args)...) {}

  template <class _Up = _Tp, enable_if_t<_CheckOptionalArgsCtor<_Up>::template __enable_implicit<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI constexpr optional(_Up&& __v) : __base(in_place, std::forward<_Up>(__v)) {}

  template <class _Up                                                                        = remove_cv_t<_Tp>,
            enable_if_t<_CheckOptionalArgsCtor<_Up>::template __enable_explicit<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit optional(_Up&& __v) : __base(in_place, std::forward<_Up>(__v)) {}

  // LWG2756: conditionally explicit conversion from const optional<_Up>&
  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up const&>::template __enable_implicit<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 optional(const optional<_Up>& __v) {
    this->__construct_from(__v);
  }

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up const&>::template __enable_explicit<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 explicit optional(const optional<_Up>& __v) {
    this->__construct_from(__v);
  }

  // LWG2756: conditionally explicit conversion from optional<_Up>&&
  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::template __enable_implicit<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 optional(optional<_Up>&& __v) {
    this->__construct_from(std::move(__v));
  }

  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::template __enable_explicit<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 explicit optional(optional<_Up>&& __v) {
    this->__construct_from(std::move(__v));
  }

#  if _LIBCPP_STD_VER >= 23
  template <class _Tag,
            class _Fp,
            class... _Args,
            enable_if_t<_IsSame<_Tag, __optional_construct_from_invoke_tag>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit optional(_Tag, _Fp&& __f, _Args&&... __args)
      : __base(__optional_construct_from_invoke_tag{}, std::forward<_Fp>(__f), std::forward<_Args>(__args)...) {}
#  endif

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 optional& operator=(nullopt_t) noexcept {
    reset();
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr optional& operator=(const optional&) = default;
  _LIBCPP_HIDE_FROM_ABI constexpr optional& operator=(optional&&)      = default;

  // LWG2756
  template <class _Up        = remove_cv_t<_Tp>,
            enable_if_t<_And<_IsNotSame<__remove_cvref_t<_Up>, optional>,
                             _Or<_IsNotSame<__remove_cvref_t<_Up>, _Tp>, _Not<is_scalar<_Tp>>>,
                             is_constructible<_Tp, _Up>,
                             is_assignable<_Tp&, _Up>>::value,
                        int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 optional& operator=(_Up&& __v) {
    if (this->has_value())
      this->__get() = std::forward<_Up>(__v);
    else
      this->__construct(std::forward<_Up>(__v));
    return *this;
  }

  // LWG2756
  template <class _Up, enable_if_t<_CheckOptionalLikeAssign<_Up, _Up const&>::template __enable_assign<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 optional& operator=(const optional<_Up>& __v) {
    this->__assign_from(__v);
    return *this;
  }

  // LWG2756
  template <class _Up, enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::template __enable_assign<_Up>(), int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 optional& operator=(optional<_Up>&& __v) {
    this->__assign_from(std::move(__v));
    return *this;
  }

  template <class... _Args, enable_if_t<__is_constructible_for_optional_v<_Tp, _Args...>, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp& emplace(_Args&&... __args) {
    reset();
    this->__construct(std::forward<_Args>(__args)...);
    return this->__get();
  }

  template <class _Up,
            class... _Args,
            enable_if_t<__is_constructible_for_optional_initializer_list_v<_Tp, _Up, _Args...>, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp& emplace(initializer_list<_Up> __il, _Args&&... __args) {
    reset();
    this->__construct(__il, std::forward<_Args>(__args)...);
    return this->__get();
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
  swap(optional& __opt) noexcept((is_nothrow_move_constructible_v<_Tp> && is_nothrow_swappable_v<_Tp>)) {
    this->__swap(__opt);
  }

  using __base::operator*;
  using __base::operator->;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit operator bool() const noexcept { return has_value(); }

  using __base::__get;
  using __base::has_value;
  using __base::value;

  template <class _Up = remove_cv_t<_Tp>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp value_or(_Up&& __v) const& {
    static_assert(is_copy_constructible_v<_Tp>, "optional<T>::value_or: T must be copy constructible");
    static_assert(is_convertible_v<_Up, _Tp>, "optional<T>::value_or: U must be convertible to T");
    return this->has_value() ? this->__get() : static_cast<_Tp>(std::forward<_Up>(__v));
  }

  template <class _Up = remove_cv_t<_Tp>>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp value_or(_Up&& __v) && {
    static_assert(is_move_constructible_v<_Tp>, "optional<T>::value_or: T must be move constructible");
    static_assert(is_convertible_v<_Up, _Tp>, "optional<T>::value_or: U must be convertible to T");
    return this->has_value() ? std::move(this->__get()) : static_cast<_Tp>(std::forward<_Up>(__v));
  }

#  if _LIBCPP_STD_VER >= 23
  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto and_then(_Func&& __f) & {
    using _Up = invoke_result_t<_Func, _Tp&>;
    static_assert(__is_std_optional<remove_cvref_t<_Up>>::value,
                  "Result of f(value()) must be a specialization of std::optional");
    if (*this)
      return std::invoke(std::forward<_Func>(__f), value());
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto and_then(_Func&& __f) const& {
    using _Up = invoke_result_t<_Func, const _Tp&>;
    static_assert(__is_std_optional<remove_cvref_t<_Up>>::value,
                  "Result of f(value()) must be a specialization of std::optional");
    if (*this)
      return std::invoke(std::forward<_Func>(__f), value());
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto and_then(_Func&& __f) && {
    using _Up = invoke_result_t<_Func, _Tp&&>;
    static_assert(__is_std_optional<remove_cvref_t<_Up>>::value,
                  "Result of f(std::move(value())) must be a specialization of std::optional");
    if (*this)
      return std::invoke(std::forward<_Func>(__f), std::move(value()));
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto and_then(_Func&& __f) const&& {
    using _Up = invoke_result_t<_Func, const _Tp&&>;
    static_assert(__is_std_optional<remove_cvref_t<_Up>>::value,
                  "Result of f(std::move(value())) must be a specialization of std::optional");
    if (*this)
      return std::invoke(std::forward<_Func>(__f), std::move(value()));
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto transform(_Func&& __f) & {
    using _Up = remove_cv_t<invoke_result_t<_Func, _Tp&>>;
    static_assert(!is_array_v<_Up>, "Result of f(value()) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(value()) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(value()) should not be std::nullopt_t");
    static_assert(__is_valid_optional_type<_Up>, _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE);
    if (*this)
      return optional<_Up>(__optional_construct_from_invoke_tag{}, std::forward<_Func>(__f), value());
    return optional<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto transform(_Func&& __f) const& {
    using _Up = remove_cv_t<invoke_result_t<_Func, const _Tp&>>;
    static_assert(!is_array_v<_Up>, "Result of f(value()) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(value()) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(value()) should not be std::nullopt_t");
    static_assert(__is_valid_optional_type<_Up>, _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE);
    if (*this)
      return optional<_Up>(__optional_construct_from_invoke_tag{}, std::forward<_Func>(__f), value());
    return optional<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto transform(_Func&& __f) && {
    using _Up = remove_cv_t<invoke_result_t<_Func, _Tp&&>>;
    static_assert(!is_array_v<_Up>, "Result of f(std::move(value())) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(std::move(value())) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(std::move(value())) should not be std::nullopt_t");
    static_assert(__is_valid_optional_type<_Up>, _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE);
    if (*this)
      return optional<_Up>(__optional_construct_from_invoke_tag{}, std::forward<_Func>(__f), std::move(value()));
    return optional<_Up>();
  }

  template <class _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto transform(_Func&& __f) const&& {
    using _Up = remove_cv_t<invoke_result_t<_Func, const _Tp&&>>;
    static_assert(!is_array_v<_Up>, "Result of f(std::move(value())) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(std::move(value())) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(std::move(value())) should not be std::nullopt_t");
    static_assert(__is_valid_optional_type<_Up>, _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE);
    if (*this)
      return optional<_Up>(__optional_construct_from_invoke_tag{}, std::forward<_Func>(__f), std::move(value()));
    return optional<_Up>();
  }

  template <invocable _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr optional or_else(_Func&& __f) const&
    requires is_copy_constructible_v<_Tp>
  {
    static_assert(is_same_v<remove_cvref_t<invoke_result_t<_Func>>, optional>,
                  "Result of f() should be the same type as this optional");
    if (*this)
      return *this;
    return std::forward<_Func>(__f)();
  }

  template <invocable _Func>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr optional or_else(_Func&& __f) &&
    requires is_move_constructible_v<_Tp>
  {
    static_assert(is_same_v<remove_cvref_t<invoke_result_t<_Func>>, optional>,
                  "Result of f() should be the same type as this optional");
    if (*this)
      return std::move(*this);
    return std::forward<_Func>(__f)();
  }
#  endif // _LIBCPP_STD_VER >= 23

  using __base::reset;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif
