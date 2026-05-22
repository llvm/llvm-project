//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::optional<T&>

#ifndef _LIBCPP_OPTIONAL_OPTIONAL_REF_H
#define _LIBCPP_OPTIONAL_OPTIONAL_REF_H

#include <__concepts/invocable.h>
#include <__config>
#include <__functional/invoke.h>
#include <__fwd/optional.h>
#include <__optional/common.h>
#include <__optional/nullopt_t.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__type_traits/reference_constructs_from_temporary.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>

#include <__optional/comparison.h>
#include <__optional/hash.h>
#include <__optional/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
class _LIBCPP_DECLSPEC_EMPTY_BASES optional<_Tp&> : public __optional_iterator_base<_Tp&> {
  using __base _LIBCPP_NODEBUG = __optional_iterator_base<_Tp&>;

  template <class _Up, class _QualUp>
  static constexpr bool __check_optionalU_ctor =
      !std::is_same_v<std::remove_cv_t<_Tp>, optional<_Up>> && !std::is_same_v<_Tp&, _Up> &&
      std::is_constructible_v<_Tp&, _QualUp>;

  static_assert(!is_same_v<remove_cv_t<_Tp>, in_place_t>, "instantiation of optional with in_place_t is ill-formed");
  static_assert(!is_same_v<remove_cv_t<_Tp>, nullopt_t>, "instantiation of optional with nullopt_t is ill-formed");

public:
  using value_type = _Tp;

  constexpr optional() noexcept = default;
  constexpr optional(nullopt_t) noexcept {}
  constexpr optional(const optional&) noexcept = default;

  template <class _Arg>
    requires(std::is_constructible_v<_Tp&, _Arg> && !std::reference_constructs_from_temporary_v<_Tp&, _Arg>)
  constexpr explicit optional(in_place_t, _Arg&& __arg) : __base(in_place, std::forward<_Arg>(__arg)) {}

  template <class _Up>
    requires(!std::is_same_v<std::remove_cvref_t<_Up>, optional> && !is_same_v<std::remove_cvref_t<_Up>, in_place_t> &&
             is_constructible_v<_Tp&, _Up> && !std::reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>) optional(_Up&& __v) noexcept(is_nothrow_constructible_v<_Tp&, _Up>)
      : __base(in_place, std::forward<_Up>(__v)) {}

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up&> && !std::reference_constructs_from_temporary_v<_Tp&, _Up&>)
  constexpr explicit(!is_convertible_v<_Up&, _Tp&>)
      optional(optional<_Up>& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, _Up&>) {
    this->__construct_from(__rhs);
  }

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up&> && !std::reference_constructs_from_temporary_v<_Tp&, const _Up&>)
  constexpr explicit(!is_convertible_v<const _Up&, _Tp&>)
      optional(const optional<_Up>& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, const _Up&>) {
    this->__construct_from(__rhs);
  }

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up> && !std::reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>)
      optional(optional<_Up>&& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, _Up>) {
    this->__construct_from(std::move(__rhs));
  }

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up> && !std::reference_constructs_from_temporary_v<_Tp&, const _Up>)
  constexpr explicit(!is_convertible_v<const _Up, _Tp&>)
      optional(const optional<_Up>&& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, const _Up>) {
    this->__construct_from(std::move(__rhs));
  }

  template <class _Tag, class _Fp, class... _Args>
    requires(std::is_same_v<_Tag, __optional_construct_from_invoke_tag>)
  _LIBCPP_HIDE_FROM_ABI constexpr explicit optional(_Tag, _Fp&& __f, _Args&&... __args)
      : __base(__optional_construct_from_invoke_tag{}, std::forward<_Fp>(__f), std::forward<_Args>(__args)...) {}

  // deleted overloads

  template <class _Up>
    requires(!std::is_same_v<std::remove_cvref_t<_Up>, optional> && !is_same_v<std::remove_cvref_t<_Up>, in_place_t> &&
             is_constructible_v<_Tp&, _Up> && std::reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>)
      optional(_Up&& __v) noexcept(is_nothrow_constructible_v<_Tp&, _Up>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up&> && std::reference_constructs_from_temporary_v<_Tp&, _Up&>)
  constexpr explicit(!is_convertible_v<_Up&, _Tp&>)
      optional(optional<_Up>& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, _Up&>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up&> && std::reference_constructs_from_temporary_v<_Tp&, const _Up&>)
  constexpr explicit(!is_convertible_v<const _Up&, _Tp&>)
      optional(const optional<_Up>& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, const _Up&>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up> && std::reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>)
      optional(optional<_Up>&& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, _Up>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up> && std::reference_constructs_from_temporary_v<_Tp&, const _Up>)
  constexpr explicit(!is_convertible_v<const _Up, _Tp&>)
      optional(const optional<_Up>&& __rhs) noexcept(std::is_nothrow_constructible_v<_Tp&, const _Up>) = delete;

  constexpr ~optional() = default;

  using __base::__get;
  using __base::reset;

  _LIBCPP_HIDE_FROM_ABI constexpr optional& operator=(nullopt_t) noexcept {
    reset();
    return *this;
  }

  constexpr optional& operator=(const optional&) noexcept = default;

  template <class _Up>
    requires(is_constructible_v<_Tp&, _Up> && !std::reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr _Tp& emplace(_Up&& __u) noexcept(std::is_nothrow_constructible_v<_Tp&, _Up>) {
    this->__construct(std::forward<_Up>(__u));

    return this->__get();
  }

  constexpr void swap(optional& __rhs) noexcept { this->__swap(__rhs); }

  using __base::operator->;
  using __base::operator*;
  constexpr explicit operator bool() const noexcept { return has_value(); }
  using __base::has_value;
  using __base::value;

  template <class _Up = remove_cv_t<_Tp>>
    requires(is_object_v<_Tp> && !is_array_v<_Tp>)
  [[nodiscard]] constexpr decay_t<_Tp> value_or(_Up&& __v) const {
    using _XTp = remove_cv_t<_Tp>;
    static_assert(is_constructible_v<_XTp, _Tp&>, "optional<T&>::value_or: remove_cv_t<T> must be constructible");
    static_assert(is_convertible_v<_Up, _XTp>, "optional<T&>::value_or: U must be convertible to remove_cv_t<T>");
    return this->has_value() ? this->__get() : static_cast<_XTp>(std::forward<_Up>(__v));
  }

  template <class _Func>
  [[nodiscard]] constexpr auto and_then(_Func&& __f) const {
    using _Up = invoke_result_t<_Func, _Tp&>;
    static_assert(__is_std_optional<remove_cvref_t<_Up>>::value,
                  "Result of f(value()) must be a specialization of std::optional");
    if (*this)
      return std::invoke(std::forward<_Func>(__f), value());
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  [[nodiscard]] constexpr optional<remove_cv_t<invoke_result_t<_Func, _Tp&>>> transform(_Func&& __f) const {
    using _Up = remove_cv_t<invoke_result_t<_Func, _Tp&>>;
    static_assert(!is_array_v<_Up>, "Result of f(value()) should not be an Array");
    static_assert(!is_same_v<_Up, in_place_t>, "Result of f(value()) should not be std::in_place_t");
    static_assert(!is_same_v<_Up, nullopt_t>, "Result of f(value()) should not be std::nullopt_t");
    static_assert(__is_valid_optional_type<_Up>, _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE);

    if (*this)
      return optional<_Up>(__optional_construct_from_invoke_tag{}, std::forward<_Func>(__f), value());
    return optional<_Up>();
  }

  template <invocable _Func>
  [[nodiscard]] constexpr optional or_else(_Func&& __f) const {
    static_assert(is_same_v<remove_cvref_t<invoke_result_t<_Func>>, optional>,
                  "Result of f() should be the same type as this optional");
    if (*this)
      return *this;
    return std::forward<_Func>(__f)();
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP_OPTIONAL_OPTIONAL_REF_H
