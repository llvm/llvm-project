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

#include <__assert>
#include <__concepts/invocable.h>
#include <__config>
#include <__functional/invoke.h>
#include <__iterator/bounded_iter.h>
#include <__iterator/capacity_aware_iterator.h>
#include <__memory/addressof.h>
#include <__type_traits/add_pointer.h>
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
#include <__type_traits/remove_reference.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <__utility/swap.h>

#include <__fwd/optional.h>
#include <__optional/comparison.h>
#include <__optional/swap.h>

#include <__optional/common.h>
#include <__optional/hash.h>
#include <__optional/nullopt_t.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __optional_ref_base {
  using value_type                 = _Tp;
  using __raw_type _LIBCPP_NODEBUG = remove_reference_t<_Tp>;
  __raw_type* __value_;

  _LIBCPP_HIDE_FROM_ABI constexpr __optional_ref_base() noexcept : __value_(nullptr) {}

  template <class _Up>
  _LIBCPP_HIDE_FROM_ABI constexpr void __convert_init_ref_val(_Up&& __val) {
    _Tp& __r(std::forward<_Up>(__val));
    __value_ = std::addressof(__r);
  }

  template <class _UArg>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit __optional_ref_base(in_place_t, _UArg&& __uarg) {
    static_assert(!__reference_constructs_from_temporary_v<_Tp, _UArg>,
                  "Attempted to construct a reference element in optional from a "
                  "possible temporary");
    __convert_init_ref_val(std::forward<_UArg>(__uarg));
  }

  template <class _Fp, class... _Args>
  constexpr __optional_ref_base(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args) {
    __convert_init_ref_val(std::forward<invoke_result_t<_Fp, _Args...>>(
        std::invoke(std::forward<_Fp>(__f), std::forward<_Args>(__args)...)));
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void reset() noexcept { __value_ = nullptr; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr bool has_value() const noexcept { return __value_ != nullptr; }

  _LIBCPP_HIDE_FROM_ABI constexpr value_type& __get() const noexcept { return *__value_; }

  template <class _UArg>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __construct(_UArg&& __val) {
    static_assert(!__reference_constructs_from_temporary_v<_Tp, _UArg>,
                  "Attempted to construct a reference element in tuple from a "
                  "possible temporary");
    __convert_init_ref_val(std::forward<_UArg>(__val));
  }

  template <class _That>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __construct_from(_That&& __opt) {
    if (__opt.has_value())
      __construct(std::forward<_That>(__opt).__get());
  }

  template <class _That>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __assign_from(_That&& __opt) {
    if (has_value() == __opt.has_value()) {
      if (has_value())
        *__value_ = std::forward<_That>(__opt).__get();
    } else {
      if (has_value())
        reset();
      else
        __construct(std::forward<_That>(__opt).__get());
    }
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __swap(__optional_ref_base& __rhs) noexcept {
    std::swap(__value_, __rhs.__value_);
  }

  // [optional.ref.observe]
  _LIBCPP_HIDE_FROM_ABI constexpr add_pointer_t<_Tp> operator->() const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator-> called on a disengaged value");
    return std::addressof(this->__get());
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp& operator*() const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp& value() const {
    if (!this->has_value())
      std::__throw_bad_optional_access();
    return this->__get();
  }
};

template <class _Tp>
struct __optional_ref_iterator_base;

template <class _Tp>
struct __optional_ref_iterator_base<_Tp&> : __optional_ref_base<_Tp&> {
  using __optional_ref_base<_Tp&>::__optional_ref_base;
};

#  if _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR

template <class _Tp>
  requires(is_object_v<_Tp> && !__is_unbounded_array_v<_Tp>)
struct __optional_ref_iterator_base<_Tp&> : __optional_ref_base<_Tp&> {
private:
  using __pointer _LIBCPP_NODEBUG = add_pointer_t<_Tp>;

public:
  using __optional_ref_base<_Tp&>::__optional_ref_base;

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
  using iterator = __bounded_iter<__pointer>;
#    else
  using iterator = __capacity_aware_iterator<__pointer, optional<_Tp&>, 1>;
#    endif

  // [optional.ref.iterators], iterator support

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const noexcept {
    auto* __ptr = this->has_value() ? std::addressof(this->__get()) : nullptr;

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
    return std::__make_bounded_iter(__ptr, __ptr, __ptr + (this->has_value() ? 1 : 0));
#    else
    return std::__make_capacity_aware_iterator<__pointer, optional<_Tp&>, 1>(__ptr);
#    endif
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end() const noexcept {
    return begin() + (this->has_value() ? 1 : 0);
  }
};

#  endif

template <class _Tp>
class optional<_Tp&> : public __optional_ref_iterator_base<_Tp&> {
  using __base _LIBCPP_NODEBUG = __optional_ref_iterator_base<_Tp&>;

  template <class _Up, class _QualUp>
  static constexpr bool __check_optionalU_ctor =
      !is_same_v<remove_cv_t<_Tp>, optional<_Up>> && !is_same_v<_Tp&, _Up> && is_constructible_v<_Tp&, _QualUp>;
  static_assert(!is_same_v<remove_cv_t<_Tp>, in_place_t>, "instantiation of optional with in_place_t is ill-formed");
  static_assert(!is_same_v<remove_cv_t<_Tp>, nullopt_t>, "instantiation of optional with nullopt_t is ill-formed");

public:
  using value_type = _Tp;

  constexpr optional() noexcept = default;
  constexpr optional(nullopt_t) noexcept {}
  constexpr optional(const optional&) noexcept = default;

  template <class _Arg>
    requires(is_constructible_v<_Tp&, _Arg> && !reference_constructs_from_temporary_v<_Tp&, _Arg>)
  constexpr explicit optional(in_place_t, _Arg&& __arg) : __base(in_place, std::forward<_Arg>(__arg)) {}

  template <class _Up>
    requires(!is_same_v<remove_cvref_t<_Up>, optional> && !is_same_v<remove_cvref_t<_Up>, in_place_t> &&
             is_constructible_v<_Tp&, _Up> && !reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>) optional(_Up&& __v) noexcept(is_nothrow_constructible_v<_Tp&, _Up>)
      : __base(in_place, std::forward<_Up>(__v)) {}

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up&> && !reference_constructs_from_temporary_v<_Tp&, _Up&>)
  constexpr explicit(!is_convertible_v<_Up&, _Tp&>)
      optional(optional<_Up>& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, _Up&>) {
    this->__construct_from(__rhs);
  }

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up&> && !reference_constructs_from_temporary_v<_Tp&, const _Up&>)
  constexpr explicit(!is_convertible_v<const _Up&, _Tp&>)
      optional(const optional<_Up>& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, const _Up&>) {
    this->__construct_from(__rhs);
  }

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up> && !reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>)
      optional(optional<_Up>&& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, _Up>) {
    this->__construct_from(std::move(__rhs));
  }

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up> && !reference_constructs_from_temporary_v<_Tp&, const _Up>)
  constexpr explicit(!is_convertible_v<const _Up, _Tp&>)
      optional(const optional<_Up>&& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, const _Up>) {
    this->__construct_from(std::move(__rhs));
  }

  template <class _Tag, class _Fp, class... _Args>
    requires(is_same_v<_Tag, __optional_construct_from_invoke_tag>)
  _LIBCPP_HIDE_FROM_ABI constexpr explicit optional(_Tag, _Fp&& __f, _Args&&... __args)
      : __base(__optional_construct_from_invoke_tag{}, std::forward<_Fp>(__f), std::forward<_Args>(__args)...) {}

  // deleted overloads

  template <class _Up>
    requires(!is_same_v<remove_cvref_t<_Up>, optional> && !is_same_v<remove_cvref_t<_Up>, in_place_t> &&
             is_constructible_v<_Tp&, _Up> && reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>)
      optional(_Up&& __v) noexcept(is_nothrow_constructible_v<_Tp&, _Up>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up&> && reference_constructs_from_temporary_v<_Tp&, _Up&>)
  constexpr explicit(!is_convertible_v<_Up&, _Tp&>)
      optional(optional<_Up>& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, _Up&>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up&> && reference_constructs_from_temporary_v<_Tp&, const _Up&>)
  constexpr explicit(!is_convertible_v<const _Up&, _Tp&>)
      optional(const optional<_Up>& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, const _Up&>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, _Up> && reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr explicit(!is_convertible_v<_Up, _Tp&>)
      optional(optional<_Up>&& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, _Up>) = delete;

  template <class _Up>
    requires(__check_optionalU_ctor<_Up, const _Up> && reference_constructs_from_temporary_v<_Tp&, const _Up>)
  constexpr explicit(!is_convertible_v<const _Up, _Tp&>)
      optional(const optional<_Up>&& __rhs) noexcept(is_nothrow_constructible_v<_Tp&, const _Up>) = delete;

  constexpr ~optional() = default;

  using __base::__get;

  _LIBCPP_HIDE_FROM_ABI constexpr optional& operator=(nullopt_t) noexcept {
    reset();
    return *this;
  }

  constexpr optional& operator=(const optional&) noexcept = default;

  template <class _Up>
    requires(is_constructible_v<_Tp&, _Up> && !reference_constructs_from_temporary_v<_Tp&, _Up>)
  constexpr _Tp& emplace(_Up&& __u) noexcept(is_nothrow_constructible_v<_Tp&, _Up>) {
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
    requires(!is_array_v<_Tp> && is_object_v<_Tp>)
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
    static_assert(
        __is_valid_optional_contained_type<_Up>, "Result of f(value()) should be a valid contained type for optional");

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

  using __base::reset;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP_OPTIONAL_OPTIONAL_REF_H
