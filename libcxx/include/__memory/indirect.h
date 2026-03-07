// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_INDIRECT_H
#define _LIBCPP___MEMORY_INDIRECT_H

#include <__config>

#include <__assert>
#include <__compare/strong_order.h>
#include <__compare/synth_three_way.h>
#include <__functional/hash.h>
#include <__fwd/memory_resource.h>
#include <__memory/addressof.h>
#include <__memory/allocation_guard.h>
#include <__memory/allocator_arg_t.h>
#include <__memory/allocator_traits.h>
#include <__memory/pointer_traits.h>
#include <__memory/swap_allocator.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/exchange.h>
#include <__utility/forward.h>
#include <__utility/forward_like.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <__utility/swap.h>
#include <initializer_list>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Allocator = allocator<_Tp>>
class _LIBCPP_NO_SPECIALIZATIONS indirect {
public:
  using value_type     = _Tp;
  using allocator_type = _Allocator;
  using pointer        = allocator_traits<_Allocator>::pointer;
  using const_pointer  = allocator_traits<_Allocator>::const_pointer;

  static_assert(__check_valid_allocator<allocator_type>::value);
  static_assert(is_same_v<typename allocator_type::value_type, value_type>,
                "allocator's value_type type must match std::indirect's held type");
  static_assert(is_object_v<value_type>, "std::indirect cannot hold void or a reference or function type");
  static_assert(!is_array_v<value_type>, "std::indirect cannot hold an array type");
  static_assert(!is_same_v<value_type, in_place_t>, "std::indirect cannot hold std::in_place_t");
  static_assert(!__is_inplace_type<value_type>::value,
                "std::indirect cannot hold a specialization of std::in_place_type_t");
  static_assert(std::is_same_v<value_type, remove_cv_t<value_type>>,
                "std::indirect cannot hold a const or volatile qualified type");

  // [indirect.ctor], constructors
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect()
    requires is_default_constructible_v<_Allocator>
      : __ptr_(__allocate_owned_object(__alloc_)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(allocator_arg_t, const _Allocator& __a)
      : __alloc_(__a), __ptr_(__allocate_owned_object(__alloc_)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr indirect(const indirect& __other)
      : __alloc_(allocator_traits<_Allocator>::select_on_container_copy_construction(__other.__alloc_)),
        __ptr_(__other.valueless_after_move() ? nullptr : __allocate_owned_object(__alloc_, *__other)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr indirect(allocator_arg_t, const _Allocator& __a, const indirect& __other)
      : __alloc_(__a), __ptr_(__other.valueless_after_move() ? nullptr : __allocate_owned_object(__alloc_, *__other)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr indirect(indirect&& __other) noexcept
      : __alloc_(std::move(__other.__alloc_)), __ptr_(std::exchange(__other.__ptr_, nullptr)) {}

  _LIBCPP_HIDE_FROM_ABI constexpr indirect(allocator_arg_t, const _Allocator& __a, indirect&& __other) noexcept(
      allocator_traits<_Allocator>::is_always_equal::value)
      : __alloc_(__a) {
    if constexpr (allocator_traits<_Allocator>::is_always_equal::value) {
      __ptr_ = std::exchange(__other.__ptr_, nullptr);
    } else if (__other.valueless_after_move()) {
      __ptr_ = nullptr;
    } else if (__alloc_ == __other.__alloc_) {
      __ptr_ = std::exchange(__other.__ptr_, nullptr);
    } else {
      __ptr_ = __allocate_owned_object(__alloc_, *std::move(__other));
      __other.__destroy_owned_object();
      __other.__ptr_ = nullptr;
    }
  }

  template <class _Up = _Tp>
    requires(!is_same_v<remove_cvref_t<_Up>, indirect> && !is_same_v<remove_cvref_t<_Up>, in_place_t> &&
             is_constructible_v<_Tp, _Up> && is_default_constructible_v<_Allocator>)
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(_Up&& __u)
      : __ptr_(__allocate_owned_object(__alloc_, std::forward<_Up>(__u))) {}

  template <class _Up = _Tp>
    requires(!is_same_v<remove_cvref_t<_Up>, indirect> && !is_same_v<remove_cvref_t<_Up>, in_place_t> &&
             is_constructible_v<_Tp, _Up>)
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(allocator_arg_t, const _Allocator& __a, _Up&& __u)
      : __alloc_(__a), __ptr_(__allocate_owned_object(__alloc_, std::forward<_Up>(__u))) {}

  template <class... _Us>
    requires(is_constructible_v<_Tp, _Us...> && is_default_constructible_v<_Allocator>)
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(in_place_t, _Us&&... __us)
      : __ptr_(__allocate_owned_object(__alloc_, std::forward<_Us>(__us)...)) {}

  template <class... _Us>
    requires is_constructible_v<_Tp, _Us...>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(allocator_arg_t, const _Allocator& __a, in_place_t, _Us&&... __us)
      : __alloc_(__a), __ptr_(__allocate_owned_object(__alloc_, std::forward<_Us>(__us)...)) {}

  template <class _In, class... _Us>
    requires(is_constructible_v<_Tp, initializer_list<_In>&, _Us...> && is_default_constructible_v<_Allocator>)
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(in_place_t, initializer_list<_In> __ilist, _Us&&... __us)
      : __ptr_(__allocate_owned_object(__alloc_, __ilist, std::forward<_Us>(__us)...)) {}

  template <class _In, class... _Us>
    requires is_constructible_v<_Tp, initializer_list<_In>&, _Us...>
  _LIBCPP_HIDE_FROM_ABI constexpr explicit indirect(
      allocator_arg_t, const _Allocator& __a, in_place_t, initializer_list<_In> __ilist, _Us&&... __us)
      : __alloc_(__a), __ptr_(__allocate_owned_object(__alloc_, __ilist, std::forward<_Us>(__us)...)) {}

  // [indirect.dtor], destructor
  _LIBCPP_HIDE_FROM_ABI constexpr ~indirect() { __destroy_owned_object(); }

  // [indirect.assign], assignment
  _LIBCPP_HIDE_FROM_ABI constexpr indirect& operator=(const indirect& __other) {
    if (std::addressof(__other) == this)
      return *this;

    static constexpr bool __propagate_allocator =
        allocator_traits<_Allocator>::propagate_on_container_copy_assignment::value;
    if (__other.valueless_after_move()) {
      __destroy_owned_object();
      __ptr_ = nullptr;
    } else if (!valueless_after_move() && __alloc_ == __other.__alloc_) {
      *__ptr_ = *__other;
    } else {
      pointer __new_ptr;
      if constexpr (__propagate_allocator) {
        // We need a mutable instance of the allocator, so make a copy.
        _Allocator __alloc_copy = __other.__alloc_;
        __new_ptr               = __allocate_owned_object(__alloc_copy, *__other);
      } else {
        __new_ptr = __allocate_owned_object(__alloc_, *__other);
      }
      __destroy_owned_object();
      __ptr_ = __new_ptr;
    }

    if constexpr (__propagate_allocator)
      __alloc_ = __other.__alloc_;

    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr indirect& operator=(indirect&& __other) noexcept(
      allocator_traits<_Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<_Allocator>::is_always_equal::value) {
    if (std::addressof(__other) == this)
      return *this;

    static constexpr bool __propagate_allocator =
        allocator_traits<_Allocator>::propagate_on_container_move_assignment::value;

    pointer __new_ptr;
    if constexpr (__propagate_allocator || allocator_traits<_Allocator>::is_always_equal::value) {
      __new_ptr = __other.__ptr_;
    } else if (__other.valueless_after_move()) {
      __new_ptr = nullptr;
    } else if (__alloc_ == __other.__alloc_) {
      __new_ptr = __other.__ptr_;
    } else {
      __new_ptr = __allocate_owned_object(__alloc_, *std::move(__other));
      __other.__destroy_owned_object();
    }
    __other.__ptr_ = nullptr;
    __destroy_owned_object();
    __ptr_ = __new_ptr;

    if constexpr (__propagate_allocator)
      __alloc_ = __other.__alloc_;

    return *this;
  }

  template <class _Up = _Tp>
    requires(!is_same_v<remove_cvref_t<_Up>, indirect> && is_constructible_v<_Tp, _Up> && is_assignable_v<_Tp&, _Up>)
  _LIBCPP_HIDE_FROM_ABI constexpr indirect& operator=(_Up&& __u) {
    if (valueless_after_move())
      __ptr_ = __allocate_owned_object(__alloc_, std::forward<_Up>(__u));
    else
      *__ptr_ = std::forward<_Up>(__u);
    return *this;
  }

  // [indirect.obs], observers
  template <class _Self>
    requires(!is_volatile_v<_Self>)
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto&& operator*(this _Self&& __self) noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS((!std::__forward_as<_Self, indirect>(__self).valueless_after_move()),
                                        "operator* called on a valueless std::indirect object");
    return std::forward_like<_Self>(*__self.__ptr_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const_pointer operator->() const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !valueless_after_move(), "operator-> called on a valueless std::indirect object");
    return __ptr_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr pointer operator->() noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        !valueless_after_move(), "operator-> called on a valueless std::indirect object");
    return __ptr_;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr bool valueless_after_move() const noexcept { return !__ptr_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr allocator_type get_allocator() const noexcept { return __alloc_; }

  // [indirect.swap], swap
  _LIBCPP_HIDE_FROM_ABI constexpr void
  swap(indirect& __other) noexcept(allocator_traits<_Allocator>::propagate_on_container_swap::value ||
                                   allocator_traits<_Allocator>::is_always_equal::value) {
    _LIBCPP_ASSERT_COMPATIBLE_ALLOCATOR(
        allocator_traits<_Allocator>::propagate_on_container_swap::value || get_allocator() == __other.get_allocator(),
        "swapping std::indirect objects with different allocators");
    std::swap(__ptr_, __other.__ptr_);
    std::__swap_allocator(__alloc_, __other.__alloc_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  swap(indirect& __lhs, indirect& __rhs) noexcept(noexcept(__lhs.swap(__rhs))) {
    __lhs.swap(__rhs);
  }

  // [indirect.relops], relational operators
  template <class _Up, class _AA>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const indirect& __lhs, const indirect<_Up, _AA>& __rhs) noexcept(noexcept(*__lhs == *__rhs)) {
    return (__lhs.valueless_after_move() == __rhs.valueless_after_move()) &&
           (__lhs.valueless_after_move() || *__lhs == *__rhs);
  }

  template <class _Up, class _AA>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __synth_three_way_result<_Tp, _Up>
  operator<=>(const indirect& __lhs, const indirect<_Up, _AA>& __rhs) {
    if (__lhs.valueless_after_move() || __rhs.valueless_after_move())
      return !__lhs.valueless_after_move() <=> !__rhs.valueless_after_move();
    return std::__synth_three_way(*__lhs, *__rhs);
  }

  // [indirect.comp.with.t], comparison with T
  template <class _Up>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const indirect& __lhs, const _Up& __rhs) noexcept(noexcept(*__lhs == __rhs)) {
    return !__lhs.valueless_after_move() && *__lhs == __rhs;
  }

  template <class _Up>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __synth_three_way_result<_Tp, _Up>
  operator<=>(const indirect& __lhs, const _Up& __rhs) {
    return __lhs.valueless_after_move() ? strong_ordering::less : std::__synth_three_way(*__lhs, __rhs);
  }

private:
  template <class... _Us>
  _LIBCPP_HIDE_FROM_ABI static constexpr pointer __allocate_owned_object(_Allocator& __a, _Us&&... __us) {
    __allocation_guard<_Allocator> __guard(__a, 1);
    allocator_traits<_Allocator>::construct(__a, std::to_address(__guard.__get()), std::forward<_Us>(__us)...);
    return __guard.__release_ptr();
  }

  _LIBCPP_HIDE_FROM_ABI constexpr void __destroy_owned_object() noexcept {
    if (!valueless_after_move()) {
      allocator_traits<_Allocator>::destroy(__alloc_, std::to_address(__ptr_));
      allocator_traits<_Allocator>::deallocate(__alloc_, __ptr_, 1);
    }
  }

  _LIBCPP_NO_UNIQUE_ADDRESS _Allocator __alloc_ = _Allocator();
  pointer __ptr_;
};

template <class _Value>
indirect(_Value) -> indirect<_Value>;

template <class _Allocator, class _Value>
indirect(allocator_arg_t, _Allocator, _Value) -> indirect<_Value, __rebind_alloc<allocator_traits<_Allocator>, _Value>>;

template <class _Tp, class _Allocator>
  requires is_default_constructible_v<hash<_Tp>>
struct hash<indirect<_Tp, _Allocator>> {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t operator()(const indirect<_Tp, _Allocator>& __i) const {
    return __i.valueless_after_move() ? 0 : hash<_Tp>()(*__i);
  }
};

namespace pmr {

template <class _Tp>
using indirect _LIBCPP_AVAILABILITY_PMR = indirect<_Tp, polymorphic_allocator<_Tp>>;

} // namespace pmr

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_INDIRECT_H
