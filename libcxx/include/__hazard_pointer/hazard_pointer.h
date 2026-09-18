// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HAZARD_POINTER_HAZARD_POINTER_H
#define _LIBCPP___HAZARD_POINTER_HAZARD_POINTER_H

#include <__assert>
#include <__atomic/atomic.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__cstddef/nullptr_t.h>
#include <__hazard_pointer/domain.h>
#include <__hazard_pointer/hazard_pointer_obj_base.h>
#include <__memory/addressof.h>
#include <__utility/exchange.h>
#include <__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_AVAILABILITY_HAZARD_POINTER hazard_pointer {
  __hazard_pointer_slot* __slot_ = nullptr; // empty <=> nullptr

  friend _LIBCPP_HIDE_FROM_ABI hazard_pointer make_hazard_pointer();

  _LIBCPP_HIDE_FROM_ABI explicit hazard_pointer(__hazard_pointer_slot* __slot) noexcept : __slot_(__slot) {}

  // The address a hazard pointer holds for *__ptr: that of its (private) node base. Deduces D through the
  // derived-to-base conversion; hazard_pointer is a friend of hazard_pointer_obj_base, so the conversion to
  // the private base is accessible here. A null T* yields a null node.
  template <class _Tp, class _Dp>
  _LIBCPP_HIDE_FROM_ABI static const __hazard_pointer_obj_node*
  __node_of(const hazard_pointer_obj_base<_Tp, _Dp>* __ptr) noexcept {
    return __ptr;
  }

public:
  _LIBCPP_HIDE_FROM_ABI hazard_pointer() noexcept = default;

  _LIBCPP_HIDE_FROM_ABI hazard_pointer(hazard_pointer&& __other) noexcept
      : __slot_(std::exchange(__other.__slot_, nullptr)) {}

  _LIBCPP_HIDE_FROM_ABI hazard_pointer& operator=(hazard_pointer&& __other) noexcept {
    if (this != std::addressof(__other)) {
      if (__slot_ != nullptr)
        std::__hazard_pointer_release(__slot_);
      __slot_ = std::exchange(__other.__slot_, nullptr);
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI ~hazard_pointer() {
    if (__slot_ != nullptr)
      std::__hazard_pointer_release(__slot_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool empty() const noexcept { return __slot_ == nullptr; }

  template <class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI _Tp* protect(const atomic<_Tp*>& __src) noexcept {
    _Tp* __ptr = __src.load(memory_order_relaxed);
    while (!try_protect(__ptr, __src)) {
    }
    return __ptr;
  }

  template <class _Tp>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool try_protect(_Tp*& __ptr, const atomic<_Tp*>& __src) noexcept {
    static_assert(
        __hazard_protectable<_Tp>,
        "hazard_pointer::try_protect(): T must be a hazard-protectable type (a class with exactly one public, "
        "non-virtual hazard_pointer_obj_base<T, D> base)");
    _Tp* __old = __ptr;
    reset_protection(__old);
    std::__hazard_pointer_reader_fence();
    __ptr = __src.load(memory_order_acquire);
    if (__old != __ptr) {
      reset_protection();
      return false;
    }
    return true;
  }

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI void reset_protection(const _Tp* __ptr) noexcept {
    static_assert(__hazard_protectable<_Tp>,
                  "hazard_pointer::reset_protection(): T must be a hazard-protectable type (a class with exactly one "
                  "public, non-virtual hazard_pointer_obj_base<T, D> base)");
    _LIBCPP_ASSERT_NON_NULL(__slot_ != nullptr, "hazard_pointer::reset_protection(): hazard_pointer is empty");
    __slot_->__value_.store(__node_of(__ptr), memory_order_release);
  }

  _LIBCPP_HIDE_FROM_ABI void reset_protection(nullptr_t = nullptr) noexcept {
    _LIBCPP_ASSERT_NON_NULL(__slot_ != nullptr, "hazard_pointer::reset_protection(): hazard_pointer is empty");
    __slot_->__value_.store(nullptr, memory_order_release);
  }

  _LIBCPP_HIDE_FROM_ABI void swap(hazard_pointer& __other) noexcept { std::swap(__slot_, __other.__slot_); }
};

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_AVAILABILITY_HAZARD_POINTER inline hazard_pointer make_hazard_pointer() {
  return hazard_pointer(std::__hazard_pointer_acquire());
}

// The availability attribute is required, not just informative: where it expands to
// __attribute__((unavailable)), naming hazard_pointer in a declaration that is not itself unavailable is an
// error even inside a system header.
_LIBCPP_HIDE_FROM_ABI _LIBCPP_AVAILABILITY_HAZARD_POINTER inline void
swap(hazard_pointer& __a, hazard_pointer& __b) noexcept {
  __a.swap(__b);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS

_LIBCPP_POP_MACROS

#endif // _LIBCPP___HAZARD_POINTER_HAZARD_POINTER_H
