//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_DISENGAGED_TRAITS_H
#define _LIBCPP___TYPE_TRAITS_DISENGAGED_TRAITS_H

#include <__config>
#include <__memory/construct_at.h>
#include <__type_traits/datasizeof.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_fundamental.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/void_t.h>
#include <__utility/forward_like.h>
#include <__utility/in_place.h>
#include <__utility/piecewise_construct.h>
#include <__utility/pointer_int_pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class>
struct __tombstone_memory_layout;

// bools have always exactly one bit set. If there is more than one set it's disengaged.
template <>
struct __tombstone_memory_layout<bool> {
  static constexpr uint8_t __disengaged_value_    = 3;
  static constexpr size_t __is_disengaged_offset_ = 0;
};

struct __tombstone_pointer_layout {
  static constexpr uint8_t __disengaged_value_ = 1;
#ifdef _LIBCPP_LITTLE_ENDIAN
  static constexpr size_t __is_disengaged_offset_ = 0;
#else
  static constexpr size_t __is_disengaged_offset_ = sizeof(void*) - 1;
#endif
};

// TODO: Look into
// - filesystem::directory_iterator
// - vector<T> with alignof(T) == 1

template <class _Tp>
struct __tombstone_memory_layout<__enable_specialization_if<is_fundamental_v<_Tp> && alignof(_Tp) >= 2, _Tp*>>
    : __tombstone_pointer_layout {};

template <class _Tp>
struct __tombstone_memory_layout<_Tp**> : __tombstone_pointer_layout {};

inline constexpr struct __init_engaged_t {
} __init_engaged;
inline constexpr struct __init_disengaged_t {
} __init_disengaged;

template <class _Tp, class _Payload>
struct __tombstone_data {
  using _TombstoneLayout = __tombstone_memory_layout<_Tp>;
  using _IsDisengagedT   = remove_cv_t<decltype(_TombstoneLayout::__disengaged_value_)>;

  _LIBCPP_NO_UNIQUE_ADDRESS _Payload __payload_;
  char __padding_[_TombstoneLayout::__is_disengaged_offset_ - __datasizeof_v<_Payload>];
  _IsDisengagedT __is_disengaged_;

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr __tombstone_data(_Args&&... __args)
      : __payload_(std::forward<_Args>(__args)...), __is_disengaged_(_TombstoneLayout::__disengaged_value_) {}
};

template <class _Tp, class _Payload>
  requires(__tombstone_memory_layout<_Tp>::__is_disengaged_offset_ == 0)
struct __tombstone_data<_Tp, _Payload> {
  using _TombstoneLayout = __tombstone_memory_layout<_Tp>;
  using _IsDisengagedT   = remove_cv_t<decltype(_TombstoneLayout::__disengaged_value_)>;

  _IsDisengagedT __is_disengaged_;
  _LIBCPP_NO_UNIQUE_ADDRESS _Payload __payload_;

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr __tombstone_data(_Args&&... __args)
      : __is_disengaged_(_TombstoneLayout::__disengaged_value_), __payload_(std::forward<_Args>(__args)...) {}
};

template <class _Tp, class _Payload>
struct __tombstone_traits {
  using _TombstoneLayout = __tombstone_memory_layout<_Tp>;
  using _TombstoneData   = __tombstone_data<_Tp, _Payload>;

  union {
    _Tp __value_;
    _TombstoneData __tombstone_;
  };

  static_assert(sizeof(__tombstone_data<_Tp, _Payload>) <= sizeof(_Tp));
  static_assert(is_integral_v<decltype(_TombstoneLayout::__disengaged_value_)>);
  static_assert(offsetof(_TombstoneData, __is_disengaged_) == _TombstoneLayout::__is_disengaged_offset_);

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr __tombstone_traits(__init_disengaged_t, _Args&&... __args)
      : __tombstone_(std::forward<_Args>(__args)...) {}

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr __tombstone_traits(__init_engaged_t, _Args&&... __args)
      : __value_(std::forward<_Args>(__args)...) {}

  template <class _Class>
  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_value(this _Class&& __self) noexcept {
    return std::forward<_Class>(__self).__value_;
  }

  template <class _Class>
  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_payload(this _Class&& __self) noexcept {
    return std::forward<_Class>(__self).__tombstone_.__payload_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool __is_engaged() const noexcept {
    if (__libcpp_is_constant_evaluated())
      return !__builtin_constant_p(__tombstone_.__is_disengaged_ == _TombstoneLayout::__disengaged_value_);
    return __tombstone_.__is_disengaged_ != _TombstoneLayout::__disengaged_value_;
  }

  template <class _Class, class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr void __engage(this _Class& __self, piecewise_construct_t, _Args&&... __args) {
    std::destroy_at(&__self.__tombstone_);
    std::construct_at(&__self.__value_, std::forward<_Args>(__args)...);
  }

  template <class _Class, class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr void __disengage(this _Class& __self, piecewise_construct_t, _Args&&... __args) {
    std::destroy_at(&__self.__data_.__value_);
    std::construct_at(&__self.__data_.__payload_, std::forward<_Args>(__args)...);
    __self.__data_.__is_disengaged_ = _TombstoneLayout::__disengaged_value_;
  }

  __tombstone_traits(const __tombstone_traits&)            = default;
  __tombstone_traits(__tombstone_traits&&)                 = default;
  __tombstone_traits& operator=(const __tombstone_traits&) = default;
  __tombstone_traits& operator=(__tombstone_traits&&)      = default;

  _LIBCPP_HIDE_FROM_ABI constexpr ~__tombstone_traits() {
    if (__is_engaged()) {
      std::destroy_at(&__value_);
    } else {
      std::destroy_at(&__tombstone_);
    }
  }

  ~__tombstone_traits()
    requires is_trivially_destructible_v<_Tp> && is_trivially_destructible_v<_Payload>
  = default;
};

template <class _Tp, class = void>
inline constexpr bool __has_tombstone_v = false;

template <class _Tp>
inline constexpr bool __has_tombstone_v<_Tp, void_t<decltype(sizeof(__tombstone_memory_layout<_Tp>))>> = true;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_DISENGAGED_TRAITS_H
