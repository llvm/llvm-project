//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_TOMBSTONE_TRAITS_H
#define _LIBCPP___TYPE_TRAITS_TOMBSTONE_TRAITS_H

#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__memory/construct_at.h>
#include <__type_traits/datasizeof.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_fundamental.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_trivial.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/void_t.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class>
struct __tombstone_traits;

#if _LIBCPP_STD_VER >= 17

// bools always have exactly one bit set. If there is more than one set it's disengaged.
template <>
struct __tombstone_traits<bool> {
  static constexpr uint8_t __disengaged_value_    = 3;
  static constexpr size_t __is_disengaged_offset_ = 0;
};

// Pointers to a type that has an alignment greater than one always have the lowest bits set to zero. This is a single
// implementation for all the places where we have an invalid pointer to _Tp as the "invalid state" representation.
template <class _Tp>
struct __tombstone_traits_assume_aligned_pointer {
  static constexpr uint8_t __disengaged_value_ = 1;
#  ifdef _LIBCPP_LITTLE_ENDIAN
  static constexpr size_t __is_disengaged_offset_ = 0;
#  else
  static constexpr size_t __is_disengaged_offset_ = sizeof(_Tp*) - 1;
#  endif

  static_assert(alignof(_Tp) >= 2);
};

// TODO: Look into
// - filesystem::directory_iterator
// - vector<T> with alignof(T) == 1
// - string_view (basic_string_view<T> works with alignof(T) >= 2)

// This is constrained on fundamental types because we might not always know the alignment of a user-defined type.
// For example, in one TU there may only be a forward declaration and in another there is already the definition
// available. If we made this optimization conditional on the completeness of the type this would result in a non-benign
// ODR violation.
template <class _Tp>
struct __tombstone_traits<__enable_specialization_if<is_fundamental_v<_Tp> && alignof(_Tp) >= 2, _Tp*>>
    : __tombstone_traits_assume_aligned_pointer<_Tp> {};

template <class _Tp>
struct __tombstone_traits<_Tp**> : __tombstone_traits_assume_aligned_pointer<_Tp*> {
  static_assert(alignof(_Tp*) >= 2, "alignment of a pointer isn't at least 2!?");
};

inline constexpr struct __init_engaged_t {
} __init_engaged;
inline constexpr struct __init_disengaged_t {
} __init_disengaged;

template <class _Tp, class _Payload, bool = __tombstone_traits<_Tp>::__is_disengaged_offset_ == 0>
struct __tombstone_is_disengaged {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _IsDisengagedT   = remove_cv_t<decltype(_TombstoneLayout::__disengaged_value_)>;

  char __padding_[_TombstoneLayout::__is_disengaged_offset_];
  _IsDisengagedT __is_disengaged_;
};

template <class _Tp, class _Payload>
struct __tombstone_is_disengaged<_Tp, _Payload, true> {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _IsDisengagedT   = remove_cv_t<decltype(_TombstoneLayout::__disengaged_value_)>;

  _IsDisengagedT __is_disengaged_;
};

template <class _Tp, class _Payload, bool = __tombstone_traits<_Tp>::__is_disengaged_offset_ == 0>
struct __tombstone_data {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _IsDisengagedT   = remove_cv_t<decltype(_TombstoneLayout::__disengaged_value_)>;

  static_assert(is_trivial<_IsDisengagedT>::value, "disengaged type has to be trivial!");
  static_assert(_TombstoneLayout::__is_disengaged_offset_ >= __datasizeof_v<_Payload>);

  _LIBCPP_NO_UNIQUE_ADDRESS _Payload __payload_;
  char __padding_[_TombstoneLayout::__is_disengaged_offset_ - __datasizeof_v<_Payload>];
  _IsDisengagedT __is_disengaged_;

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr __tombstone_data(_Args&&... __args)
      : __payload_(std::forward<_Args>(__args)...), __is_disengaged_(_TombstoneLayout::__disengaged_value_) {}
};

template <class _Tp, class _Payload>
struct __tombstone_data<_Tp, _Payload, true> {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _IsDisengagedT   = remove_cv_t<decltype(_TombstoneLayout::__disengaged_value_)>;

  _IsDisengagedT __is_disengaged_;
  _LIBCPP_NO_UNIQUE_ADDRESS _Payload __payload_;

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr __tombstone_data(_Args&&... __args)
      : __is_disengaged_(_TombstoneLayout::__disengaged_value_), __payload_(std::forward<_Args>(__args)...) {}
};

template <class _Tp, class _Payload, bool = is_trivially_destructible_v<_Tp> && is_trivially_destructible_v<_Payload>>
union _MaybeTombstone {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _TombstoneData   = __tombstone_data<_Tp, _Payload>;

  _Tp __value_;
  _TombstoneData __tombstone_;

  template <class... _Args>
  explicit constexpr _MaybeTombstone(__init_disengaged_t, _Args&&... __args)
      : __tombstone_(std::forward<_Args>(__args)...) {}

  template <class... _Args>
  explicit constexpr _MaybeTombstone(__init_engaged_t, _Args&&... __args) : __value_(std::forward<_Args>(__args)...) {}

  _MaybeTombstone(const _MaybeTombstone&)            = default;
  _MaybeTombstone(_MaybeTombstone&&)                 = default;
  _MaybeTombstone& operator=(const _MaybeTombstone&) = default;
  _MaybeTombstone& operator=(_MaybeTombstone&&)      = default;

  _LIBCPP_HIDE_FROM_ABI constexpr bool __is_engaged() const noexcept {
    if (__libcpp_is_constant_evaluated())
      return !__builtin_constant_p(__tombstone_.__is_disengaged_ == _TombstoneLayout::__disengaged_value_);
    __tombstone_is_disengaged<_Tp, _Payload> __is_disengaged;
    static_assert(sizeof(__tombstone_is_disengaged<_Tp, _Payload>) <= sizeof(_MaybeTombstone));
    __builtin_memcpy(&__is_disengaged, this, sizeof(__tombstone_is_disengaged<_Tp, _Payload>));

    return __is_disengaged.__is_disengaged_ != _TombstoneLayout::__disengaged_value_;
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 ~_MaybeTombstone() {
    if (__is_engaged()) {
      std::destroy_at(&__value_);
    } else {
      std::destroy_at(&__tombstone_);
    }
  }
};

template <class _Tp, class _Payload>
union _MaybeTombstone<_Tp, _Payload, true> {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _TombstoneData   = __tombstone_data<_Tp, _Payload>;

  _Tp __value_;
  _TombstoneData __tombstone_;

  template <class... _Args>
  explicit constexpr _MaybeTombstone(__init_disengaged_t, _Args&&... __args)
      : __tombstone_(std::forward<_Args>(__args)...) {}

  template <class... _Args>
  explicit constexpr _MaybeTombstone(__init_engaged_t, _Args&&... __args) : __value_(std::forward<_Args>(__args)...) {}

  _MaybeTombstone(const _MaybeTombstone&)            = default;
  _MaybeTombstone(_MaybeTombstone&&)                 = default;
  _MaybeTombstone& operator=(const _MaybeTombstone&) = default;
  _MaybeTombstone& operator=(_MaybeTombstone&&)      = default;

  _LIBCPP_HIDE_FROM_ABI constexpr bool __is_engaged() const noexcept {
    if (__libcpp_is_constant_evaluated())
      return !__builtin_constant_p(__tombstone_.__is_disengaged_ == _TombstoneLayout::__disengaged_value_);
    __tombstone_is_disengaged<_Tp, _Payload> __is_disengaged;
    static_assert(sizeof(__tombstone_is_disengaged<_Tp, _Payload>) <= sizeof(_MaybeTombstone));
    __builtin_memcpy(&__is_disengaged, this, sizeof(__tombstone_is_disengaged<_Tp, _Payload>));

    return __is_disengaged.__is_disengaged_ != _TombstoneLayout::__disengaged_value_;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 ~_MaybeTombstone() = default;
};

template <class _Tp, class _Payload>
struct __tombstoned_value final {
  using _TombstoneLayout = __tombstone_traits<_Tp>;
  using _TombstoneData   = __tombstone_data<_Tp, _Payload>;

  _MaybeTombstone<_Tp, _Payload> __data_;

  static_assert(sizeof(__tombstone_data<_Tp, _Payload>) <= sizeof(_Tp));
  static_assert(is_integral_v<decltype(_TombstoneLayout::__disengaged_value_)>);
  static_assert(__builtin_offsetof(_TombstoneData, __is_disengaged_) == _TombstoneLayout::__is_disengaged_offset_);

  template <class... _Args>
  explicit _LIBCPP_HIDE_FROM_ABI constexpr __tombstoned_value(__init_disengaged_t, _Args&&... __args)
      : __data_(__init_disengaged, std::forward<_Args>(__args)...) {}

  template <class... _Args>
  explicit _LIBCPP_HIDE_FROM_ABI constexpr __tombstoned_value(__init_engaged_t, _Args&&... __args)
      : __data_(__init_engaged, std::forward<_Args>(__args)...) {}

  _LIBCPP_HIDE_FROM_ABI constexpr bool __is_engaged() const noexcept { return __data_.__is_engaged(); }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_value() & noexcept {
    _LIBCPP_ASSERT_INTERNAL(__is_engaged(), "Trying to get the value of a disenagaged tombstoned value");
    return __data_.__value_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_value() const& noexcept {
    _LIBCPP_ASSERT_INTERNAL(__is_engaged(), "Trying to get the value of a disenagaged tombstoned value");
    return __data_.__value_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_value() && noexcept {
    _LIBCPP_ASSERT_INTERNAL(__is_engaged(), "Trying to get the value of a disenagaged tombstoned value");
    return std::move(__data_.__value_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_value() const&& noexcept {
    _LIBCPP_ASSERT_INTERNAL(__is_engaged(), "Trying to get the value of a disenagaged tombstoned value");
    return std::move(__data_.__value_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_payload() & noexcept {
    _LIBCPP_ASSERT_INTERNAL(!__is_engaged(), "Trying to get the payload of an enagaged tombstoned value");
    return __data_.__tombstone_.__payload_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_payload() const& noexcept {
    _LIBCPP_ASSERT_INTERNAL(!__is_engaged(), "Trying to get the payload of an enagaged tombstoned value");
    return __data_.__tombstone_.__payload_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_payload() && noexcept {
    _LIBCPP_ASSERT_INTERNAL(!__is_engaged(), "Trying to get the payload of an enagaged tombstoned value");
    return std::move(__data_.__tombstone_.__payload_);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto&& __get_payload() const&& noexcept {
    _LIBCPP_ASSERT_INTERNAL(!__is_engaged(), "Trying to get the payload of an enagaged tombstoned value");
    return std::move(__data_.__tombstone_.__payload_);
  }

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr void __engage(in_place_t, _Args&&... __args) {
    _LIBCPP_ASSERT_INTERNAL(!__is_engaged(), "Trying to enage a already engaged tombstoned value");
    std::destroy_at(&__data_.__tombstone_);
    std::__construct_at(&__data_.__value_, std::forward<_Args>(__args)...);
  }

  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI constexpr void __disengage(in_place_t, _Args&&... __args) {
    _LIBCPP_ASSERT_INTERNAL(!__is_engaged(), "Trying to disenage a disengaged tombstoned value");
    std::destroy_at(&__data_.__value_);
    std::__construct_at(&__data_.__tombstone_, std::forward<_Args>(__args)...);
  }
};

template <class _Tp, class = void>
inline constexpr bool __has_tombstone_v = false;

template <class _Tp>
inline constexpr bool __has_tombstone_v<_Tp, void_t<decltype(sizeof(__tombstone_traits<_Tp>))>> = true;

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___TYPE_TRAITS_TOMBSTONE_TRAITS_H
