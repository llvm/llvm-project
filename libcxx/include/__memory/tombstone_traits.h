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
  template <class _Payload, size_t _Np>
  struct __representation {
    template <class... _Args>
    constexpr __representation(_Args&&... __args) : __payload_(std::forward<_Args>(__args)...), __value_(_Np + 2) {}

    bool __is_tombstone() { return __value_ == _Np + 2; }

    _LIBCPP_NO_UNIQUE_ADDRESS _Payload __payload_;
    unsigned char __value_;
  };

  static constexpr size_t __max_tombstone = 254;
};

template <size_t _PaddingSize>
struct __padding {
  char __padding_[_PaddingSize];
};

template <>
struct __padding<0> {};

// Pointers to a type that has an alignment greater than one always have the lowest bits set to zero. This is a single
// implementation for all the places where we have an invalid pointer to _Tp as the "invalid state" representation.
template <class _Tp>
struct __tombstone_traits_assume_aligned_pointer {
  template <class _Payload, size_t _Np>
  struct __representation {
    template <class... _Args>
    constexpr __representation(_Args&&... __args) : __payload_(std::forward<_Args>(__args)...), __value_(_Np + 1) {}

    bool __is_tombstone() { return __value_ == _Np + 1; }

    _LIBCPP_NO_UNIQUE_ADDRESS _Payload __payload_;
    _LIBCPP_NO_UNIQUE_ADDRESS __padding<sizeof(_Tp*) - 1 - __datasizeof_v<_Payload>> __padding_;
    unsigned char __value_;
  };

  static constexpr size_t __max_tombstone = 0; // TODO: Calculate this correctly
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

template <class _Tp, class = void>
struct alignas(_Tp) __dummy_payload {
  char __data_[__datasizeof_v<_Tp>];

  static_assert(alignof(__dummy_payload) == alignof(_Tp));
  static_assert(sizeof(__dummy_payload) == sizeof(_Tp));
  static_assert(__datasizeof_v<__dummy_payload> == __datasizeof_v<_Tp>);
};

template <class _Tp>
struct alignas(_Tp) __dummy_payload<_Tp, __enable_if_t<__datasizeof_v<_Tp> == 0>> {};

template <class _Tp, class _Payload, bool = is_trivially_destructible_v<_Tp> && is_trivially_destructible_v<_Payload>>
union _MaybeTombstone {
  using _Tombstone      = __tombstone_traits<_Tp>::template __representation<_Payload, 0>;
  using _DummyTombstone = __tombstone_traits<_Tp>::template __representation<__dummy_payload<_Payload>, 0>;

  _Tp __value_;
  _Tombstone __tombstone_;

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
    if consteval {
      return __builtin_is_within_lifetime(&__value_);
    } else {
      _DummyTombstone __dummy;
      static_assert(sizeof(_DummyTombstone) <= sizeof(*this));
      __builtin_memcpy(&__dummy, this, sizeof(_DummyTombstone));

      return !__dummy.__is_tombstone();
    }
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
  using _Tombstone      = __tombstone_traits<_Tp>::template __representation<_Payload, 0>;
  using _DummyTombstone = __tombstone_traits<_Tp>::template __representation<__dummy_payload<_Payload>, 0>;

  _Tp __value_;
  _Tombstone __tombstone_;

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
    if consteval {
      return __builtin_is_within_lifetime(&__value_);
    } else {
      _DummyTombstone __dummy;
      static_assert(sizeof(_DummyTombstone) <= sizeof(*this));
      __builtin_memcpy(&__dummy, this, sizeof(_DummyTombstone));

      return !__dummy.__is_tombstone();
    }
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 ~_MaybeTombstone() = default;
};

template <class _Tp, class _Payload>
struct __tombstoned_value final {
  using _Tombstone      = __tombstone_traits<_Tp>::template __representation<_Payload, 0>;
  using _DummyTombstone = __tombstone_traits<_Tp>::template __representation<__dummy_payload<_Payload>, 0>;

  static_assert(__builtin_is_implicit_lifetime(_DummyTombstone));
  static_assert(sizeof(_Tombstone) == sizeof(_DummyTombstone));
  static_assert(alignof(_Tombstone) == alignof(_DummyTombstone));
  static_assert(__datasizeof_v<_Tombstone> == __datasizeof_v<_DummyTombstone>);
  static_assert(sizeof(_Tombstone) <= sizeof(_Tp), "Trying to use tobstone which can't hold T");

  _MaybeTombstone<_Tp, _Payload> __data_;

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
