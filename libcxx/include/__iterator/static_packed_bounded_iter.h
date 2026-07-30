// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP__STATIC_PACKED_BOUNDED_ITER_H
#define _LIBCPP__STATIC_PACKED_BOUNDED_ITER_H

#include <__assert>
#include <__bit/bit_cast.h>
#include <__bit/countr.h>
#include <__compare/ordering.h>
#include <__compare/three_way_comparable.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/is_pointer.h>

#include <cstdint>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

// static_packed_bounded_iter is a bounded, contiguous iterator that is aware of its container's (compile-time) maximum
// capacity. It reuses the bottom unused bits of a pointer to keep track of its current position.
// This only applies if the container's maximum range can fit inside (2 ^ available_bits) - 1

_LIBCPP_BEGIN_NAMESPACE_STD

consteval bool __range_fits_in_alignment(size_t __alignment, size_t __num_elems) {
  size_t __bits = std::countr_zero(__alignment);

  // Example: For alignof(T) == 4, we have two bits free, which has a range of 0-3. We need to
  // reserve one for the end position, so __num_elems must be < 3.
  size_t __allowed_range = (1 << __bits) - 1;
  return __allowed_range > __num_elems;
}

template <class _Ptr, class _Tag, size_t _RangeCapacity>
  requires(is_pointer_v<_Ptr> && std::__range_fits_in_alignment(_LIBCPP_ALIGNOF(iter_value_t<_Ptr>), _RangeCapacity))
class __static_packed_bounded_iterator {
public:
  using iterator_category = iterator_traits<_Ptr>::iterator_category;
  using iterator_concept  = contiguous_iterator_tag;
  using difference_type   = iter_difference_t<_Ptr>;
  using pointer           = iterator_traits<_Ptr>::pointer;
  using reference         = iter_reference_t<_Ptr>;
  using value_type        = iter_value_t<_Ptr>;

private:
  static constexpr uintptr_t __count_mask_ = (1 << std::countr_zero(_LIBCPP_ALIGNOF(value_type))) - 1;
  static constexpr uintptr_t __ptr_mask_   = ~__count_mask_;

  union {
    pointer __ptr_;
    alignas(_Ptr) unsigned char __data_[sizeof(_Ptr)];
  };

  uintptr_t __as_num() const { return std::bit_cast<uintptr_t>(__data_); }
  uintptr_t __count() const { return __as_num() & __count_mask_; }

  constexpr _Ptr __current() const {
    if consteval {
      return __ptr_;
    } else {
      return std::bit_cast<pointer>(__as_num() & __ptr_mask_) + __count();
    }
  }

  constexpr void __update(difference_type __n) {
    if consteval {
      __ptr_ += __n;
    } else {
      uintptr_t __num = __as_num() + __n;
      __builtin_memcpy(__data_, &__num, sizeof(__data_));
    }
  }

  constexpr explicit __static_packed_bounded_iterator(_Ptr __p) noexcept : __ptr_(__p) {
    if !consteval {
      __update(0);
    }
  }

public:
  template <class _Ptr2, class _Tag2, size_t _RangeCapacity2>
  friend constexpr auto __make_static_packed_bounded_iter(_Ptr2) noexcept;

  constexpr __static_packed_bounded_iterator()
    requires is_default_constructible_v<_Ptr>
  = default;

  template <class _Ptr2>
    requires is_convertible_v<_Ptr2, _Ptr>
  constexpr __static_packed_bounded_iterator(const __static_packed_bounded_iterator<_Ptr2, _Tag, _RangeCapacity>& __y)
      : __ptr_(__y.__ptr_) {}

  [[nodiscard]] constexpr decltype(auto) operator*() const noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __count() != _RangeCapacity,
          "__static_packed_bounded_iterator::operator*: Attempt to dereference an iterator at the end");
    }

    return *(__current());
  }

  constexpr decltype(auto) operator->() const noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __count() != _RangeCapacity,
          "__static_packed_bounded_iterator::operator->: Attempt to dereference an iterator at the end");
    }

    return __current();
  }

  constexpr __static_packed_bounded_iterator& operator++() noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __count() != _RangeCapacity,
          "__static_packed_bounded_iterator::operator++: Attempt to advance an iterator past the end");
    }

    __update(1);

    return *this;
  }

  constexpr __static_packed_bounded_iterator operator++(int) noexcept {
    __static_packed_bounded_iterator __tmp(*this);
    ++*this;
    return __tmp;
  }

  constexpr __static_packed_bounded_iterator& operator--() noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __count() != 0u,
          "__static_packed_bounded_iterator::operator--: Attempt to rewind an iterator past the start");
    }

    __update(-1);

    return *this;
  }

  constexpr __static_packed_bounded_iterator operator--(int) noexcept {
    __static_packed_bounded_iterator __tmp(*this);
    --*this;
    return __tmp;
  }

  constexpr __static_packed_bounded_iterator& operator+=(difference_type __n) noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          (static_cast<difference_type>(__count()) + __n) >= 0,
          "__static_packed_bounded_iterator::operator+=: Attempt to rewind an iterator past the start");
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          static_cast<size_t>(__count() + __n) <= _RangeCapacity,
          "__static_packed_bounded_iterator::operator+=: Attempt to advance an iterator past the end");
    }

    __update(__n);

    return *this;
  }

  constexpr __static_packed_bounded_iterator& operator-=(difference_type __n) noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          (static_cast<difference_type>(__count()) - __n) >= 0,
          "__static_packed_bounded_iterator::operator-=: Attempt to rewind an iterator past the start");
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          static_cast<size_t>(__count() - __n) <= _RangeCapacity,
          "__static_packed_bounded_iterator::operator-=: Attempt to advance an iterator past the end");
    }

    __update(-__n);

    return *this;
  }

  [[nodiscard]] constexpr decltype(auto) operator[](difference_type __n) const noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          (static_cast<difference_type>(__count()) + __n) >= 0,
          "__static_packed_bounded_iterator::operator[]: Attempt to index an iterator past the start");
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          static_cast<size_t>(__count() + __n) < _RangeCapacity,
          "__static_packed_bounded_iterator::operator[]: Attempt to index an iterator at or past the end");
    }
    return *(*this + __n);
  }

  friend constexpr bool
  operator==(const __static_packed_bounded_iterator& __x, const __static_packed_bounded_iterator& __y) noexcept {
    return __x.__current() == __y.__current();
  }

  friend constexpr auto
  operator<=>(const __static_packed_bounded_iterator& __x, const __static_packed_bounded_iterator& __y) noexcept {
    return __x.__current() <=> __y.__current();
  }

  [[nodiscard]] friend constexpr __static_packed_bounded_iterator
  operator+(const __static_packed_bounded_iterator& __i, difference_type __n) noexcept {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  [[nodiscard]] friend constexpr __static_packed_bounded_iterator
  operator+(difference_type __n, const __static_packed_bounded_iterator& __i) noexcept {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  [[nodiscard]] friend constexpr __static_packed_bounded_iterator
  operator-(const __static_packed_bounded_iterator& __i, difference_type __n) noexcept {
    auto __tmp = __i;
    __tmp -= __n;
    return __tmp;
  }

  [[nodiscard]] friend constexpr difference_type
  operator-(const __static_packed_bounded_iterator& __x, const __static_packed_bounded_iterator& __y) noexcept {
    return difference_type(__x.__current() - __y.__current());
  }
};

template <class _Ptr, class _Tag, size_t _RangeCapacity>
constexpr auto __make_static_packed_bounded_iter(_Ptr __p) noexcept {
  return __static_packed_bounded_iterator<_Ptr, _Tag, _RangeCapacity>(__p);
}

_LIBCPP_END_NAMESPACE_STD

#endif
_LIBCPP_POP_MACROS
#endif
