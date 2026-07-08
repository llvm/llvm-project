// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CAPACITY_AWARE_ITERATOR_H
#define _LIBCPP___CAPACITY_AWARE_ITERATOR_H

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
#include <__memory/pointer_traits.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/is_pointer.h>
#include <__utility/declval.h>
#include <__utility/move.h>

#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26

_LIBCPP_BEGIN_NAMESPACE_STD

// __capacity_aware_iterator is an iterator that wraps a contiguous iterator and encodes the maximum number of
// elements that can appear in a range of such iterators. That maximum number of elements must be known at compile-time.
// As of writing, the only standard library containers which fulfill these requirements are inplace_vector and optional.
//
// It also embeds a tag type to prevent mixing iterators from e.g. different containers. This also allows for some
// algorithms to detect this iterator and perform optimizations based on the added semantic information.

template <class _Iter, class _Tag, size_t _RangeMaxElements>
class __capacity_aware_iterator {
private:
  _Iter __iter_;

  template <class, class, size_t>
  friend class __capacity_aware_iterator;

public:
  static_assert(contiguous_iterator<_Iter>, "__capacity_aware_iterator can only be used with contiguous iterators");
  constexpr static bool __can_track_count_ = false;

  using iterator_category = iterator_traits<_Iter>::iterator_category;
  using iterator_concept  = contiguous_iterator_tag;
  using difference_type   = iter_difference_t<_Iter>;
  using pointer           = iterator_traits<_Iter>::pointer;
  using reference         = iter_reference_t<_Iter>;
  using value_type        = iter_value_t<_Iter>;

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator()
    requires is_default_constructible_v<_Iter>
  = default;

  template <typename _Iter2>
    requires is_convertible_v<_Iter2, _Iter>
  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator(
      const __capacity_aware_iterator<_Iter2, _Tag, _RangeMaxElements>& __y) noexcept
      : __iter_(__y.__iter_) {}

  template <class _It, class _Tag2, size_t _RangeMaxElems2>
  _LIBCPP_HIDE_FROM_ABI friend constexpr auto __make_capacity_aware_iterator(_It __iter) noexcept;

private:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit __capacity_aware_iterator(_Iter __iter) : __iter_(std::move(__iter)) {}

public:
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const noexcept { return *__iter_; }
  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator->() const noexcept { return std::__to_address(__iter_); }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator++() noexcept {
    ++__iter_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator operator++(int) noexcept {
    __capacity_aware_iterator __tmp(*this);
    ++*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator--() noexcept {
    --__iter_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator operator--(int) noexcept {
    __capacity_aware_iterator __tmp(*this);
    --*this;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator+=(difference_type __n) noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        static_cast<size_t>(__n >= 0 ? __n : -__n) <= _RangeMaxElements,
        "__capacity_aware_iterator::operator+=: Attempting to move iterator past its container's possible range");

    __iter_ += __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __capacity_aware_iterator& operator-=(difference_type __n) noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        static_cast<size_t>(__n >= 0 ? __n : -__n) <= _RangeMaxElements,
        "__capacity_aware_iterator::operator-=: Attempting to move iterator past its container's possible range");

    __iter_ -= __n;
    return *this;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        static_cast<size_t>(__n >= 0 ? __n : -__n) < _RangeMaxElements,
        "__capacity_aware_iterator::operator[]: Attempting to index iterator past its container's possible range");
    return *(*this + __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) noexcept {
    return __x.__iter_ == __y.__iter_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) noexcept {
    if constexpr (three_way_comparable_with<_Iter, _Iter, strong_ordering>) {
      return __x.__iter_ <=> __y.__iter_;
    } else {
      if (__x.__iter_ < __y.__iter_) {
        return strong_ordering::less;
      } else if (__x.__iter_ == __y.__iter_) {
        return strong_ordering::equal;
      }
      return strong_ordering::greater;
    }
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __capacity_aware_iterator
  operator+(const __capacity_aware_iterator& __i, difference_type __n) noexcept {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __capacity_aware_iterator
  operator+(difference_type __n, const __capacity_aware_iterator& __i) noexcept {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr __capacity_aware_iterator
  operator-(const __capacity_aware_iterator& __i, difference_type __n) noexcept {
    auto __tmp = __i;
    __tmp -= __n;
    return __tmp;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type
  operator-(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) noexcept {
    return difference_type(__x.__iter_ - __y.__iter_);
  }
};

template <class _Tp>
consteval bool __range_fits_in_alignment(std::size_t __num_elems) {
  auto __bits = std::countr_zero(alignof(_Tp));

  // Example: For alignof(T) == 4, we have two bits free, which has a range of 0-3. We need to
  // reserve one for the end position, so __num_elems must be < 3.
  auto __allowed_range = (1 << __bits) - 1;
  return __allowed_range > __num_elems;
}

// A specialization of capacity_aware_iterator where we store a runtime count of the current position inside
// the unused bottom bits of a pointer to T. Only applies if capacity can fit inside those bits.
template <class _Iter, class _Tag, size_t _RangeMaxElements>
  requires(std::is_pointer_v<_Iter> &&
           std::__range_fits_in_alignment<decltype(* std::declval<_Iter>())>(_RangeMaxElements))
class __capacity_aware_iterator<_Iter, _Tag, _RangeMaxElements> {
  constexpr static std::size_t __bits_ = std::countr_zero(alignof(decltype(*std::declval<_Iter>())));

  union {
    _Iter __ptr_;
    std::uintptr_t __current_ : __bits_;
  };

  template <class, class, size_t>
  friend class __capacity_aware_iterator;

public:
  constexpr static bool __can_track_count_ = false;

  using iterator_category = iterator_traits<_Iter>::iterator_category;
  using iterator_concept  = contiguous_iterator_tag;
  using difference_type   = iter_difference_t<_Iter>;
  using pointer           = iterator_traits<_Iter>::pointer;
  using reference         = iter_reference_t<_Iter>;
  using value_type        = iter_value_t<_Iter>;

  constexpr __capacity_aware_iterator()
    requires is_default_constructible_v<_Iter>
  = default;

  template <typename _Iter2>
    requires is_convertible_v<_Iter2, _Iter>
  constexpr __capacity_aware_iterator(const __capacity_aware_iterator<_Iter2, _Tag, _RangeMaxElements>& __y) noexcept
      : __ptr_(__y.__ptr_) {
    if !consteval {
      __current_ = __y.__current_;
    }
  }

  template <class _It, class _Tag2, size_t _RangeMaxElems2>
  friend constexpr auto __make_capacity_aware_iterator(_It __iter) noexcept;

private:
  constexpr explicit __capacity_aware_iterator(_Iter __iter) : __ptr_(std::move(__iter)) {
    if !consteval {
      __current_ = 0u;
    }
  }

  constexpr _Iter __curr_iter() const {
    if consteval {
      return __ptr_;
    }

    return std::bit_cast<_Iter>((std::bit_cast<std::uintptr_t>(__ptr_) >> __bits_) << __bits_) + __current_;
  }

  constexpr void __update(difference_type __n) {
    if consteval {
      __ptr_ += __n;
    } else {
      __current_ += __n;
    }
  }

public:
  [[nodiscard]] constexpr decltype(auto) operator*() const noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __current_ < _RangeMaxElements,
          "__capacity_aware_iterator::operator*: Attempt to dereference an iterator at the end");
    }

    return *(__curr_iter());
  }

  constexpr decltype(auto) operator->() const noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __current_ < _RangeMaxElements,
          "__capacity_aware_iterator::operator->: Attempt to dereference an iterator at the end");
    }

    return __curr_iter();
  }

  constexpr __capacity_aware_iterator& operator++() noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __current_ != _RangeMaxElements,
          "__capacity_aware_iterator::operator++: Attempt to advance an iterator past the end");
    }

    __update(1);

    return *this;
  }

  constexpr __capacity_aware_iterator operator++(int) noexcept {
    __capacity_aware_iterator __tmp(*this);
    ++*this;
    return __tmp;
  }

  constexpr __capacity_aware_iterator& operator--() noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          __current_ != 0u, "__capacity_aware_iterator::operator--: Attempt to rewind an iterator past the start");
    }

    __update(-1);

    return *this;
  }

  constexpr __capacity_aware_iterator operator--(int) noexcept {
    __capacity_aware_iterator __tmp(*this);
    --*this;
    return __tmp;
  }

  constexpr __capacity_aware_iterator& operator+=(difference_type __n) noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          (static_cast<difference_type>(__current_) + __n) >= 0,
          "__capacity_aware_iterator::operator+=: Attempt to rewind an iterator past the start");
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          static_cast<std::size_t>(__current_ + __n) <= _RangeMaxElements,
          "__capacity_aware_iterator::operator+=: Attempt to advance an iterator past the end");
    }

    __update(__n);

    return *this;
  }

  constexpr __capacity_aware_iterator& operator-=(difference_type __n) noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          (static_cast<difference_type>(__current_) - __n) >= 0,
          "__capacity_aware_iterator::operator-=: Attempt to rewind an iterator past the start");
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          static_cast<std::size_t>(__current_ - __n) <= _RangeMaxElements,
          "__capacity_aware_iterator::operator-=: Attempt to advance an iterator past the end");
    }

    __update(-__n);

    return *this;
  }

  [[nodiscard]] constexpr decltype(auto) operator[](difference_type __n) const noexcept {
    if !consteval {
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          (static_cast<difference_type>(__current_) + __n) >= 0,
          "__capacity_aware_iterator::operator[]: Attempt to index an iterator past the start");
      _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
          static_cast<std::size_t>(__current_ + __n) < _RangeMaxElements,
          "__capacity_aware_iterator::operator[]: Attempt to index an iterator at or past the end");
    }
    return *(*this + __n);
  }

  friend constexpr bool
  operator==(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) noexcept {
    return __x.__curr_iter() == __y.__curr_iter();
  }

  friend constexpr auto
  operator<=>(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) noexcept {
    if constexpr (three_way_comparable_with<_Iter, _Iter, strong_ordering>) {
      return __x.__curr_iter() <=> __y.__curr_iter();
    } else {
      if (__x.__curr_iter() < __y.__curr_iter()) {
        return strong_ordering::less;
      } else if (__x.__curr_iter() == __y.__curr_iter()) {
        return strong_ordering::equal;
      }
      return strong_ordering::greater;
    }
  }

  [[nodiscard]] friend constexpr __capacity_aware_iterator
  operator+(const __capacity_aware_iterator& __i, difference_type __n) noexcept {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  [[nodiscard]] friend constexpr __capacity_aware_iterator
  operator+(difference_type __n, const __capacity_aware_iterator& __i) noexcept {
    auto __tmp = __i;
    __tmp += __n;
    return __tmp;
  }

  [[nodiscard]] friend constexpr __capacity_aware_iterator
  operator-(const __capacity_aware_iterator& __i, difference_type __n) noexcept {
    auto __tmp = __i;
    __tmp -= __n;
    return __tmp;
  }

  [[nodiscard]] friend constexpr difference_type
  operator-(const __capacity_aware_iterator& __x, const __capacity_aware_iterator& __y) noexcept {
    return difference_type(__x.__curr_iter() - __y.__curr_iter());
  }
};

template <class _It, class _Tag2, size_t _RangeMaxElems2>
_LIBCPP_HIDE_FROM_ABI constexpr auto __make_capacity_aware_iterator(_It __iter) noexcept {
  return __capacity_aware_iterator<_It, _Tag2, _RangeMaxElems2>(__iter);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CAPACITY_AWARE_ITERATOR_H
