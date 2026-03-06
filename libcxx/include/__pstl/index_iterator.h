//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_INDEX_ITERATOR_H
#define _LIBCPP___PSTL_INDEX_ITERATOR_H

#include <__config>
#include <__iterator/iterator_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// This iterator type is a stopgap solution to use the existing backend algorithms in PSTL and be able to implement
// position-sensitive algorithms on top. While providing an iterator interface, this type is essentially a wrapper over
// an index type.
// Once the backends start providing a less restrictive interface, e.g. working with chunks of iterator
// ranges, and supporting forward iterators, any algorithms that were implemented using this index wrapper should be
// reimplemented using iterators as-is.

template <typename _Index>
class __index_iterator {
public:
  using value_type        = _Index;
  using difference_type   = _Index;
  using iterator_category = std::random_access_iterator_tag;
  using reference         = _Index;
  using pointer           = void;

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator() noexcept = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __index_iterator(_Index __index) noexcept : __index_(__index) {}

  _LIBCPP_HIDE_FROM_ABI constexpr _Index operator*() const noexcept { return __index_; }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator& operator++() noexcept {
    ++__index_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator operator++(int) noexcept {
    auto __tmp = *this;
    ++__index_;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator& operator--() noexcept {
    --__index_;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator operator--(int) noexcept {
    auto __tmp = *this;
    --__index_;
    return __tmp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator operator+(_Index __n) const noexcept {
    return __index_iterator{__index_ + __n};
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __index_iterator operator+(_Index __n, const __index_iterator& __x) noexcept {
    return __x + __n;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator& operator+=(_Index __n) noexcept {
    __index_ += __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator operator-(_Index __n) const noexcept {
    return __index_iterator{__index_ - __n};
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr _Index
  operator-(const __index_iterator& __lhs, const __index_iterator& __rhs) noexcept {
    return __lhs.__index_ - __rhs.__index_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __index_iterator& operator-=(_Index __n) noexcept {
    __index_ -= __n;
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr _Index operator[](_Index __n) const noexcept { return __index_ + __n; }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __index_iterator& __other) const noexcept {
    return __index_ == __other.__index_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(const __index_iterator& __other) const noexcept {
    return !(*this == __other);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator<(const __index_iterator& __other) const noexcept {
    return __index_ < __other.__index_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator<=(const __index_iterator& __other) const noexcept {
    return __index_ <= __other.__index_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator>(const __index_iterator& __other) const noexcept {
    return __index_ > __other.__index_;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr bool operator>=(const __index_iterator& __other) const noexcept {
    return __index_ >= __other.__index_;
  }

private:
  _Index __index_ = {};
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_INDEX_ITERATOR_H
