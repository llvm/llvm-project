//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_MIN_SEQUENCE_CONTAINER_H
#define SUPPORT_MIN_SEQUENCE_CONTAINER_H

#include <initializer_list>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

template <class T,
          class Iterator      = three_way_random_access_iterator<T*>,
          class ConstIterator = three_way_random_access_iterator<const T*>>
struct MinSequenceContainer {
  using value_type      = T;
  using difference_type = int;
  using size_type       = unsigned int;
  using iterator        = Iterator;
  using const_iterator  = ConstIterator;

  explicit MinSequenceContainer() = default;
  template <class It>
  explicit TEST_CONSTEXPR_CXX20 MinSequenceContainer(It first, It last) : data_(first, last) {}
  TEST_CONSTEXPR_CXX20 MinSequenceContainer(std::initializer_list<T> il) : data_(il) {}
#if TEST_STD_VER >= 23
  template <class Range>
  constexpr MinSequenceContainer(std::from_range_t, Range&& rg) : data_(std::from_range, std::forward<Range>(rg)) {}
#endif
  TEST_CONSTEXPR_CXX20 MinSequenceContainer(size_type n, T value) : data_(n, value) {}

  TEST_CONSTEXPR_CXX20 MinSequenceContainer& operator=(std::initializer_list<T> il) { data_ = il; }

  template <class It>
  TEST_CONSTEXPR_CXX20 void assign(It first, It last) {
    data_.assign(first, last);
  }
  TEST_CONSTEXPR_CXX20 void assign(std::initializer_list<T> il) { data_.assign(il); }
  TEST_CONSTEXPR_CXX20 void assign(size_type n, value_type t) { data_.assign(n, t); }
#if TEST_STD_VER >= 23
  template <class Range>
  constexpr void assign_range(Range&& rg) {
    data_.assign_range(std::forward<Range>(rg));
  }
#endif
  TEST_CONSTEXPR_CXX20 iterator begin() { return iterator(data_.data()); }
  TEST_CONSTEXPR_CXX20 const_iterator begin() const { return const_iterator(data_.data()); }
  TEST_CONSTEXPR_CXX20 const_iterator cbegin() const { return const_iterator(data_.data()); }
  TEST_CONSTEXPR_CXX20 iterator end() { return begin() + size(); }
  TEST_CONSTEXPR_CXX20 const_iterator end() const { return begin() + size(); }
  TEST_CONSTEXPR_CXX20 size_type size() const { return static_cast<size_type>(data_.size()); }
  TEST_CONSTEXPR_CXX20 bool empty() const { return data_.empty(); }

  TEST_CONSTEXPR_CXX20 void clear() { data_.clear(); }

  template <class It>
  TEST_CONSTEXPR_CXX20 iterator insert(const_iterator p, It first, It last) {
    return from_vector_iterator(data_.insert(to_vector_iterator(p), first, last));
  }

  TEST_CONSTEXPR_CXX20 iterator insert(const_iterator p, T value) {
    return from_vector_iterator(data_.insert(to_vector_iterator(p), std::move(value)));
  }

  TEST_CONSTEXPR_CXX20 iterator insert(const_iterator p, size_type n, T value) {
    return from_vector_iterator(data_.insert(to_vector_iterator(p), n, value));
  }

  TEST_CONSTEXPR_CXX20 iterator insert(const_iterator p, std::initializer_list<T> il) {
    return from_vector_iterator(data_.insert(to_vector_iterator(p), il));
  }

#if TEST_STD_VER >= 23
  template <class Range>
  constexpr iterator insert_range(const_iterator p, Range&& rg) {
    return from_vector_iterator(data_.insert_range(to_vector_iterator(p), std::forward<Range>(rg)));
  }
#endif

  TEST_CONSTEXPR_CXX20 iterator erase(const_iterator first, const_iterator last) {
    return from_vector_iterator(data_.erase(to_vector_iterator(first), to_vector_iterator(last)));
  }

  TEST_CONSTEXPR_CXX20 iterator erase(const_iterator iter) {
    return from_vector_iterator(data_.erase(to_vector_iterator(iter)));
  }

  template <class... Args>
  TEST_CONSTEXPR_CXX20 iterator emplace(const_iterator pos, Args&&... args) {
    return from_vector_iterator(data_.emplace(to_vector_iterator(pos), std::forward<Args>(args)...));
  }

private:
  TEST_CONSTEXPR_CXX20 std::vector<T>::const_iterator to_vector_iterator(const_iterator cit) const {
    return cit - cbegin() + data_.begin();
  }

  TEST_CONSTEXPR_CXX20 iterator from_vector_iterator(std::vector<T>::iterator it) {
    return it - data_.begin() + begin();
  }

  std::vector<T> data_;
};

namespace MinSequenceContainer_detail {

// MinSequenceContainer is non-allocator-aware, because flat_set supports
// such (non-STL) container types, and we want to make sure they are supported.
template <class T>
concept HasAllocatorType = requires { typename T::allocator_type; };
static_assert(!HasAllocatorType<MinSequenceContainer<int>>);

// MinSequenceContainer by itself doesn't support .emplace(), because we want
// to at least somewhat support (non-STL) container types with nothing but .insert().
template <class T>
concept HasEmplace = requires(T& t) { t.emplace(42); };
static_assert(!HasEmplace<MinSequenceContainer<int>>);

} // namespace MinSequenceContainer_detail

#endif // SUPPORT_MIN_SEQUENCE_CONTAINER_H
