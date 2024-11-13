//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXCEPTION_TEST_HELPER_H
#define EXCEPTION_TEST_HELPER_H

#include "count_new.h"

template <class T>
struct throwing_allocator {
  using value_type      = T;
  using is_always_equal = std::false_type;

  bool throw_on_copy_ = false;

  throwing_allocator(bool throw_on_ctor = true, bool throw_on_copy = false) : throw_on_copy_(throw_on_copy) {
    if (throw_on_ctor)
      throw 0;
  }

  template <class U>
  throwing_allocator(const throwing_allocator<U>& rhs) : throw_on_copy_(rhs.throw_on_copy_) {
    if (throw_on_copy_)
      throw 0;
  }

  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* ptr, std::size_t n) { std::allocator<T>().deallocate(ptr, n); }

  template <class U>
  friend bool operator==(const throwing_allocator&, const throwing_allocator<U>&) {
    return true;
  }
};

template <class T, class IterCat>
struct throwing_iterator {
  using iterator_category = IterCat;
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using reference         = T&;
  using pointer           = T*;

  int i_;
  T v_;

  throwing_iterator(int i = 0, const T& v = T()) : i_(i), v_(v) {}

  reference operator*() {
    if (i_ == 1)
      throw 1;
    return v_;
  }

  friend bool operator==(const throwing_iterator& lhs, const throwing_iterator& rhs) { return lhs.i_ == rhs.i_; }
  friend bool operator!=(const throwing_iterator& lhs, const throwing_iterator& rhs) { return lhs.i_ != rhs.i_; }

  throwing_iterator& operator++() {
    ++i_;
    return *this;
  }

  throwing_iterator operator++(int) {
    auto tmp = *this;
    ++i_;
    return tmp;
  }
};

inline void check_new_delete_called() {
  assert(globalMemCounter.new_called == globalMemCounter.delete_called);
  assert(globalMemCounter.new_array_called == globalMemCounter.delete_array_called);
  assert(globalMemCounter.aligned_new_called == globalMemCounter.aligned_delete_called);
  assert(globalMemCounter.aligned_new_array_called == globalMemCounter.aligned_delete_array_called);
}

#endif // EXCEPTION_TEST_HELPER_H