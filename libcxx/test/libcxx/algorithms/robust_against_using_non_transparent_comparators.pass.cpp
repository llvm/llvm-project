//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <iterator>

#include "test_macros.h"

template <class T>
struct Iterator {
  using value_type        = T;
  using pointer           = value_type*;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;
  struct reference {
    T* ptr_;

    reference(T* ptr) : ptr_(ptr) {}

    friend bool operator<(reference a, reference b) { return *a.ptr_ < *b.ptr_; }
    friend bool operator<(reference a, value_type const& b) { return *a.ptr_ < b; }
    friend bool operator<(value_type const& a, reference b) { return a < *b.ptr_; }

    operator T&() const;
  };

  Iterator& operator++() {
    ptr_++;
    return *this;
  }

  Iterator operator++(int) {
    Iterator tmp = *this;
    ptr_++;
    return tmp;
  }

  friend bool operator==(Iterator const& a, Iterator const& b) { return a.ptr_ == b.ptr_; }
  friend bool operator!=(Iterator const& a, Iterator const& b) { return !(a == b); }

  reference operator*() const { return reference(ptr_); }

  explicit Iterator(T* ptr) : ptr_(ptr) {}
  Iterator()                = default;
  Iterator(Iterator const&) = default;
  Iterator(Iterator&&)      = default;

  Iterator& operator=(Iterator const&) = default;
  Iterator& operator=(Iterator&&)      = default;

private:
  T* ptr_;
};

int main(int, char**) {
  int array[5] = {1, 2, 3, 4, 5};
  Iterator<int> first(array);
  Iterator<int> middle(array + 3);
  Iterator<int> last(array + 5);
  (void)std::binary_search(first, last, 3);
  (void)std::equal_range(first, last, 3);
  (void)std::includes(first, last, first, last);
  (void)std::is_sorted_until(first, last);
  (void)std::is_sorted(first, last);
  (void)std::lexicographical_compare(first, last, first, last);
  (void)std::lower_bound(first, last, 3);
  (void)std::max_element(first, last);
  (void)std::min_element(first, last);
  (void)std::minmax_element(first, last);
  (void)std::upper_bound(first, last, 3);

  return 0;
}
