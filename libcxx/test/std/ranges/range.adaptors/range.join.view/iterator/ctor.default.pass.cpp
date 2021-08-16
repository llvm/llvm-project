//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// iterator() requires default_initializable<OuterIter> = default;

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

template<class T>
struct DefaultCtorParent : std::ranges::view_base {
  T *ptr_;
  constexpr DefaultCtorParent(T *ptr) : ptr_(ptr) {}

  constexpr cpp17_input_iterator<T *> begin() { return cpp17_input_iterator<T *>(ptr_); }
  constexpr cpp17_input_iterator<const T *> begin() const { return cpp17_input_iterator<const T *>(ptr_); }
  constexpr T *end() { return ptr_ + 4; }
  constexpr const T *end() const { return ptr_ + 4; }
};

template<class T>
constexpr bool operator==(const cpp17_input_iterator<T*> &lhs, const T *rhs) { return lhs.base() == rhs; }
template<class T>
constexpr bool operator==(const T *lhs, const cpp17_input_iterator<T*> &rhs) { return rhs.base() == lhs; }

constexpr bool test() {
  using Base = DefaultCtorParent<ChildView>;
  // Note, only the outer iterator is default_initializable:
  static_assert( std::default_initializable<std::ranges::iterator_t<Base>>);
  static_assert(!std::default_initializable<std::ranges::iterator_t<std::ranges::range_reference_t<Base>>>);

  std::ranges::iterator_t<std::ranges::join_view<Base>> iter1;
  (void) iter1;

  static_assert(!std::default_initializable<std::ranges::iterator_t<std::ranges::join_view<ParentView<ChildView>>>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
