//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that whats used for __wrap_iter<T> does not try to look into T
// for ADL

#include <__iterator/wrap_iter.h>
#include <vector>
#include <array>
#include <iterator>

#include "test_macros.h"

#ifdef _LIBCPP_ABI_WRAP_ITER_ADL_PROOF

struct incomplete;
template <class T>
struct holder {
  T t;
};

struct nefarious {
  friend void operator<=>(std::vector<nefarious>::iterator, std::vector<nefarious>::iterator)            = delete;
  friend void operator==(std::vector<nefarious>::const_iterator, std::vector<nefarious>::const_iterator) = delete;
};

template <class Container>
void test() {
  using iterator       = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  (void)(iterator() < iterator());
  (void)(iterator() < const_iterator());
  (void)(const_iterator() < const_iterator());
  (void)(iterator() == iterator());
  (void)(iterator() == const_iterator());
  (void)(const_iterator() == const_iterator());

#  if TEST_STD_VER >= 20
  static_assert(std::contiguous_iterator<iterator>);
  static_assert(std::contiguous_iterator<const_iterator>);
  static_assert(std::ranges::contiguous_range<Container>);
  static_assert(std::ranges::contiguous_range<const Container>);
#  endif
}

void tests() {
  test<std::vector<holder<incomplete>*>>();
  test<std::array<holder<incomplete>*, 10>>();
  test<std::vector<nefarious>>();
  test<std::array<nefarious, 10>>();
}
#endif
