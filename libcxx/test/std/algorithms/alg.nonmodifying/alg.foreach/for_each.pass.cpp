//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Callable<auto, Iter::reference> Function>
//   requires CopyConstructible<Function>
//   constexpr Function   // constexpr after C++17
//   for_each(Iter first, Iter last, Function f);

#include <algorithm>
#include <cassert>
#include <deque>

#include "test_macros.h"
#include "test_iterators.h"

struct for_each_test {
  TEST_CONSTEXPR for_each_test(int c) : count(c) {}
  int count;
  TEST_CONSTEXPR_CXX14 void operator()(int& i) {
    ++i;
    ++count;
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  int ia[]         = {0, 1, 2, 3, 4, 5};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  for_each_test f = std::for_each(cpp17_input_iterator<int*>(ia), cpp17_input_iterator<int*>(ia + s), for_each_test(0));
  assert(f.count == s);
  for (unsigned i = 0; i < s; ++i)
    assert(ia[i] == static_cast<int>(i + 1));

  return true;
}

struct deque_test {
  std::deque<int>* d_;
  int* i_;

  deque_test(std::deque<int>& d, int& i) : d_(&d), i_(&i) {}

  void operator()(int& v) {
    assert(&(*d_)[(*i_)++] == &v);
  }
};

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  // check that segmented iterators work properly
  std::deque<int> d(50);
  int index = 0;

  std::for_each(d.begin(), d.end(), deque_test(d, index));

  return 0;
}
