//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter1, ForwardIterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   Iter2
//   swap_ranges(Iter1 first1, Iter1 last1, Iter2 first2);

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <utility>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

struct TestPtr {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::forward_iterator_list<int*>(), TestPtrImpl<Iter>());
  }

  template <class Iter1>
  struct TestPtrImpl {
    template <class Iter2>
    TEST_CONSTEXPR_CXX20 void operator()() {
      int a[] = {1, 2, 3};
      int b[] = {4, 5, 6};
      Iter2 r = std::swap_ranges(Iter1(a), Iter1(a + 3), Iter2(b));
      assert(base(r) == b + 3);
      assert(a[0] == 4 && a[1] == 5 && a[2] == 6);
      assert(b[0] == 1 && b[1] == 2 && b[2] == 3);
    }
  };
};

#if TEST_STD_VER >= 11
struct TestUniquePtr {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::forward_iterator_list<std::unique_ptr<int>*>(), TestUniquePtrImpl<Iter>());
  }

  template <class Iter1>
  struct TestUniquePtrImpl {
    template <class Iter2>
    TEST_CONSTEXPR_CXX23 void operator()() {
      std::unique_ptr<int> a[3];
      for (int k = 0; k < 3; ++k)
        a[k].reset(new int(k + 1));
      std::unique_ptr<int> b[3];
      for (int k = 0; k < 3; ++k)
        b[k].reset(new int(k + 4));
      Iter2 r = std::swap_ranges(Iter1(a), Iter1(a + 3), Iter2(b));
      assert(base(r) == b + 3);
      assert(*a[0] == 4 && *a[1] == 5 && *a[2] == 6);
      assert(*b[0] == 1 && *b[1] == 2 && *b[2] == 3);
    }
  };
};
#endif

TEST_CONSTEXPR_CXX20 bool test_simple_cases() {
  {
    std::array<int, 3> a = {1, 2, 3}, a0 = a;
    std::array<int, 3> b = {4, 5, 6}, b0 = b;
    std::swap_ranges(a.begin(), a.end(), b.begin());
    assert(a == b0);
    assert(b == a0);
  }
  {
    std::array<std::array<int, 2>, 2> a = {{{0, 1}, {2, 3}}}, a0 = a;
    std::array<std::array<int, 2>, 2> b = {{{9, 8}, {7, 6}}}, b0 = b;
    std::swap_ranges(a.begin(), a.end(), b.begin());
    assert(a == b0);
    assert(b == a0);
  }
  {
    std::array<std::array<int, 3>, 3> a = {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}}, a0 = a;
    std::array<std::array<int, 3>, 3> b = {{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}}, b0 = b;
    std::swap_ranges(a.begin(), a.end(), b.begin());
    assert(a == b0);
    assert(b == a0);
  }

  return true;
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test_simple_cases();
  types::for_each(types::forward_iterator_list<int*>(), TestPtr());
#if TEST_STD_VER >= 11
  types::for_each(types::forward_iterator_list<std::unique_ptr<int>*>(), TestUniquePtr());
#endif
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif

  return 0;
}
