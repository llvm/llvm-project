//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator, class Size, class Function>
//    constexpr InputIterator      // constexpr after C++17
//    for_each_n(InputIterator first, Size n, Function f);

#include <algorithm>
#include <cassert>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <ranges>
#include <vector>

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

struct deque_test {
  std::deque<int>* d_;
  int* i_;

  deque_test(std::deque<int>& d, int& i) : d_(&d), i_(&i) {}

  void operator()(int& v) {
    assert(&(*d_)[*i_] == &v);
    ++*i_;
  }
};

/*TEST_CONSTEXPR_CXX26*/
void test_deque_and_join_view_iterators() { // TODO: Mark as TEST_CONSTEXPR_CXX26 once std::deque is constexpr
  {                                         // Verify that segmented deque iterators work properly
    int sizes[] = {0, 1, 2, 1023, 1024, 1025, 2047, 2048, 2049};
    for (const int size : sizes) {
      std::deque<int> d(size);
      int index = 0;

      std::for_each_n(d.begin(), d.size(), deque_test(d, index));
    }
  }
#if TEST_STD_VER >= 20
  { // Verify that join_view of lists work properly. Note that join_view of (non-random access) lists does
    // not produce segmented iterators.
    std::list<std::list<int>> lst = {{}, {0}, {1, 2}, {}, {3, 4, 5}, {6, 7, 8, 9}, {10}, {11, 12, 13}};
    auto v                        = lst | std::views::join;
    std::for_each_n(v.begin(), std::ranges::distance(v), [i = 0](int& a) mutable { assert(a == i++); });
  }
#endif
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    typedef cpp17_input_iterator<int*> Iter;
    int ia[]         = {0, 1, 2, 3, 4, 5};
    const unsigned s = sizeof(ia) / sizeof(ia[0]);

    {
      unsigned count = 0;
      Iter it        = std::for_each_n(Iter(ia), 0, [&count](int& i) {
        ++i;
        ++count;
      });
      assert(it == Iter(ia));
      assert(count == 0);
    }

    {
      unsigned count = 0;
      Iter it        = std::for_each_n(Iter(ia), s, [&count](int& i) {
        ++i;
        ++count;
      });
      assert(it == Iter(ia + s));
      assert(count == s);
      for (unsigned i = 0; i < s; ++i)
        assert(ia[i] == static_cast<int>(i + 1));
    }

    {
      unsigned count = 0;
      Iter it        = std::for_each_n(Iter(ia), 1, [&count](int& i) {
        ++i;
        ++count;
      });
      assert(it == Iter(ia + 1));
      assert(count == 1);
      for (unsigned i = 0; i < 1; ++i)
        assert(ia[i] == static_cast<int>(i + 2));
    }
  }

  {
    int ia[]            = {1, 3, 6, 7};
    int expected[]      = {3, 5, 8, 9};
    const std::size_t N = 4;

    auto it = std::for_each_n(std::begin(ia), N, [](int& a) { a += 2; });
    assert(it == (std::begin(ia) + N) && std::equal(std::begin(ia), std::end(ia), std::begin(expected)));
  }

  if (!TEST_IS_CONSTANT_EVALUATED) // TODO: Use TEST_STD_AT_LEAST_26_OR_RUNTIME_EVALUATED when std::deque is made constexpr
    test_deque_and_join_view_iterators();

#if TEST_STD_VER >= 20
  { // join_views of (random-access) vectors yield segmented iterators
    std::vector<std::vector<int>> vec = {{}, {0}, {1, 2}, {}, {3, 4, 5}, {6, 7, 8, 9}, {10}, {11, 12, 13}};
    auto v                            = vec | std::views::join;
    std::for_each_n(v.begin(), std::ranges::distance(v), [i = 0](int& a) mutable { assert(a == i++); });
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
