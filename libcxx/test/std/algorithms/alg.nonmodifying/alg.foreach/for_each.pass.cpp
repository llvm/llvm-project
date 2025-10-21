//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Callable<auto, Iter::reference> Function>
//   constexpr Function   // constexpr since C++20
//   for_each(Iter first, Iter last, Function f);

#include <algorithm>
#include <cassert>
#include <deque>
#include <map>
#include <ranges>
#include <set>
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"

struct for_each_test {
  TEST_CONSTEXPR for_each_test(int c) : count(c) {}

  // for_each functors only have to be move constructible
  for_each_test(const for_each_test&)            = delete;
  for_each_test(for_each_test&&)                 = default;
  for_each_test& operator=(const for_each_test&) = delete;
  for_each_test& operator=(for_each_test&&)      = delete;

  int count;
  TEST_CONSTEXPR_CXX14 void operator()(int& i) {
    ++i;
    ++count;
  }
};

struct Test {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    int sizes[] = {0, 1, 6};
    for (const int size : sizes) {
      int ia[]        = {0, 1, 2, 3, 4, 5};
      for_each_test f = std::for_each(Iter(ia), Iter(ia + size), for_each_test(0));
      assert(f.count == size);
      for (int i = 0; i < size; ++i)
        assert(ia[i] == static_cast<int>(i + 1));
    }
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<int*>(), Test());

#if TEST_STD_VER >= 20
  { // Make sure that the segmented iterator optimization works during constant evaluation
    std::vector<std::vector<int>> vecs = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto v                             = std::views::join(vecs);
    std::for_each(v.begin(), v.end(), [i = 0](int& a) mutable { assert(a == ++i); });
  }
#endif

  return true;
}

struct deque_test {
  std::deque<int>* d_;
  int* i_;

  deque_test(std::deque<int>& d, int& i) : d_(&d), i_(&i) {}

  void operator()(int& v) {
    assert(&(*d_)[*i_] == &v);
    ++*i_;
  }
};

template <class Container, class Converter>
void test_node_container(Converter conv) {
  Container c;
  using value_type = typename Container::value_type;
  for (int i = 0; i != 10; ++i)
    c.insert(conv(i));
  { // Simple check
    int invoke_count = 0;
    std::for_each(c.begin(), c.end(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 10);
  }
  { // Make sure that a start within the container works as expected
    int invoke_count = 1;
    std::for_each(std::next(c.begin()), c.end(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 10);
  }
  { // Make sure that a start within the container works as expected
    int invoke_count = 2;
    std::for_each(std::next(c.begin(), 2), c.end(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 10);
  }
  { // Make sure that an end within the container works as expected
    int invoke_count = 1;
    std::for_each(std::next(c.begin()), std::prev(c.end()), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 9);
  }
  { // Make sure that an empty range works
    int invoke_count = 0;
    std::for_each(c.begin(), c.begin(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 0);
  }
  { // Make sure that a single-element range works
    int invoke_count = 0;
    std::for_each(c.begin(), std::next(c.begin()), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 1);
  }
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  test_node_container<std::set<int> >([](int i) { return i; });
  test_node_container<std::multiset<int> >([](int i) { return i; });
  test_node_container<std::map<int, int> >([](int i) { return std::make_pair(i, i); });
  test_node_container<std::multimap<int, int> >([](int i) { return std::make_pair(i, i); });

  // check that segmented iterators work properly
  int sizes[] = {0, 1, 2, 1023, 1024, 1025, 2047, 2048, 2049};
  for (const int size : sizes) {
    std::deque<int> d(size);
    int index = 0;

    std::for_each(d.begin(), d.end(), deque_test(d, index));
  }

  return 0;
}
