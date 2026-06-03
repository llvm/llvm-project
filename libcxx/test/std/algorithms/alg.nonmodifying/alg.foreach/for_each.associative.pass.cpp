//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Check that the special implementation of std::for_each for the associative container iterators works as expected

// template<InputIterator Iter, class Function>
//   constexpr Function   // constexpr since C++20
//   for_each(Iter first, Iter last, Function f);

#include <algorithm>
#include <cassert>
#include <map>
#include <set>

#include "test_macros.h"

template <class Container, class Converter>
TEST_CONSTEXPR_CXX26 void test_node_container(Converter conv) {
  Container c;
  using value_type = typename Container::value_type;
  for (int i = 0; i != 10; ++i)
    c.insert(conv(i));
  { // Make sure that a start within the container works as expected
    for (int i = 0; i != 10; ++i) {
      int invoke_count = i;
      std::for_each(std::next(c.begin(), i), c.end(), [&c, &invoke_count](const value_type& val) {
        assert(&val == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 10);
    }
  }
  { // Make sure that an end within the container works as expected
    for (int i = 0; i != 10; ++i) {
      int invoke_count = 0;
      std::for_each(c.begin(), std::prev(c.end(), i), [&c, &invoke_count](const value_type& val) {
        assert(&val == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 10 - i);
    }
  }
  {   // Make sure that an empty range works
    { // With an element as the pointee
      int invoke_count = 0;
      std::for_each(c.begin(), c.begin(), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 0);
    }
    { // With no element as the pointee
      int invoke_count = 0;
      std::for_each(c.end(), c.end(), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 0);
    }
  }
  { // Make sure that a single-element range works
    int invoke_count = 0;
    std::for_each(c.begin(), std::next(c.begin()), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 1);
  }
}

TEST_CONSTEXPR_CXX26 bool test() {
  // FIXME: remove when set is made constexpr
  if (!TEST_IS_CONSTANT_EVALUATED)
    test_node_container<std::set<int> >([](int i) { return i; });
  // FIXME: remove when multiset is made constexpr
  if (!TEST_IS_CONSTANT_EVALUATED)
    test_node_container<std::multiset<int> >([](int i) { return i; });
  test_node_container<std::map<int, int> >([](int i) { return std::make_pair(i, i); });
  // FIXME: remove when multimap is made constexpr
  if (!TEST_IS_CONSTANT_EVALUATED)
    test_node_container<std::multimap<int, int> >([](int i) { return std::make_pair(i, i); });

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
