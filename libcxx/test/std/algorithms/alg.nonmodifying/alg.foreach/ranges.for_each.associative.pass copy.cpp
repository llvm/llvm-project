//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Check that the special implementation of ranges::for_each for the associative container iterators works as expected

// template<input_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirectly_unary_invocable<projected<I, Proj>> Fun>
//   constexpr ranges::for_each_result<I, Fun>
//     ranges::for_each(I first, S last, Fun f, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirectly_unary_invocable<projected<iterator_t<R>, Proj>> Fun>
//   constexpr ranges::for_each_result<borrowed_iterator_t<R>, Fun>
//     ranges::for_each(R&& r, Fun f, Proj proj = {});

#include <algorithm>
#include <cassert>
#include <map>
#include <set>

template <class Container, class Converter>
void test_node_container(Converter conv) {
  using value_type = typename Container::value_type;

  { // Check that an empty container works
    Container c;
    int invoke_count = 0;
    std::ranges::for_each(c.begin(), c.end(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 0);
  }
  { // Check that a single-element container works
    Container c;
    c.insert(conv(0));
    int invoke_count = 0;
    std::ranges::for_each(c.begin(), c.end(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 1);
  }
  { // Check that a two-element container works
    Container c;
    c.insert(conv(0));
    c.insert(conv(1));
    int invoke_count = 0;
    std::ranges::for_each(c.begin(), c.end(), [&c, &invoke_count](const value_type& i) {
      assert(&i == &*std::next(c.begin(), invoke_count++));
    });
    assert(invoke_count == 2);
  }

  Container c;
  for (int i = 0; i != 10; ++i)
    c.insert(conv(i));

  { // Simple check
    {
      int invoke_count = 0;
      std::ranges::for_each(c.begin(), c.end(), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 10);
    }
    {
      int invoke_count = 0;
      std::ranges::for_each(c, [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 10);
    }
  }
  { // Make sure that a start within the container works as expected
    {
      int invoke_count = 1;
      std::ranges::for_each(std::next(c.begin()), c.end(), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 10);
    }
    {
      int invoke_count = 1;
      std::ranges::for_each(
          std::ranges::subrange(std::next(c.begin()), c.end()),
          [&c, &invoke_count](const value_type& i) { assert(&i == &*std::next(c.begin(), invoke_count++)); });
      assert(invoke_count == 10);
    }
  }
  { // Make sure that a start within the container works as expected
    {
      int invoke_count = 2;
      std::ranges::for_each(std::next(c.begin(), 2), c.end(), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 10);
    }
    {
      int invoke_count = 2;
      std::ranges::for_each(
          std::ranges::subrange(std::next(c.begin(), 2), c.end()),
          [&c, &invoke_count](const value_type& i) { assert(&i == &*std::next(c.begin(), invoke_count++)); });
      assert(invoke_count == 10);
    }
  }
  { // Make sure that an end within the container works as expected
    {
      int invoke_count = 1;
      std::ranges::for_each(std::next(c.begin()), std::prev(c.end()), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 9);
    }
    {
      int invoke_count = 1;
      std::ranges::for_each(
          std::ranges::subrange(std::next(c.begin()), std::prev(c.end())),
          [&c, &invoke_count](const value_type& i) { assert(&i == &*std::next(c.begin(), invoke_count++)); });
      assert(invoke_count == 9);
    }
  }
  { // Make sure that an empty range works
    {
      int invoke_count = 0;
      std::ranges::for_each(c.begin(), c.begin(), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 0);
    }
    {
      int invoke_count = 0;
      std::ranges::for_each(std::ranges::subrange(c.begin(), c.begin()), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 0);
    }
  }
  { // Make sure that a single-element range works
    {
      int invoke_count = 0;
      std::ranges::for_each(c.begin(), std::next(c.begin()), [&c, &invoke_count](const value_type& i) {
        assert(&i == &*std::next(c.begin(), invoke_count++));
      });
      assert(invoke_count == 1);
    }
    {
      int invoke_count = 0;
      std::ranges::for_each(
          std::ranges::subrange(c.begin(), std::next(c.begin())),
          [&c, &invoke_count](const value_type& i) { assert(&i == &*std::next(c.begin(), invoke_count++)); });
      assert(invoke_count == 1);
    }
  }
}

int main(int, char**) {
  test_node_container<std::set<int> >([](int i) { return i; });
  test_node_container<std::multiset<int> >([](int i) { return i; });
  test_node_container<std::map<int, int> >([](int i) { return std::make_pair(i, i); });
  test_node_container<std::multimap<int, int> >([](int i) { return std::make_pair(i, i); });

  return 0;
}
