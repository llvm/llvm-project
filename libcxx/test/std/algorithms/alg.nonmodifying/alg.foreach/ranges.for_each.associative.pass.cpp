//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>
// UNSUPPORTED: c++03, c++11, c++14, c++17

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
  { // Check that an empty container works, taking the whole range
    Container c;
    int invoke_count = 0;
    std::ranges::for_each(c, [&c, &invoke_count](const value_type& i) {
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
  { // Check that a single-element container works, taking the whole range
    Container c;
    c.insert(conv(0));
    int invoke_count = 0;
    std::ranges::for_each(c, [&c, &invoke_count](const value_type& i) {
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
  { // Check that a two-element container works, taking the whole range
    Container c;
    c.insert(conv(0));
    c.insert(conv(1));
    int invoke_count = 0;
    std::ranges::for_each(c, [&c, &invoke_count](const value_type& i) {
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

template <template <class> class Container>
void test_invoke_set_like() {
  { // check that std::invoke is used
    struct T {
      mutable int i = 3;

      void zero() const { i = 0; }
    };

    class S {
      int val_;

    public:
      S(int val) : val_(val) {}

      T j;

      bool operator<(const S& rhs) const { return val_ < rhs.val_; }
    };

    { // Iterator overload
      Container<S> a = {S{2}, S{4}, S{6}};
      std::ranges::for_each(a.begin(), a.end(), &T::zero, &S::j);
      assert(a.find(2)->j.i == 0);
      assert(a.find(4)->j.i == 0);
      assert(a.find(6)->j.i == 0);
    }
    { // Range overload
      Container<S> a = {S{2}, S{4}, S{6}};
      std::ranges::for_each(a, &T::zero, &S::j);
      assert(a.find(2)->j.i == 0);
      assert(a.find(4)->j.i == 0);
      assert(a.find(6)->j.i == 0);
    }
  }
}

template <template <class, class> class Container>
void test_invoke_map_like() {
  { // check that std::invoke is used
    struct S {
      int i;

      void zero() { i = 0; }
    };

    { // Iterator overload
      Container<int, S> a = {{1, S{2}}, {3, S{4}}, {5, S{6}}};
      std::ranges::for_each(a.begin(), a.end(), &S::zero, &std::pair<const int, S>::second);
      assert(a.find(1)->second.i == 0);
      assert(a.find(3)->second.i == 0);
      assert(a.find(5)->second.i == 0);
    }
    { // Range overload
      Container<int, S> a = {{1, S{2}}, {3, S{4}}, {5, S{6}}};
      std::ranges::for_each(a, &S::zero, &std::pair<const int, S>::second);
      assert(a.find(1)->second.i == 0);
      assert(a.find(3)->second.i == 0);
      assert(a.find(5)->second.i == 0);
    }
  }
}

int main(int, char**) {
  test_node_container<std::set<int> >([](int i) { return i; });
  test_node_container<std::multiset<int> >([](int i) { return i; });
  test_node_container<std::map<int, int> >([](int i) { return std::make_pair(i, i); });
  test_node_container<std::multimap<int, int> >([](int i) { return std::make_pair(i, i); });

  test_invoke_set_like<std::set>();
  test_invoke_set_like<std::multiset>();

  test_invoke_map_like<std::map>();
  test_invoke_map_like<std::multimap>();

  return 0;
}
