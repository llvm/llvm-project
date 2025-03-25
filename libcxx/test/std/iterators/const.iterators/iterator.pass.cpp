//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// std::basic_const_iterator

#include <iterator>
#include <list>
#include <memory>
#include <ranges>
#include <vector>
#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

template <class T>
concept has_iterator_category = requires { typename T::iterator_category; };

template <class It>
constexpr bool check_category_and_concept() {
  using ConstIt = std::basic_const_iterator<It>;

  if constexpr (std::contiguous_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::contiguous_iterator_tag);
    static_assert(std::contiguous_iterator<ConstIt>);
  } else if constexpr (std::random_access_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::random_access_iterator_tag);
    static_assert(!std::contiguous_iterator<ConstIt>);
    static_assert(std::random_access_iterator<ConstIt>);
  } else if constexpr (std::bidirectional_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::bidirectional_iterator_tag);
    static_assert(!std::random_access_iterator<ConstIt>);
    static_assert(std::bidirectional_iterator<ConstIt>);
  } else if constexpr (std::forward_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::forward_iterator_tag);
    static_assert(!std::bidirectional_iterator<ConstIt>);
    static_assert(std::forward_iterator<ConstIt>);
  } else {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_concept, std::input_iterator_tag);
    static_assert(!std::forward_iterator<ConstIt>);
    static_assert(std::input_iterator<ConstIt>);
  }

  if constexpr (std::forward_iterator<It>) {
    ASSERT_SAME_TYPE(typename ConstIt::iterator_category, typename std::iterator_traits<It>::iterator_category);
  } else {
    static_assert(!has_iterator_category<ConstIt>);
  }

  return true;
}

constexpr bool test_p2836r1() {
  auto f = [](std::vector<int>::const_iterator) {};

  auto v = std::vector<int>();
  {
    auto i1 = std::ranges::cbegin(v);
    f(i1);
  }

  auto t = v | std::views::take_while([](int const x) { return x < 100; });
  {
    auto i2 = std::ranges::cbegin(t);
    f(i2);
  }

  return true;
}

struct S {
  int x;
};

template <class It>
constexpr void test_basic_operations() {
  S arr[10]{};

  std::basic_const_iterator<It> first = It{arr};
  std::basic_const_iterator<It> last  = It{arr + 10};

  ASSERT_NOEXCEPT(first.base());
  assert(first.base() == It{arr});
  assert(last.base() == It{arr + 10});

  static_assert(noexcept(iter_move(first)) == noexcept(std::ranges::iter_move(first.base())));
  static_assert(noexcept(std::ranges::iter_move(first)) == noexcept(std::ranges::iter_move(first.base())));

  for (auto it = first; it != last; ++it) {
    (void)*it;
    (void)it->x;
    (void)iter_move(it);

    std::same_as<decltype(it)> auto& it_ref = ++it;
    assert(std::addressof(it_ref) == std::addressof(it));
  }
  static_assert(!std::is_invocable_v<decltype(std::ranges::iter_swap), decltype(first), decltype(first)>);

  if constexpr (std::bidirectional_iterator<It>) {
    {
      std::same_as<decltype(first)> auto& it_ref = ++first;
      assert(std::addressof(it_ref) == std::addressof(first));
      assert(it_ref == It{arr + 1});
    }
    {
      std::same_as<decltype(first)> auto& it_ref = --first;
      assert(std::addressof(it_ref) == std::addressof(first));
      assert(--first == It{arr + 0});
    }
    assert(first++ == It{arr + 0});
    assert(first-- == It{arr + 1});
  }

  if constexpr (std::random_access_iterator<It>) {
    assert(first + 3 == It{arr + 3});
    assert(last - 1 == It{arr + 9});

    {
      std::same_as<decltype(first)> auto& it_ref = first += 3;
      assert(std::addressof(it_ref) == std::addressof(first));
      assert(first == It{arr + 3});
    }
    {
      std::same_as<decltype(first)> auto& it_ref = first -= 2;
      assert(std::addressof(it_ref) == std::addressof(first));
      assert(first == It{arr + 1});
    }
    --first;

    assert(first < last);
    assert(last > first);
    assert(first <= last);
    assert(last >= first);

    assert(first < It{arr + 1});
    assert(It{arr + 1} > first);
    assert(first <= It{arr + 1});
    assert(It{arr + 1} >= first);

    if constexpr (std::three_way_comparable<It>) {
      assert((first <=> last) < 0);
      assert((first <=> It{arr + 1}) < 0);
      assert((It{arr + 1} <=> first) > 0);
    }
  }
}

constexpr bool test_basic_operations() {
  test_basic_operations<S*>();
  test_basic_operations<cpp17_input_iterator<S*>>();
  test_basic_operations<forward_iterator<S*>>();
  test_basic_operations<bidirectional_iterator<S*>>();
  test_basic_operations<random_access_iterator<S*>>();

  return true;
}

int main() {
  types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class It>() {
    using ConstIt = std::basic_const_iterator<It>;
    ASSERT_SAME_TYPE(typename ConstIt::value_type, int);
    static_assert(check_category_and_concept<It>());

    ASSERT_SAME_TYPE(std::iter_reference_t<ConstIt>, const int&);
    ASSERT_SAME_TYPE(std::iter_rvalue_reference_t<ConstIt>, const int&&);
  });

  test_p2836r1();
  static_assert(test_p2836r1());

  test_basic_operations();
  static_assert(test_basic_operations());

  return 0;
}
