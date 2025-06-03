//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

#include <algorithm>
#include <cassert>
#include <climits>
#include <deque>
#include <initializer_list>
#include <list>
#include <flat_set>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

void test() {
  {
    // deduction guide generated from
    // flat_set(const flat_set&)
    std::flat_set<long> source = {1, 2};
    std::flat_set s(source);
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    // deduction guide generated from
    // flat_set(const flat_set&)
    std::flat_set<short, std::greater<short>> source = {1, 2};
    std::flat_set s{source}; // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    // deduction guide generated from
    // flat_set(const flat_set&, const Allocator&)
    std::flat_set<long, std::greater<long>> source = {1, 2};
    std::flat_set s(source, std::allocator<int>());
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }

  {
    // various overloads that takes a container
    std::deque<int, test_allocator<int>> ks({1, 2, 1, INT_MAX, 3}, test_allocator<int>(0, 42));
    std::deque<int, test_allocator<int>> sorted_ks({1, 2, 3, INT_MAX}, test_allocator<int>(0, 42));
    int expected[] = {1, 2, 3, INT_MAX};
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_set(KeyContainer, Compare = Compare())
      //   -> flat_set<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_set s(ks);

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_set(sorted_unique_t, KeyContainer, Compare = Compare())
      //   -> flat_set<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_set s(std::sorted_unique, sorted_ks);

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Allocator>
      // flat_set(KeyContainer, Allocator)
      //   -> flat_set<typename KeyContainer::value_type,
      //               less<typename KeyContainer::value_type>, KeyContainer>;
      std::flat_set s(ks, test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
    {
      // template<class KeyContainer, class Allocator>
      // flat_set(sorted_unique_t, KeyContainer, Allocator)
      //   -> flat_set<typename KeyContainer::value_type,
      //               less<typename KeyContainer::value_type>, KeyContainer>;
      std::flat_set s(std::sorted_unique, sorted_ks, test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
  }
  {
    // various overloads that takes a container and a comparator
    std::deque<int, test_allocator<int>> ks({1, 2, 1, INT_MAX, 3}, test_allocator<int>(0, 42));
    std::deque<int, test_allocator<int>> sorted_ks({INT_MAX, 3, 2, 1}, test_allocator<int>(0, 42));
    int expected[] = {INT_MAX, 3, 2, 1};
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_set(KeyContainer, Compare = Compare())
      //   -> flat_set<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_set s(ks, std::greater<int>());

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_set(sorted_unique_t, KeyContainer, Compare = Compare())
      //   -> flat_set<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_set s(std::sorted_unique, sorted_ks, std::greater<int>());

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Compare, class Allocator>
      // flat_set(KeyContainer, Compare, Allocator)
      //   -> flat_set<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_set s(ks, std::greater<int>(), test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
    {
      // template<class KeyContainer, class Compare, class Allocator>
      // flat_set(sorted_unique_t, KeyContainer, Compare, Allocator)
      //   -> flat_set<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_set s(std::sorted_unique, sorted_ks, std::greater<int>(), test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
  }
  {
    // overloads that take pair of iterators
    int arr[]               = {1, 2, 1, INT_MAX, 3};
    int sorted_arr[]        = {1, 2, 3, INT_MAX};
    const int arrc[]        = {1, 2, 1, INT_MAX, 3};
    const int sorted_arrc[] = {1, 2, 3, INT_MAX};
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_set(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_set<iter-value-type<InputIterator>, Compare>;
      std::flat_set m(std::begin(arr), std::end(arr));

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // const
      std::flat_set m(std::begin(arrc), std::end(arrc));

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_set(sorted_unique_t, InputIterator, InputIterator, Compare = Compare())
      //   -> flat_set<iter-value-type<InputIterator>, Compare>;
      std::flat_set m(std::sorted_unique, std::begin(sorted_arr), std::end(sorted_arr));

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // const
      std::flat_set m(std::sorted_unique, std::begin(sorted_arrc), std::end(sorted_arrc));

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // from flat_set::iterator
      std::flat_set<int> mo;
      std::flat_set m(mo.begin(), mo.end());
      ASSERT_SAME_TYPE(decltype(m), decltype(mo));
    }
    {
      // from flat_set::const_iterator
      std::flat_set<int> mo;
      std::flat_set m(mo.cbegin(), mo.cend());
      ASSERT_SAME_TYPE(decltype(m), decltype(mo));
    }
    {
      // This does not deduce to flat_set(InputIterator, InputIterator)
      // But deduces to flat_set(initializer_list<int*>)
      std::flat_set s = {arr, arr + 3};
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int*>);
      assert(s.size() == 2);
    }
    {
      // This deduces to flat_set(sorted_unique_t, InputIterator, InputIterator)
      std::flat_set s{std::sorted_unique, sorted_arr, sorted_arr + 3};
      static_assert(std::is_same_v<decltype(s), std::flat_set<int>>);
      assert(s.size() == 3);
    }
  }
  {
    // overloads that take pair of iterators and comparator
    int arr[]               = {1, 2, 1, INT_MAX, 3};
    int sorted_arr[]        = {INT_MAX, 3, 2, 1};
    const int arrc[]        = {1, 2, 1, INT_MAX, 3};
    const int sorted_arrc[] = {INT_MAX, 3, 2, 1};
    using C                 = std::greater<long>;
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_set(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_set<iter-value-type<InputIterator>, Compare>;
      std::flat_set m(std::begin(arr), std::end(arr), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // const
      std::flat_set m(std::begin(arrc), std::end(arrc), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_set(sorted_unique_t, InputIterator, InputIterator, Compare = Compare())
      //   -> flat_set<iter-value-type<InputIterator>, Compare>;
      std::flat_set m(std::sorted_unique, std::begin(sorted_arr), std::end(sorted_arr), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // const
      std::flat_set m(std::sorted_unique, std::begin(sorted_arrc), std::end(sorted_arrc), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // from flat_set::iterator
      std::flat_set<int> mo;
      std::flat_set m(mo.begin(), mo.end(), C());
      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
    }
    {
      // from flat_set::const_iterator
      std::flat_set<int> mo;
      std::flat_set m(mo.cbegin(), mo.cend(), C());
      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
    }
  }
  {
    // overloads that take an initializer_list
    const int sorted_arr[] = {1, 2, 3, INT_MAX};
    {
      // template<class Key, class Compare = less<Key>>
      // flat_set(initializer_list<Key>, Compare = Compare())
      //   -> flat_set<Key, Compare>;
      std::flat_set m{1, 2, 1, INT_MAX, 3};

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class Key, class Compare = less<Key>>
      // flat_set(sorted_unique_t, initializer_list<Key>, Compare = Compare())
      //   -> flat_set<Key, Compare>;
      std::flat_set m(std::sorted_unique, {1, 2, 3, INT_MAX});

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // one argument/element of an int -> should be treated as an initializer_list<int>
      std::flat_set s = {1};
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int>);
      assert(s.size() == 1);
    }
    {
      // two of the flat_sets -> should be treated as an initializer_list<flat_set<int>>
      using M = std::flat_set<int>;
      M m;
      std::flat_set s{m, m}; // flat_set(initializer_list<M>)
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<M>);
      assert(s.size() == 1);
    }
  }
  {
    // overloads that take an initializer_list and a comparator
    const int sorted_arr[] = {INT_MAX, 3, 2, 1};
    using C                = std::greater<long>;
    {
      // template<class Key, class Compare = less<Key>>
      // flat_set(initializer_list<Key>, Compare = Compare())
      //   -> flat_set<Key, Compare>;
      std::flat_set m({1, 2, 1, INT_MAX, 3}, C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class Key, class Compare = less<Key>>
      // flat_set(sorted_unique_t, initializer_list<Key>, Compare = Compare())
      //   -> flat_set<Key, Compare>;
      std::flat_set m(std::sorted_unique, {INT_MAX, 3, 2, 1}, C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_set<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
  }
  {
    // from_range without comparator
    std::list<int> r     = {1, 2, 1, INT_MAX, 3};
    const int expected[] = {1, 2, 3, INT_MAX};
    {
      // template<ranges::input_range R, class Compare = less<ranges::range_value_t<R>>,
      // class Allocator = allocator<ranges::range_value_t<R>>>
      // flat_set(from_range_t, R&&, Compare = Compare(), Allocator = Allocator())
      //  -> flat_set<ranges::range_value_t<R>, Compare,
      //              vector<ranges::range_value_t<R>,
      //                     alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_set s(std::from_range, r);
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::less<int>>);
      assert(std::ranges::equal(s, expected));
    }
    {
      // template<ranges::input_range R, class Allocator>
      // flat_set(from_range_t, R&&, Allocator)
      //   -> flat_set<ranges::range_value_t<R>, less<ranges::range_value_t<R>>,
      //               vector<ranges::range_value_t<R>,
      //                      alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_set s(std::from_range, r, test_allocator<long>(0, 42));
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::less<int>, std::vector<int, test_allocator<int>>>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
  }
  {
    // from_range with comparator
    std::list<int> r     = {1, 2, 1, INT_MAX, 3};
    const int expected[] = {INT_MAX, 3, 2, 1};
    {
      // template<ranges::input_range R, class Compare = less<ranges::range_value_t<R>>,
      // class Allocator = allocator<ranges::range_value_t<R>>>
      // flat_set(from_range_t, R&&, Compare = Compare(), Allocator = Allocator())
      //  -> flat_set<ranges::range_value_t<R>, Compare,
      //              vector<ranges::range_value_t<R>,
      //                     alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_set s(std::from_range, r, std::greater<int>());
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::greater<int>>);
      assert(std::ranges::equal(s, expected));
    }
    {
      // template<ranges::input_range R, class Allocator>
      // flat_set(from_range_t, R&&, Allocator)
      //   -> flat_set<ranges::range_value_t<R>, less<ranges::range_value_t<R>>,
      //               vector<ranges::range_value_t<R>,
      //                      alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_set s(std::from_range, r, std::greater<int>(), test_allocator<long>(0, 42));
      ASSERT_SAME_TYPE(decltype(s), std::flat_set<int, std::greater<int>, std::vector<int, test_allocator<int>>>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
  }

  AssociativeContainerDeductionGuidesSfinaeAway<std::flat_set, std::flat_set<int>>();
}

int main(int, char**) {
  test();

  return 0;
}
