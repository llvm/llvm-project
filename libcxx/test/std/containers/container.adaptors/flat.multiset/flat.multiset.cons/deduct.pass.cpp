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
    // Deduction guide generated from
    // flat_multiset(const flat_multiset&)
    std::flat_multiset<long> source = {1, 2, 2};
    std::flat_multiset s(source);
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    // Deduction guide generated from
    // flat_multiset(const flat_multiset&)
    // braces instead of parens
    std::flat_multiset<short, std::greater<short>> source = {1, 2, 2};
    std::flat_multiset s{source};
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    // Deduction guide generated from
    // flat_set(const flat_set&, const Allocator&)
    std::flat_multiset<long, std::greater<long>> source = {1, 2, 2};
    std::flat_multiset s(source, std::allocator<int>());
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    std::deque<int, test_allocator<int>> ks({1, 2, 1, INT_MAX, 3}, test_allocator<int>(0, 42));
    std::deque<int, test_allocator<int>> sorted_ks({1, 1, 2, 3, INT_MAX}, test_allocator<int>(0, 42));
    int expected[] = {1, 1, 2, 3, INT_MAX};
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_multiset(KeyContainer, Compare = Compare())
      //   -> flat_multiset<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_multiset s(ks);

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_multiset(sorted_equivalent_t, KeyContainer, Compare = Compare())
      //   -> flat_multiset<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_multiset s(std::sorted_equivalent, sorted_ks);

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Allocator>
      // flat_multiset(KeyContainer, Allocator)
      //   -> flat_multiset<typename KeyContainer::value_type,
      //                    less<typename KeyContainer::value_type>, KeyContainer>;
      std::flat_multiset s(ks, test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
    {
      // template<class KeyContainer, class Allocator>
      // flat_multiset(sorted_equivalent_t, KeyContainer, Allocator)
      //   -> flat_multiset<typename KeyContainer::value_type,
      //                    less<typename KeyContainer::value_type>, KeyContainer>;
      std::flat_multiset s(std::sorted_equivalent, sorted_ks, test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
  }
  {
    std::deque<int, test_allocator<int>> ks({1, 2, 1, INT_MAX, 3}, test_allocator<int>(0, 42));
    std::deque<int, test_allocator<int>> sorted_ks({INT_MAX, 3, 2, 1, 1}, test_allocator<int>(0, 42));
    int expected[] = {INT_MAX, 3, 2, 1, 1};
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_multiset(KeyContainer, Compare = Compare())
      //   -> flat_multiset<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_multiset s(ks, std::greater<int>());

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Compare = less<typename KeyContainer::value_type>>
      // flat_multiset(sorted_equivalent_t, KeyContainer, Compare = Compare())
      //   -> flat_multiset<typename KeyContainer::value_type, Compare, KeyContainer>;

      std::flat_multiset s(std::sorted_equivalent, sorted_ks, std::greater<int>());

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
    {
      // template<class KeyContainer, class Compare, class Allocator>
      // flat_multiset(KeyContainer, Compare, Allocator)
      //   -> flat_multiset<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_multiset s(ks, std::greater<int>(), test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
    {
      // template<class KeyContainer, class Compare, class Allocator>
      // flat_multiset(sorted_equivalent_t, KeyContainer, Compare, Allocator)
      //   -> flat_multiset<typename KeyContainer::value_type, Compare, KeyContainer>;
      std::flat_multiset s(std::sorted_equivalent, sorted_ks, std::greater<int>(), test_allocator<long>(0, 44));

      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 44);
    }
  }

  {
    int arr[]               = {1, 2, 1, INT_MAX, 3};
    int sorted_arr[]        = {1, 1, 2, 3, INT_MAX};
    const int arrc[]        = {1, 2, 1, INT_MAX, 3};
    const int sorted_arrc[] = {1, 1, 2, 3, INT_MAX};
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      std::flat_multiset m(std::begin(arr), std::end(arr));

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // const
      std::flat_multiset m(std::begin(arrc), std::end(arrc));

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arr), std::end(sorted_arr));

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // const
      std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arrc), std::end(sorted_arrc));

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // flat_multiset iterator
      std::flat_multiset<int> mo;
      std::flat_multiset m(mo.begin(), mo.end());
      ASSERT_SAME_TYPE(decltype(m), decltype(mo));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // flat_multiset const_iterator
      std::flat_multiset<int> mo;
      std::flat_multiset m(mo.cbegin(), mo.cend());
      ASSERT_SAME_TYPE(decltype(m), decltype(mo));
    }
    {
      // This does not deduce to flat_multiset(InputIterator, InputIterator)
      // But deduces to flat_multiset(initializer_list<int*>)
      int source[3]        = {1, 2, 3};
      std::flat_multiset s = {source, source + 3};
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int*>);
      assert(s.size() == 2);
    }
    {
      // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator)
      // braces
      int source[3] = {1, 2, 3};
      std::flat_multiset s{std::sorted_equivalent, source, source + 3};
      static_assert(std::is_same_v<decltype(s), std::flat_multiset<int>>);
      assert(s.size() == 3);
    }
  }

  {
    int arr[]               = {1, 2, 1, INT_MAX, 3};
    int sorted_arr[]        = {INT_MAX, 3, 2, 1, 1};
    const int arrc[]        = {1, 2, 1, INT_MAX, 3};
    const int sorted_arrc[] = {INT_MAX, 3, 2, 1, 1};
    using C                 = std::greater<long>;
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      std::flat_multiset m(std::begin(arr), std::end(arr), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // const
      std::flat_multiset m(std::begin(arrc), std::end(arrc), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arr), std::end(sorted_arr), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // const
      std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arrc), std::end(sorted_arrc), C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // flat_multiset iterator
      std::flat_multiset<int> mo;
      std::flat_multiset m(mo.begin(), mo.end(), C());
      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    }
    {
      // template<class InputIterator, class Compare = less<iter-value-type<InputIterator>>>
      // flat_multiset(InputIterator, InputIterator, Compare = Compare())
      //   -> flat_multiset<iter-value-type<InputIterator>, Compare>;
      // flat_multiset const_iterator
      std::flat_multiset<int> mo;
      std::flat_multiset m(mo.cbegin(), mo.cend(), C());
      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    }
  }
  {
    const int sorted_arr[] = {1, 1, 2, 3, INT_MAX};
    {
      // template<class Key, class Compare = less<Key>>
      // flat_multiset(initializer_list<Key>, Compare = Compare())
      //   -> flat_multiset<Key, Compare>;
      std::flat_multiset m{1, 2, 1, INT_MAX, 3};

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class Key, class Compare = less<Key>>
      // flat_multiset(sorted_equivalent_t, initializer_list<Key>, Compare = Compare())
      //     -> flat_multiset<Key, Compare>;
      std::flat_multiset m(std::sorted_equivalent, {1, 1, 2, 3, INT_MAX});

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // One element with brace was treated as initializer_list
      std::flat_multiset s = {1};
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int>);
      assert(s.size() == 1);
    }
    {
      // Two elements with brace was treated as initializer_list
      using M = std::flat_multiset<int>;
      M m;
      std::flat_multiset s{m, m}; // flat_multiset(initializer_list<M>)
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<M>);
      assert(s.size() == 2);
    }
  }
  {
    const int sorted_arr[] = {INT_MAX, 3, 2, 1, 1};
    using C                = std::greater<long>;
    {
      // template<class Key, class Compare = less<Key>>
      // flat_multiset(initializer_list<Key>, Compare = Compare())
      //   -> flat_multiset<Key, Compare>;
      std::flat_multiset m({1, 2, 1, INT_MAX, 3}, C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
    {
      // template<class Key, class Compare = less<Key>>
      // flat_multiset(sorted_equivalent_t, initializer_list<Key>, Compare = Compare())
      //     -> flat_multiset<Key, Compare>;
      std::flat_multiset m(std::sorted_equivalent, {INT_MAX, 3, 2, 1, 1}, C());

      ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
      assert(std::ranges::equal(m, sorted_arr));
    }
  }
  {
    std::list<int> r     = {1, 2, 1, INT_MAX, 3};
    const int expected[] = {1, 1, 2, 3, INT_MAX};
    {
      // template<ranges::input_range R, class Compare = less<ranges::range_value_t<R>>,
      //     class Allocator = allocator<ranges::range_value_t<R>>>
      // flat_multiset(from_range_t, R&&, Compare = Compare(), Allocator = Allocator())
      // -> flat_multiset<ranges::range_value_t<R>, Compare,
      //                 vector<ranges::range_value_t<R>,
      //                        alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_multiset s(std::from_range, r);
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>>);
      assert(std::ranges::equal(s, expected));
    }
    {
      // template<ranges::input_range R, class Allocator>
      // flat_multiset(from_range_t, R&&, Allocator)
      //   -> flat_multiset<ranges::range_value_t<R>, less<ranges::range_value_t<R>>,
      //                    vector<ranges::range_value_t<R>,
      //                           alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_multiset s(std::from_range, r, test_allocator<long>(0, 42));
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, std::vector<int, test_allocator<int>>>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
  }

  {
    // with comparator
    std::list<int> r     = {1, 2, 1, INT_MAX, 3};
    const int expected[] = {INT_MAX, 3, 2, 1, 1};
    {
      // template<ranges::input_range R, class Compare = less<ranges::range_value_t<R>>,
      //     class Allocator = allocator<ranges::range_value_t<R>>>
      // flat_multiset(from_range_t, R&&, Compare = Compare(), Allocator = Allocator())
      // -> flat_multiset<ranges::range_value_t<R>, Compare,
      //                 vector<ranges::range_value_t<R>,
      //                        alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_multiset s(std::from_range, r, std::greater<int>());
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>>);
      assert(std::ranges::equal(s, expected));
    }
    {
      // template<ranges::input_range R, class Allocator>
      // flat_multiset(from_range_t, R&&, Allocator)
      //   -> flat_multiset<ranges::range_value_t<R>, less<ranges::range_value_t<R>>,
      //                    vector<ranges::range_value_t<R>,
      //                           alloc-rebind<Allocator, ranges::range_value_t<R>>>>;
      std::flat_multiset s(std::from_range, r, std::greater<int>(), test_allocator<long>(0, 42));
      ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, std::vector<int, test_allocator<int>>>);
      assert(std::ranges::equal(s, expected));
      assert(std::move(s).extract().get_allocator().get_id() == 42);
    }
  }

  AssociativeContainerDeductionGuidesSfinaeAway<std::flat_multiset, std::flat_multiset<int>>();
}

int main(int, char**) {
  test();

  return 0;
}
