//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <set>

// template<class InputIterator,
//          class Compare = less<iter-value-type<InputIterator>>,
//          class Allocator = allocator<iter-value-type<InputIterator>>>
// multiset(InputIterator, InputIterator,
//          Compare = Compare(), Allocator = Allocator())
//   -> multiset<iter-value-type<InputIterator>, Compare, Allocator>;
// template<class Key, class Compare = less<Key>,
//          class Allocator = allocator<Key>>
// multiset(initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
//   -> multiset<Key, Compare, Allocator>;
// template<class InputIterator, class Allocator>
// multiset(InputIterator, InputIterator, Allocator)
//   -> multiset<iter-value-type<InputIterator>,
//               less<iter-value-type<InputIterator>>, Allocator>;
// template<class Key, class Allocator>
// multiset(initializer_list<Key>, Allocator)
//   -> multiset<Key, less<Key>, Allocator>;
//
// template<ranges::input_range R, class Compare = less<ranges::range_value_t<R>>,
//          class Allocator = allocator<ranges::range_value_t<R>>>
//   multiset(from_range_t, R&&, Compare = Compare(), Allocator = Allocator())
//     -> multiset<ranges::range_value_t<R>, Compare, Allocator>;
//
// template<ranges::input_range R, class Allocator>
//   multiset(from_range_t, R&&, Allocator)
//     -> multiset<ranges::range_value_t<R>, less<ranges::range_value_t<R>>, Allocator>;

#include <algorithm> // std::equal
#include <array>
#include <cassert>
#include <climits> // INT_MAX
#include <functional>
#include <set>
#include <type_traits>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

int main(int, char **) {
  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::multiset s(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(s), std::multiset<int>);
    const int expected_s[] = { 1, 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::multiset s(std::begin(arr), std::end(arr), std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::multiset<int, std::greater<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::multiset s(std::begin(arr), std::end(arr), std::greater<int>(),
                    test_allocator<int>(0, 42));

    ASSERT_SAME_TYPE(
        decltype(s),
        std::multiset<int, std::greater<int>, test_allocator<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 42);
  }

  {
    std::multiset<long> source;
    std::multiset s(source);
    ASSERT_SAME_TYPE(decltype(s), std::multiset<long>);
    assert(s.size() == 0);
  }

  {
    std::multiset<long> source;
    std::multiset s{ source };  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), std::multiset<long>);
    assert(s.size() == 0);
  }

  {
    std::multiset<long> source;
    std::multiset s(source, std::multiset<long>::allocator_type());
    ASSERT_SAME_TYPE(decltype(s), std::multiset<long>);
    assert(s.size() == 0);
  }

  {
    std::multiset s{ 1, 2, 1, INT_MAX, 3 };

    ASSERT_SAME_TYPE(decltype(s), std::multiset<int>);
    const int expected_s[] = { 1, 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    std::multiset s({ 1, 2, 1, INT_MAX, 3 }, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::multiset<int, std::greater<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    std::multiset s({ 1, 2, 1, INT_MAX, 3 }, std::greater<int>(),
                    test_allocator<int>(0, 43));

    ASSERT_SAME_TYPE(
        decltype(s),
        std::multiset<int, std::greater<int>, test_allocator<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 43);
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::multiset s(std::begin(arr), std::end(arr), test_allocator<int>(0, 44));

    ASSERT_SAME_TYPE(decltype(s),
                     std::multiset<int, std::less<int>, test_allocator<int> >);
    const int expected_s[] = { 1, 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 44);
  }

  {
    std::multiset s({ 1, 2, 1, INT_MAX, 3 }, test_allocator<int>(0, 45));

    ASSERT_SAME_TYPE(decltype(s),
                     std::multiset<int, std::less<int>, test_allocator<int> >);
    const int expected_s[] = { 1, 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 45);
  }

  {
    NotAnAllocator a;
    std::multiset s{ a }; // multiset(initializer_list<NotAnAllocator>)
    ASSERT_SAME_TYPE(decltype(s), std::multiset<NotAnAllocator>);
    assert(s.size() == 1);
  }

  {
    std::multiset<long> source;
    std::multiset s{ source, source }; // multiset(initializer_list<multiset<long>>)
    ASSERT_SAME_TYPE(decltype(s), std::multiset<std::multiset<long> >);
    assert(s.size() == 2);
  }

  {
    NotAnAllocator a;
    std::multiset s{ a, a }; // multiset(initializer_list<NotAnAllocator>)
    ASSERT_SAME_TYPE(decltype(s), std::multiset<NotAnAllocator>);
    assert(s.size() == 2);
  }

  {
    int source[3] = { 3, 4, 5 };
    std::multiset s(source, source + 3); // multiset(InputIterator, InputIterator)
    ASSERT_SAME_TYPE(decltype(s), std::multiset<int>);
    assert(s.size() == 3);
  }

  {
    int source[3] = { 3, 4, 5 };
    std::multiset s{ source, source + 3 }; // multiset(initializer_list<int*>)
    ASSERT_SAME_TYPE(decltype(s), std::multiset<int *>);
    assert(s.size() == 2);
  }

#if TEST_STD_VER >= 23
    {
      using Range = std::array<int, 0>;
      using Comp = std::greater<int>;
      using DefaultComp = std::less<int>;
      using Alloc = test_allocator<int>;

      { // (from_range, range)
        std::multiset c(std::from_range, Range());
        static_assert(std::is_same_v<decltype(c), std::multiset<int>>);
      }

      { // (from_range, range, comp)
        std::multiset c(std::from_range, Range(), Comp());
        static_assert(std::is_same_v<decltype(c), std::multiset<int, Comp>>);
      }

      { // (from_range, range, comp, alloc)
        std::multiset c(std::from_range, Range(), Comp(), Alloc());
        static_assert(std::is_same_v<decltype(c), std::multiset<int, Comp, Alloc>>);
      }

      { // (from_range, range, alloc)
        std::multiset c(std::from_range, Range(), Alloc());
        static_assert(std::is_same_v<decltype(c), std::multiset<int, DefaultComp, Alloc>>);
      }
    }
#endif

  AssociativeContainerDeductionGuidesSfinaeAway<std::multiset, std::multiset<int>>();

  return 0;
}
