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
// set(InputIterator, InputIterator,
//     Compare = Compare(), Allocator = Allocator())
//   -> set<iter-value-type<InputIterator>, Compare, Allocator>;
// template<class Key, class Compare = less<Key>,
//          class Allocator = allocator<Key>>
// set(initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
//   -> set<Key, Compare, Allocator>;
// template<class InputIterator, class Allocator>
// set(InputIterator, InputIterator, Allocator)
//   -> set<iter-value-type<InputIterator>,
//          less<iter-value-type<InputIterator>>, Allocator>;
// template<class Key, class Allocator>
// set(initializer_list<Key>, Allocator)
//   -> set<Key, less<Key>, Allocator>;

#include <algorithm> // std::equal
#include <cassert>
#include <climits> // INT_MAX
#include <functional>
#include <set>
#include <type_traits>

#include "test_allocator.h"

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

int main(int, char **) {
  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::set s(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(s), std::set<int>);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::set s(std::begin(arr), std::end(arr), std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::set<int, std::greater<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::set s(std::begin(arr), std::end(arr), std::greater<int>(),
               test_allocator<int>(0, 42));

    ASSERT_SAME_TYPE(decltype(s),
                     std::set<int, std::greater<int>, test_allocator<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 42);
  }

  {
    std::set<long> source;
    std::set s(source);
    ASSERT_SAME_TYPE(decltype(s), std::set<long>);
    assert(s.size() == 0);
  }

  {
    std::set<long> source;
    std::set s{ source };  // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), std::set<long>);
    assert(s.size() == 0);
  }

  {
    std::set<long> source;
    std::set s(source, std::set<long>::allocator_type());
    ASSERT_SAME_TYPE(decltype(s), std::set<long>);
    assert(s.size() == 0);
  }

  {
    std::set s{ 1, 2, 1, INT_MAX, 3 };

    ASSERT_SAME_TYPE(decltype(s), std::set<int>);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    std::set s({ 1, 2, 1, INT_MAX, 3 }, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::set<int, std::greater<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
  }

  {
    std::set s({ 1, 2, 1, INT_MAX, 3 }, std::greater<int>(),
               test_allocator<int>(0, 43));

    ASSERT_SAME_TYPE(decltype(s),
                     std::set<int, std::greater<int>, test_allocator<int> >);
    const int expected_s[] = { INT_MAX, 3, 2, 1 };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 43);
  }

  {
    const int arr[] = { 1, 2, 1, INT_MAX, 3 };
    std::set s(std::begin(arr), std::end(arr), test_allocator<int>(0, 44));

    ASSERT_SAME_TYPE(decltype(s),
                     std::set<int, std::less<int>, test_allocator<int> >);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 44);
  }

  {
    std::set s({ 1, 2, 1, INT_MAX, 3 }, test_allocator<int>(0, 45));

    ASSERT_SAME_TYPE(decltype(s),
                     std::set<int, std::less<int>, test_allocator<int> >);
    const int expected_s[] = { 1, 2, 3, INT_MAX };
    assert(std::equal(s.begin(), s.end(), std::begin(expected_s),
                      std::end(expected_s)));
    assert(s.get_allocator().get_id() == 45);
  }

  {
    NotAnAllocator a;
    std::set s{ a }; // set(initializer_list<NotAnAllocator>)
    ASSERT_SAME_TYPE(decltype(s), std::set<NotAnAllocator>);
    assert(s.size() == 1);
  }

  {
    std::set<long> source;
    std::set s{ source, source }; // set(initializer_list<set<long>>)
    ASSERT_SAME_TYPE(decltype(s), std::set<std::set<long> >);
    assert(s.size() == 1);
  }

  {
    NotAnAllocator a;
    std::set s{ a, a }; // set(initializer_list<NotAnAllocator>)
    ASSERT_SAME_TYPE(decltype(s), std::set<NotAnAllocator>);
    assert(s.size() == 1);
  }

  {
    int source[3] = { 3, 4, 5 };
    std::set s(source, source + 3); // set(InputIterator, InputIterator)
    ASSERT_SAME_TYPE(decltype(s), std::set<int>);
    assert(s.size() == 3);
  }

  {
    int source[3] = { 3, 4, 5 };
    std::set s{ source, source + 3 }; // set(initializer_list<int*>)
    ASSERT_SAME_TYPE(decltype(s), std::set<int *>);
    assert(s.size() == 2);
  }

  return 0;
}
