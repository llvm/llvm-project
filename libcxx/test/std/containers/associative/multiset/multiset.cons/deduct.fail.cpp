//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// XFAIL: clang-6, apple-clang-9.0, apple-clang-9.1, apple-clang-10.0.0
//  clang-6 gives different error messages.

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

#include <functional>
#include <set>
#include <type_traits>

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

int main(int, char **) {
  {
    // cannot deduce Key from nothing
    std::multiset s;
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'multiset'}}
  }
  {
    // cannot deduce Key from just (Compare)
    std::multiset s(std::less<int>{});
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'multiset'}}
  }
  {
    // cannot deduce Key from just (Compare, Allocator)
    std::multiset s(std::less<int>{}, std::allocator<int>{});
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'multiset'}}
  }
  {
    // cannot deduce Key from multiset(Allocator)
    std::multiset s(std::allocator<int>{});
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'multiset'}}
  }
  {
    // since we have parens, not braces, this deliberately does not find the
    // initializer_list constructor
    NotAnAllocator a;
    std::multiset s(a);
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'multiset'}}
  }

  return 0;
}
