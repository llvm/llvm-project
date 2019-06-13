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
// XFAIL: clang-6, apple-clang-9.0, apple-clang-9.1, apple-clang-10.0
//  clang-6 gives different error messages.

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

#include <functional>
#include <set>
#include <type_traits>

struct NotAnAllocator {
  friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

int main(int, char **) {
  {
    // cannot deduce Key from nothing
    std::set s;
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'set'}}
  }
  {
    // cannot deduce Key from just (Compare)
    std::set s(std::less<int>{});
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'set'}}
  }
  {
    // cannot deduce Key from just (Compare, Allocator)
    std::set s(std::less<int>{}, std::allocator<int>{});
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'set'}}
  }
  {
    // cannot deduce Key from just (Allocator)
    std::set s(std::allocator<int>{});
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'set'}}
  }
  {
    // since we have parens, not braces, this deliberately does not find the
    // initializer_list constructor
    NotAnAllocator a;
    std::set s(a);
    // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'set'}}
  }

  return 0;
}
