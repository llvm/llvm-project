//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test that libc++ does not generate a warning diagnostic about the comparator
// too early for containers of incomplete types.
//
// See PR41360.

#include <set>
#include <map>
#include <functional>

template <template <typename ...> class Container>
void test_set() {
  struct KeyBase { };
  struct KeyDerived;  // derives from KeyBase, but incomplete at this point

  // Name the type but don't instantiate it.
  using C = Container<KeyDerived*, std::less<KeyBase*>>;

  // Instantiate it but don't ODR use any members.
  typename C::value_type dummy; (void)dummy;

  // Complete the types.
  struct KeyDerived : KeyBase { };

  C c; // ODR use it, which should be OK
}

template <template <typename ...> class Container>
void test_map() {
  struct Value { };
  struct KeyBase { };
  struct KeyDerived;
  using C = Container<KeyDerived*, Value, std::less<KeyBase*>>;
  typename C::value_type dummy; (void)dummy;
  struct KeyDerived : KeyBase { };
  C c;
}

void f() {
  // expected-no-diagnostics
  test_set<std::set>();
  test_set<std::multiset>();
  test_map<std::map>();
  test_map<std::multimap>();
}
