//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// REQUIRES: verify-support

// Test that libc++ does not generate a warning diagnostic about the comparator
// or the hasher too early for containers of incomplete types.
//
// See PR41360.

#include <unordered_set>
#include <unordered_map>
#include <functional>

#include "test_macros.h"


template <template <typename ...> class Container>
void test_set() {
  struct KeyBase { };
  struct KeyDerived;  // derives from KeyBase, but incomplete at this point

  // Name the type but don't instantiate it.
  using C = Container<KeyDerived*, std::hash<KeyBase*>, std::equal_to<KeyBase*>>;

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
  using C = Container<KeyDerived*, Value, std::hash<KeyBase*>, std::equal_to<KeyBase*>>;
  typename C::value_type dummy; (void)dummy;
  struct KeyDerived : KeyBase { };
  C c;
}

int main(int, char**) {
  // expected-no-disagnostics
  test_set<std::unordered_set>();
  test_set<std::unordered_multiset>();
  test_map<std::unordered_map>();
  test_map<std::unordered_multimap>();

  return 0;
}
