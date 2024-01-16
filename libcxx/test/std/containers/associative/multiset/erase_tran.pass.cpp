//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: c++03, c++11, c++14, c++17, c++20

// <set>

// class multiset

// template<class K>
//     size_type erase(K&& x);
//
//   The member function templates find, count, lower_bound, upper_bound,
// equal_range, erase, and extract shall not participate in overload resolution
// unless the qualified-id Compare::is_transparent is valid and denotes a type.
// Additionally, the member function templates erase and extract shall not
// participate in overload resolution if is_convertible_v<K&&, iterator> ||
// is_convertible_v<K&&, const_iterator> is true, where K is the type
// substituted as the first template argument

#include <set>
#include <cassert>

#include "test_macros.h"
#include "is_transparent.h"

int main(int, char**) {
  {
    typedef std::multiset<int, transparent_less> M;
    M example;
    assert(example.erase(C2Int{5}) == 0);
  }
  {
    typedef std::multiset<int, transparent_less_not_referenceable> M;
    M example;
    assert(example.erase(C2Int{5}) == 0);
  }
  {
    typedef std::multiset<int, transparent_less> M;
    M example;
    example.insert(5);
    example.insert(6);
    example.insert(4);
    assert(example.size() == 3);
    M::size_type erased = example.erase(C2Int{5});
    assert(erased == 1);
    assert(example.size() == 2);
  }

  return 0;
}
