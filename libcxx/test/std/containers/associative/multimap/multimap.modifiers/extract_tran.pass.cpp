//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <map>

// class multimap

// template<class K>
//     node_type extract(K&& x);
//
//   The member function templates find, count, lower_bound, upper_bound,
// equal_range, erase, and extract shall not participate in overload resolution
// unless the qualified-id Compare::is_transparent is valid and denotes a type.
// Additionally, the member function templates erase and extract shall not
// participate in overload resolution if is_convertible_v<K&&, iterator> ||
// is_convertible_v<K&&, const_iterator> is true, where K is the type
// substituted as the first template argument

#include <map>
#include <cassert>

#include "test_macros.h"
#include "is_transparent.h"

int main(int, char**) {
  {
    typedef std::multimap<int, double, transparent_less> M;
    M example;
    assert(example.extract(C2Int{5}).empty());
  }
  {
    typedef std::multimap<int, double, transparent_less_not_referenceable> M;
    M example;
    assert(example.extract(C2Int{5}).empty());
  }
  {
    typedef std::multimap<int, double, transparent_less> M;
    M example;
    example.insert({5, 1.});
    example.insert({6, 2.});
    example.insert({4, 0.});
    assert(example.size() == 3);
    M::node_type nh = example.extract(C2Int{5});
    assert(nh && nh.key() == 5 && nh.mapped() == 1.);
    assert(example.size() == 2);
    example.insert(std::move(nh));
    assert(example.size() == 3);
    assert(example.find(5)->second == 1.);
  }

  return 0;
}
