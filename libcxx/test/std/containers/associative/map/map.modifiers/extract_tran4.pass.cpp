//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: c++03, c++11, c++14, c++17, c++20

// <map>

// class map

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

typedef std::map<int, double, transparent_less> M;
M example;

struct Convertible : C2Int {
  operator M::const_iterator() { return example.begin(); }
};

int main(int, char**) {
  {
    example.insert({2, 3.});
    M::node_type nh = example.extract(Convertible());
    assert(nh && nh.key() == 2 && nh.mapped() == 3.);
  }
}
