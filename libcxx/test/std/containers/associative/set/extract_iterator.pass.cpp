//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <set>

// class set

// constexpr node_type extract(const_iterator); // constexpr since C++26

#include <set>
#include "test_macros.h"
#include "min_allocator.h"
#include "Counter.h"

template <class Container>
TEST_CONSTEXPR_CXX26 void test(Container& c) {
  std::size_t sz = c.size();

  for (auto first = c.cbegin(); first != c.cend();) {
    auto key_value                  = *first;
    typename Container::node_type t = c.extract(first++);
    --sz;
    assert(t.value() == key_value);
    assert(t.get_allocator() == c.get_allocator());
    assert(sz == c.size());
  }

  assert(c.size() == 0);
}

TEST_CONSTEXPR_CXX26 bool test() {
  {
    using set_type = std::set<int>;
    set_type m     = {1, 2, 3, 4, 5, 6};
    test(m);
  }

  if (!TEST_IS_CONSTANT_EVALUATED) {
    std::set<Counter<int>> m = {1, 2, 3, 4, 5, 6};
    assert(Counter_base::gConstructed == 6);
    test(m);
    assert(Counter_base::gConstructed == 0);
  }

  {
    using min_alloc_set = std::set<int, std::less<int>, min_allocator<int>>;
    min_alloc_set m     = {1, 2, 3, 4, 5, 6};
    test(m);
  }

  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
