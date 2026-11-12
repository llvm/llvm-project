//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-localization

// <flat_set>

// As an extension, libc++ flat containers support inserting a non forward range into
// a pre-C++23 container that doesn't provide insert_range(...), since many containers
// out there are in that situation. See https://llvm.org/PR136656

#include <algorithm>
#include <cassert>
#include <flat_set>
#include <ranges>
#include <sstream>
#include <vector>

#include "../flat_helpers.h"
#include "test_iterators.h"
#include "test_macros.h"

constexpr bool test() {
  NotQuiteSequenceContainer<int> v;
  std::flat_set s(v);

  int ar[]   = {0, 1, 1, 0};
  using Iter = cpp20_input_iterator<const int*>;
  using Sent = sentinel_wrapper<Iter>;
  using R    = std::ranges::subrange<Iter, Sent>;
  auto r     = R(Iter(ar), Sent(Iter(ar + 4)));

  static_assert(
      ![](auto& t) { return requires { t.insert_range(t.end(), r); }; }(v),
      "This test is to test the case where the underlying container does not provide insert_range");
  s.insert_range(r);
  assert(std::ranges::equal(s, std::vector<int>{0, 1}));

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
