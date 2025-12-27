//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>::reference operator[](size_t pos); // constexpr since C++23

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>
#include <type_traits>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
TEST_CONSTEXPR_CXX23 void test_index() {
  typedef typename std::bitset<N>::reference BRef;
#if TEST_STD_VER >= 23
  static_assert(std::is_assignable_v<const BRef, bool>);
#else
  static_assert(!std::is_assignable<const BRef, bool>::value, "");
#endif

  std::vector<std::bitset<N> > const cases = get_test_cases<N>();
  for (std::size_t c = 0; c != cases.size(); ++c) {
    std::bitset<N> v1 = cases[c];
    if (v1.size() > 0) {
      assert(v1[N / 2] == v1.test(N / 2));
      BRef r = v1[N / 2];
      assert(r == v1.test(N / 2));
      BRef r2 = v1[N / 2];
      r       = r2;
      assert(r == v1.test(N / 2));
      r = false;
      assert(r == false);
      assert(v1.test(N / 2) == false);
      r = true;
      assert(r == true);
      assert(v1.test(N / 2) == true);
      bool b = ~r;
      assert(r == true);
      assert(v1.test(N / 2) == true);
      assert(b == false);
      r.flip();
      assert(r == false);
      assert(v1.test(N / 2) == false);

#if TEST_STD_VER >= 23
      const auto& cr     = r;
      decltype(auto) cr2 = (cr = true);
      ASSERT_SAME_TYPE(decltype(cr2), const BRef&);
      assert(&cr2 == &cr);

      assert(r == true);
      assert(v1[N / 2] == true);
#endif
    }
    ASSERT_SAME_TYPE(decltype(v1[0]), BRef);
  }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_index<0>();
  test_index<1>();
  test_index<31>();
  test_index<32>();
  test_index<33>();
  test_index<63>();
  test_index<64>();
  test_index<65>();

  std::bitset<1> set;
  set[0] = false;
  auto b = set[0];
  set[0] = true;
  assert(b);

  return true;
}

int main(int, char**) {
  test();
  test_index<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
