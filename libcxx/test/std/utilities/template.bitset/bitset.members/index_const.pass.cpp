//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test constexpr bool operator[](size_t pos) const;

#include <bitset>
#include <cassert>
#include <cstddef>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <std::size_t N>
void test_index_const() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        std::bitset<N> const v = cases[c];
        if (v.size() > 0) {
            assert(v[N/2] == v.test(N/2));
        }
    }
#if !defined(_LIBCPP_VERSION) || defined(_LIBCPP_ABI_BITSET_VECTOR_BOOL_CONST_SUBSCRIPT_RETURN_BOOL)
    ASSERT_SAME_TYPE(decltype(cases[0][0]), bool);
#else
    ASSERT_SAME_TYPE(decltype(cases[0][0]), typename std::bitset<N>::const_reference);
#endif
}

int main(int, char**) {
    test_index_const<0>();
    test_index_const<1>();
    test_index_const<31>();
    test_index_const<32>();
    test_index_const<33>();
    test_index_const<63>();
    test_index_const<64>();
    test_index_const<65>();
    test_index_const<1000>();

  std::bitset<1> set_;
  set_[0] = false;
  const auto& set = set_;
  auto b = set[0];
  set_[0] = true;
#if !defined(_LIBCPP_VERSION) || defined(_LIBCPP_ABI_BITSET_VECTOR_BOOL_CONST_SUBSCRIPT_RETURN_BOOL)
  assert(!b);
#else
  assert(b);
#endif

    return 0;
}
