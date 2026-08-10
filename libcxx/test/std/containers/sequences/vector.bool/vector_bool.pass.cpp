//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
// {
//     size_t operator()(T val) const;
// };

#include <vector>
#include <cassert>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "min_allocator.h"

template <class VB>
TEST_CONSTEXPR_CXX20 void test() {
  typedef std::hash<VB> H;
#if TEST_STD_VER <= 14
  static_assert((std::is_same<typename H::argument_type, VB>::value), "");
  static_assert((std::is_same<typename H::result_type, std::size_t>::value), "");
#endif
  ASSERT_NOEXCEPT(H()(VB()));

  bool ba[] = {true, false, true, true, false};
  VB vb(std::begin(ba), std::end(ba));
  H h;
  if (!TEST_IS_CONSTANT_EVALUATED) {
    const std::size_t hash_value = h(vb);
    assert(h(vb) == hash_value);
    LIBCPP_ASSERT(hash_value != 0);
  }
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test<std::vector<bool> >();
#if TEST_STD_VER >= 11
  test<std::vector<bool, min_allocator<bool>>>();
#endif

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
