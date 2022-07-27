//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// operator bool()

#include <cassert>
#include <type_traits>
#include <vector>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
  std::vector<bool> vec;
  typedef std::vector<bool>::reference Ref;
  static_assert(std::is_convertible<Ref, bool>::value, "");

  vec.push_back(true);
  vec.push_back(false);
  Ref true_ref = vec[0];
  Ref false_ref = vec[1];
  bool b = true_ref;
  assert(b);
  assert(true_ref);
  assert(!false_ref);

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif

  return 0;
}
