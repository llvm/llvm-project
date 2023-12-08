//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// reference& operator=(bool)

#include <cassert>
#include <vector>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
  std::vector<bool> vec;
  typedef std::vector<bool>::reference Ref;
  vec.push_back(true);
  vec.push_back(false);
  Ref ref = vec[0];
  const Ref cref = vec[0];

  assert(ref);
  ref = false;
  assert(!vec[0]);
  assert(!vec[1]);
  ref = true;
  assert(vec[0]);
  assert(!vec[1]);

  assert(cref);

#if TEST_STD_VER > 20
  cref = false;
  assert(!vec[0]);
  assert(!vec[1]);
  cref = true;
  assert(vec[0]);
  assert(!vec[1]);
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif

  return 0;
}
