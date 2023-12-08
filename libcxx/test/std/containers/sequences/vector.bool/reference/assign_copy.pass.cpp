//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// reference& operator=(const reference&)

#include <cassert>
#include <vector>
#include <utility>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
  std::vector<bool> vec;
  typedef std::vector<bool>::reference Ref;
  vec.push_back(true);
  vec.push_back(false);
  Ref ref1 = vec[0];
  Ref ref2 = vec[1];
  ref2 = ref1;
  // Ref&
  {
    vec[0] = false;
    vec[1] = true;
    ref1 = ref2;
    assert(vec[0]);
    assert(vec[1]);
  }
  {
    vec[0] = true;
    vec[1] = false;
    ref1 = ref2;
    assert(!vec[0]);
    assert(!vec[1]);
  }
  // Ref&&
  {
    vec[0] = false;
    vec[1] = true;
    ref1 = std::move(ref2);
    assert(vec[0]);
    assert(vec[1]);
  }
  {
    vec[0] = true;
    vec[1] = false;
    ref1 = std::move(ref2);
    assert(!vec[0]);
    assert(!vec[1]);
  }
  // const Ref&
  {
    vec[0] = false;
    vec[1] = true;
    ref1 = static_cast<const Ref&>(ref2);
    assert(vec[0]);
    assert(vec[1]);
  }
  {
    vec[0] = true;
    vec[1] = false;
    ref1 = static_cast<const Ref&>(ref2);
    assert(!vec[0]);
    assert(!vec[1]);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif

  return 0;
}
