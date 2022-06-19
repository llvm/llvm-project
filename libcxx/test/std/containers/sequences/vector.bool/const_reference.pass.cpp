//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

#include <cassert>
#include <vector>

#include "test_macros.h"

bool test() {
  using CRefT = std::vector<bool>::const_reference;
#if !defined(_LIBCPP_VERSION) || defined(_LIBCPP_ABI_BITSET_VECTOR_BOOL_CONST_SUBSCRIPT_RETURN_BOOL)
  ASSERT_SAME_TYPE(CRefT, bool);
#else
  ASSERT_SAME_TYPE(CRefT, std::__bit_const_reference<std::vector<bool> >);
  std::vector<bool> vec;
  vec.push_back(true);
  CRefT ref = vec[0];
  assert(ref);
  vec[0] = false;
  assert(!ref);
#endif

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
