//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map();

#include <map>

#include "test_macros.h"

struct X {
  std::map<int, X> m;
  std::map<int, X>::iterator i;
  std::map<int, X>::const_iterator ci;
#if TEST_STD_VER <= 17
  // These reverse_iterator specializations require X to be complete in C++20.
  std::map<int, X>::reverse_iterator ri;
  std::map<int, X>::const_reverse_iterator cri;
#endif // TEST_STD_VER <= 17
};
