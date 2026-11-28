//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// Make sure that a std::vector2 containing move-only types can't be copied.


#include <vector>

#include "MoveOnly.h"

void f() {
  std::vector2<MoveOnly> v;
  std::vector2<MoveOnly> copy =
      v; // expected-error-re@* {{{{(no matching function for call to '__construct_at')|(call to deleted constructor of 'MoveOnly')}}}}
  (void)copy;
}
