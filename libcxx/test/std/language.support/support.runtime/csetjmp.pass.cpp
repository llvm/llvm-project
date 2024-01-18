//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <csetjmp>

#include <csetjmp>
#include <cassert>
#include <type_traits>

int main(int, char**) {
  std::jmp_buf jb;

  switch (setjmp(jb)) {
  // First time we set the buffer, the function should return 0
  case 0:
    break;

  // If it returned 42, then we're coming from the std::longjmp call below
  case 42:
    return 0;

  // Otherwise, something is wrong
  default:
    return 1;
  }

  std::longjmp(jb, 42);
  static_assert(std::is_same<decltype(std::longjmp(jb, 0)), void>::value, "");

  return 1;
}
