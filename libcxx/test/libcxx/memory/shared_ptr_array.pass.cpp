//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// REQUIRES: -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS: -fsized-deallocation

// This test will fail with ASan if the implementation passes different sizes
// to corresponding allocation and deallocation functions.

#include <memory>

int main(int, char**) {
  std::allocate_shared<int[]>(std::allocator<int>{}, 10);
  std::make_shared<int[]>(10);

  std::allocate_shared<int[10]>(std::allocator<int>{});
  std::make_shared<int[10]>();

  return 0;
}
