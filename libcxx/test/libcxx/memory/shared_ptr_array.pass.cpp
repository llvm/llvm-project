//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// These compiler versions and platforms don't enable sized deallocation by default.
// ADDITIONAL_COMPILE_FLAGS(clang-17): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(clang-18): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(apple-clang-15): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(apple-clang-16): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(target=x86_64-w64-windows-gnu): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(target=i686-w64-windows-gnu): -fsized-deallocation

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
