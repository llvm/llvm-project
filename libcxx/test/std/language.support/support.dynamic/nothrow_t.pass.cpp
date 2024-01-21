//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// struct nothrow_t {
//   explicit nothrow_t() = default;
// };
// extern const nothrow_t nothrow;

#include <new>


int main(int, char**) {
  std::nothrow_t x = std::nothrow;
  (void)x;

  return 0;
}
