//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// move_iterator

#include <iterator>

int main(int, char**) {
  (void)std::move_iterator<int*>().operator->();
  // expected-warning@-1{{'operator->' is deprecated}}

  return 0;
}
