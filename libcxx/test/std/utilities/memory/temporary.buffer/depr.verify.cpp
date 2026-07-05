//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++17

// Ensure allocator<void> is deprecated

#include <memory>

void test() {
  auto a = std::get_temporary_buffer<int>(1); // expected-warning {{'get_temporary_buffer<int>' is deprecated}}
  std::return_temporary_buffer(a.first); // expected-warning {{'return_temporary_buffer<int>' is deprecated}}
}
