//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// check that clang warns on non-power-of-two alignment

#include <memory_resource>

void func() {
  std::pmr::polymorphic_allocator<> allocator;
  (void)allocator.allocate_bytes(0, 3); // expected-warning {{requested alignment is not a power of 2}}

}
