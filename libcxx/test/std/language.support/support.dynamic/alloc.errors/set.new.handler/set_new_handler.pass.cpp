//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// a test will allocate an impossible amount of memory on purpose
// and it will trigger an expected failure
// UNSUPPORTED: asan, msan, tsan

// test set_new_handler

#include <new>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <climits>

#include "test_macros.h"

void f1() {}
void f2() {}
void f3() { std::exit(0); }

int main(int, char**) {
  assert(std::set_new_handler(f1) == 0);
  assert(std::set_new_handler(f2) == f1);

  // note: this test must be the last test since it will trigger
  // the new handler and exit on the first call to it to signal success
  // else we would be stuck in the handler being called in a loop
  std::set_new_handler(f3);
  size_t max_size = std::numeric_limits<size_t>::max();
  // cap size to (~4GB) on 32-bit, (~1Exabytes) on 64-bit
  size_t alloc_size = max_size > UINT_MAX ? max_size / 16 : max_size - 1;
  auto ptr          = new char[alloc_size]; // huge allocation
  (void)ptr;

  return 0;
}
