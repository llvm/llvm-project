//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

// FIXME: Remove after issue https://llvm.org/PR127348 resolved.
extern "C" const char* __asan_default_options() { return "check_initialization_order=true:strict_init_order=true"; }

// Test that ios used from globals constructors doesn't trigger Asan initialization-order-fiasco.

struct Global {
  Global() { std::cout << "Hello!"; }
} global;

int main(int, char**) { return 0; }
