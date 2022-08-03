//===-- Loader test to test init and fini array iteration -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/IntegrationTest/test.h"

class A {
private:
  int val[1024];

public:
  A(int i, int a) {
    for (int k = 0; k < 1024; ++k)
      val[k] = 0;
    val[i] = a;
  }

  // TODO: When we have implementation for __cxa_atexit, an explicit definition
  // of the destructor should be provided to test that path of registering the
  // destructor callback for a global.
  ~A() = default;

  int get(int i) const { return val[i]; }
};

int GLOBAL_INDEX = 512;
int INITVAL_INITIALIZER = 0x600D;

A global(GLOBAL_INDEX, INITVAL_INITIALIZER);

int initval = 0;
__attribute__((constructor)) void set_initval() {
  initval = INITVAL_INITIALIZER;
}
__attribute__((destructor)) void reset_initval() { initval = 0; }

int preinitval = 0;
void set_preinitval() { preinitval = INITVAL_INITIALIZER; }
__attribute__((destructor)) void reset_preinitval() { preinitval = 0; }

using PreInitFunc = void();
__attribute__((section(".preinit_array"))) PreInitFunc *preinit_func_ptr =
    &set_preinitval;

TEST_MAIN() {
  ASSERT_EQ(global.get(GLOBAL_INDEX), INITVAL_INITIALIZER);
  ASSERT_EQ(initval, INITVAL_INITIALIZER);
  ASSERT_EQ(preinitval, INITVAL_INITIALIZER);
  return 0;
}
