// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux && target={{riscv64-.+}}

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>

#ifdef __riscv_vector
__attribute__((noinline)) extern "C" void stepper() {
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  // Stepping into foo() should succeed.
  assert(unw_step(&cursor) > 0);
  // Stepping past foo() should succeed, too.
  assert(unw_step(&cursor) > 0);
}

// Check correct unwinding of frame with VLENB-sized objects (vector registers).
__attribute__((noinline)) static void foo() {
  __rvv_int32m1_t v;
  asm volatile("" : "=vr"(v)); // Dummy inline asm to def v.
  stepper();                   // def-use of v has cross the function, so that
                               // will triger spill/reload to/from the stack.
  asm volatile("" ::"vr"(v));  // Dummy inline asm to use v.
}

int main(int, char **) {
  foo();
  return 0;
}
#else
int main(int, char **) { return 0; }
#endif
