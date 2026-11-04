// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{powerpc64(le)?-.+}}

// Check that unw_get_fpreg() reads and unw_set_fpreg() writes the half of a
// ppc64 VSR slot that holds the floating point register.  With VSX,
// unw_getcontext() stores and unw_resume() loads the whole 128-bit register,
// and on little endian the slot holds the little-endian image of it, so the
// FPR is in the upper eight bytes there.  Reading the wrong half returns the
// other doubleword of the VSR, and writing the wrong half leaves the FPR
// unchanged when unw_resume() reloads the register.

#undef NDEBUG
#include <assert.h>
#include <libunwind.h>
#include <stdlib.h>

static volatile int resumed = 0;

__attribute__((noinline)) static void test() {
  const double initial[2] = {3.25, 0.5};
  double after;
  unw_context_t context;
  unw_cursor_t cursor;
  unw_fpreg_t fpreg;

  // The asm statements stay in this function: declaring f31 clobbered in a
  // callee would just make the callee save and restore it.
#if defined(__VSX__)
  // Fill both doublewords of vs31 so that reading the wrong one is caught:
  // lxvd2x puts initial[0] in doubleword 0, which is f31, and initial[1] in
  // doubleword 1 on both endiannesses.
  asm volatile("lxvd2x 31, 0, %0" : : "r"(initial) : "f31", "memory");
#else
  asm volatile("lfd 31, %0" : : "m"(initial[0]) : "f31");
#endif
  unw_getcontext(&context);
  if (!resumed) {
    resumed = 1;
    assert(unw_init_local(&cursor, &context) == UNW_ESUCCESS);

    // The context must give back the value that was in f31.
    assert(unw_get_fpreg(&cursor, UNW_PPC64_F31, &fpreg) == UNW_ESUCCESS);
    assert(fpreg == 3.25);

    // A value written into the context must come back in f31 when execution
    // resumes after unw_getcontext().
    assert(unw_set_fpreg(&cursor, UNW_PPC64_F31, 1.5) == UNW_ESUCCESS);
    unw_resume(&cursor);
    abort();
  }
  asm volatile("stfd 31, %0" : "=m"(after));
  assert(after == 1.5);
}

int main(int, char **) {
  test();
  return 0;
}
