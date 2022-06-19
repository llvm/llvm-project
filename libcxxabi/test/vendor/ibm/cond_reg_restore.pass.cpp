//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the condition register is restored properly during unwinding
// on AIX. Option -O3 is required so that the compiler will re-use the value
// in the condition register instead of re-evaluating the condition expression.

// REQUIRES: target=powerpc{{(64)?}}-ibm-aix
// ADDITIONAL_COMPILE_FLAGS: -O3
// UNSUPPORTED: no-exceptions

#include <cstdlib>
#include <cassert>

int __attribute__((noinline)) test2(int i) {
  // The inline assembly forces the prologue/epilogue to save/restore the
  // condition register.
  asm volatile("nop" : : : "cr2");
  if (i > 3) {
    throw i;
  }
  srand(i);
  return rand() + i;
}

void __attribute__((noinline)) test(int argc, const char **argv) {
  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  try {
    test2(a < b ? argc : b);
  } catch (int num) {
    assert(!(a < b));
  }
}
int main(int, char**) {
  const char *av[]={"a.out", "12", "10"};
  test(4, av);
  return 0;
}
