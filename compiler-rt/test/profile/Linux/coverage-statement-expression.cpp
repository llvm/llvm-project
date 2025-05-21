// REQUIRES: lld-available
// XFAIL: powerpc64-target-arch

// RUN: %clangxx_profgen -std=gnu++17 -fuse-ld=lld -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile=%t.profdata 2>&1 | FileCheck %s

#include <stdio.h>

// clang-format off
__attribute__ ((__noreturn__))
void foo(void) { while (1); }                   // CHECK:  [[@LINE]]| 0|void foo(void)
_Noreturn void bar(void) { while (1); }         // CHECK:  [[@LINE]]| 0|_Noreturn void bar(void)
                                                // CHECK:  [[@LINE]]|  |
int main(int argc, char **argv) {               // CHECK:  [[@LINE]]| 1|int main(
  int rc = ({ if (argc > 3) foo(); 0; });       // CHECK:  [[@LINE]]| 1|  int rc =
  printf("coverage after foo is present\n");    // CHECK:  [[@LINE]]| 1|  printf(
                                                // CHECK:  [[@LINE]]|  |
  int rc2 = ({ if (argc > 3) bar(); 0; });      // CHECK:  [[@LINE]]| 1|  int rc2 =
  printf("coverage after bar is present\n");    // CHECK:  [[@LINE]]| 1|  printf(
  return rc + rc2;                              // CHECK:  [[@LINE]]| 1|  return rc
}                                               // CHECK:  [[@LINE]]| 1|}
// clang-format on
