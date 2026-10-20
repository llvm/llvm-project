// REQUIRES: lld-available
// XFAIL: powerpc64-target-arch

// RUN: %clangxx_profgen -fuse-ld=lld -fcoverage-mapping -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile=%t.profdata 2>&1 | FileCheck %s

#include <stdio.h>

// clang-format off
int main(int argc, char **argv) {           // CHECK:  [[@LINE]]| 1|int main(
  do {                                      // CHECK:  [[@LINE]]| 1|  do {
    if (argc == 87) {                       // CHECK:  [[@LINE]]| 1|    if (argc
      break;                                // CHECK:  [[@LINE]]| 0|      break
    }                                       // CHECK:  [[@LINE]]| 0|    }
  } while (0);                              // CHECK:  [[@LINE]]| 1|  } while
  printf("coverage after do is present\n"); // CHECK:  [[@LINE]]| 1|  printf(
  return 0;                                 // CHECK:  [[@LINE]]| 1|  return
}                                           // CHECK:  [[@LINE]]| 1|}
// clang-format on
