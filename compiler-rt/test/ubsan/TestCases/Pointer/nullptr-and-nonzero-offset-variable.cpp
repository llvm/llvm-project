// RUN: %clang -x c   -fsanitize=pointer-overflow -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB-C
// RUN: %clang -x c   -fsanitize=pointer-overflow -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB-C
// RUN: %clang -x c   -fsanitize=pointer-overflow -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB-C
// RUN: %clang -x c   -fsanitize=pointer-overflow -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB-C

// RUN: %clang -x c++ -fsanitize=pointer-overflow -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK
// RUN: %clang -x c++ -fsanitize=pointer-overflow -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK
// RUN: %clang -x c++ -fsanitize=pointer-overflow -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK
// RUN: %clang -x c++ -fsanitize=pointer-overflow -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK

// RUN: %clang -x c   -fsanitize=pointer-overflow -O0 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB
// RUN: %clang -x c   -fsanitize=pointer-overflow -O1 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB
// RUN: %clang -x c   -fsanitize=pointer-overflow -O2 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB
// RUN: %clang -x c   -fsanitize=pointer-overflow -O3 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB

// RUN: %clang -x c++ -fsanitize=pointer-overflow -O0 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB
// RUN: %clang -x c++ -fsanitize=pointer-overflow -O1 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB
// RUN: %clang -x c++ -fsanitize=pointer-overflow -O2 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB
// RUN: %clang -x c++ -fsanitize=pointer-overflow -O3 %s -o %t && %run %t I_AM_UB 2>&1 | FileCheck %s --implicit-check-not="error:" --check-prefix=CHECK-UB

#include <stdint.h>
#include <stdio.h>

// Just so deduplication doesn't do anything.
static char *getelementpointer_inbounds_v0(char *base, unsigned long offset) {
  // Potentially UB.
  return base + offset;
}

int main(int argc, char *argv[]) {
  char *base;
  unsigned long offset;

  printf("Dummy\n");
  // CHECK: Dummy

  base = (char *)0;
  offset = argc - 1;
  (void)getelementpointer_inbounds_v0(base, offset);
  // CHECK-UB: {{.*}}.cpp:[[@LINE-13]]:15: runtime error: applying non-zero offset 1 to null pointer
  // CHECK-UB-C: {{.*}}.cpp:[[@LINE-14]]:15: runtime error: applying zero offset to null pointer

  return 0;
}
