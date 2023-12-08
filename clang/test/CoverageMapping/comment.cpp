// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

int f() {
  int x = 0;
#include "Inputs/comment.h" /*
    */
  return x;
}

// CHECK: File 0, 3:9 -> 8:2 = #0
// CHECK-NEXT: Expansion,File 0, 5:10 -> 5:28 = #0
// CHECK-NEXT: Skipped,File 0, 6:1 -> 6:7 = 0
// CHECK-NEXT: File 1, 1:1 -> 7:1 = #0
