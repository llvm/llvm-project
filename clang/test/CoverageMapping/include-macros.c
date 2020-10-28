// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name include-macros.c %s | FileCheck %s

#include "Inputs/macros.h"

void f1() {
  M2("a", "b");
}

// CHECK-LABEL: f1:
// CHECK-NEXT:   File 0, 5:11 -> 7:2 = #0
// CHECK-NEXT:   Expansion,File 0, 6:3 -> 6:5 = #0 (Expanded file = 1)
// CHECK-NEXT:   File 1, 13:20 -> 13:50 = #0
// CHECK-NEXT:   Expansion,File 1, 13:20 -> 13:22 = #0 (Expanded file = 2)
// CHECK-NEXT:   File 2, 7:20 -> 7:46 = #0
// CHECK-NEXT:   Expansion,File 2, 7:33 -> 7:44 = #0 (Expanded file = 3)
// CHECK-NEXT:   File 3, 13:26 -> 13:34 = #0
// CHECK-NEXT:   Expansion,File 3, 13:26 -> 13:33 = #0 (Expanded file = 4)
// CHECK-NEXT:   File 4, 3:17 -> 3:18 = #0
