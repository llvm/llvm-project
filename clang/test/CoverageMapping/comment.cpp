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

int g() {

  /* Merged with the empty line above. */ return 0;
}

// CHECK-LABEL: {{_Z1gv|\?g@@YAHXZ}}:
// CHECK-NEXT: File 0, [[@LINE-6]]:9 -> [[@LINE-3]]:2 = #0
// CHECK-NEXT: Skipped,File 0, [[@LINE-6]]:1 -> [[@LINE-6]]:{{[0-9]+}} = 0

#define RET(x) return x;
#define RET0 return 0;

int h(int x) {
  RET(x) // Follows a macro.
}

// CHECK-LABEL: {{_Z1hi|\?h@@YAHH@Z}}:
// CHECK-NEXT: File 0, [[@LINE-5]]:14 -> [[@LINE-3]]:2 = #0
// CHECK-NEXT: Expansion,File 0, [[@LINE-5]]:3 -> [[@LINE-5]]:6 = #0

int i() {
  /* Precedes a macro. */ RET0
}

// CHECK-LABEL: {{_Z1iv|\?i@@YAHXZ}}:
// CHECK-NEXT: File 0, [[@LINE-5]]:9 -> [[@LINE-3]]:2 = #0
// CHECK-NEXT: Expansion,File 0, [[@LINE-5]]:27 -> [[@LINE-5]]:31 = #0
