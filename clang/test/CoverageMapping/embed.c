// RUN: %clang_cc1 -std=c23 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

int found(void) {
  int data[] = {
#embed "Inputs/ends_a_scope_only" limit(1) suffix(, 2,) prefix(1,)
// found comment

    3
  };
  return data[0];
}

int empty(void) {
  int data[] = {
#embed "Inputs/ends_a_scope_only" limit(0) if_empty(1,)
// empty comment

    2
  };
  return data[0];
}

// CHECK-LABEL: found:
// CHECK-NEXT: File 0, 3:17 -> 11:2 = #0
// CHECK-NEXT: Skipped,File 0, 6:1 -> 7:1 = 0
// CHECK-LABEL: empty:
// CHECK-NEXT: File 0, 13:17 -> 21:2 = #0
// CHECK-NEXT: Skipped,File 0, 16:1 -> 17:1 = 0
