// RUN: %clang_cc1 %s -ast-print -o - | FileCheck %s

// FIXME: Test fails because attribute order is reversed by ParsedAttributes.
// XFAIL: *

void run1(int *List, int Length) {
  int i = 0;
// CHECK: #pragma loop vectorize(4)
// CHECK-NEXT: #pragma loop interleave(8)
// CHECK-NEXT: #pragma loop vectorize(enable)
// CHECK-NEXT: #pragma loop interleave(enable)
#pragma loop vectorize(4)
#pragma loop interleave(8)
#pragma loop vectorize(enable)
#pragma loop interleave(enable)
// CHECK-NEXT: while (i < Length)
  while (i < Length) {
    List[i] = i;
    i++;
  }
}
