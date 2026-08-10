// RUN: %clang_cc1 -std=c++2a -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,REGULAR
// RUN: %clang_cc1 -std=c++2a -fsanitize=shift-base,shift-exponent -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,SANITIZED

// CHECK-LABEL: @_Z12lsh_overflow
int lsh_overflow(int a, int b) {
  // SANITIZED: %[[RHS_INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // SANITIZED-NEXT: br i1 %[[RHS_INBOUNDS]], label %[[VALID:.*]], label

  // SANITIZED: call void @__ubsan_handle_shift_out_of_bounds

  // No check for the LHS here.
  // SANITIZED: [[VALID]]:
  // SANITIZED-NEXT: shl i32 %
  // SANITIZED-NEXT: ret i32

  // Just ensure there's no nsw nuw flags here.
  // REGULAR: shl i32 %
  return a << b;
}
