// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-shift-base,shift-exponent %s -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: lsh_overflow(
unsigned lsh_overflow(unsigned a, unsigned b) {
  // CHECK: %[[RHS_INBOUNDS:.*]] = icmp ule i32 %[[RHS:.*]], 31
  // CHECK-NEXT: br i1 %[[RHS_INBOUNDS]], label %[[CHECK_BB:.*]], label %[[CONT_BB:.*]],

  // CHECK:      [[CHECK_BB]]:
  // CHECK-NEXT: %[[SHIFTED_OUT_WIDTH:.*]] = sub nuw nsw i32 31, %[[RHS]]
  // CHECK-NEXT: %[[SHIFTED_OUT:.*]] = lshr i32 %[[LHS:.*]], %[[SHIFTED_OUT_WIDTH]]

  // CHECK-NEXT: %[[SHIFTED_OUT_NOT_SIGN:.*]] = lshr i32 %[[SHIFTED_OUT]], 1

  // CHECK-NEXT: %[[NO_OVERFLOW:.*]] = icmp eq i32 %[[SHIFTED_OUT_NOT_SIGN]], 0
  // CHECK-NEXT: br label %[[CONT_BB]]

  // CHECK:      [[CONT_BB]]:
  // CHECK-NEXT: %[[VALID_BASE:.*]] = phi i1 [ true, {{.*}} ], [ %[[NO_OVERFLOW]], %[[CHECK_BB]] ]
  // CHECK-NEXT: %[[VALID:.*]] = and i1 %[[RHS_INBOUNDS]], %[[VALID_BASE]]
  // CHECK-NEXT: br i1 %[[VALID]]

  // CHECK: call void @__ubsan_handle_shift_out_of_bounds
  // CHECK-NOT: call void @__ubsan_handle_shift_out_of_bounds

  // CHECK: %[[RET:.*]] = shl i32 %[[LHS]], %[[RHS]]
  // CHECK-NEXT: ret i32 %[[RET]]
  return a << b;
}
