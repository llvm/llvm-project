// RUN: %clang_cc1 -fsanitize=shift-base -emit-llvm %s -o - -triple x86_64-linux-gnu -fwrapv | FileCheck %s

// CHECK-LABEL: @lsh_overflow
int lsh_overflow(int a, int b) {
  // CHECK-NOT: br
  // CHECK-NOT: call void @__ubsan_
  // CHECK-NOT: call void @llvm.trap
  
  // CHECK:      %[[RET:.*]] = shl i32
  // CHECK-NEXT: ret i32 %[[RET]]
  return a << b;
}
