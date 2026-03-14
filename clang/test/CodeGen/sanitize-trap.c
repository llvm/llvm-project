// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s -fsanitize=signed-integer-overflow,integer-divide-by-zero -fsanitize-trap=integer-divide-by-zero | FileCheck %s

int f(int x, int y) {
  // CHECK: %[[B1:.*]] = icmp ne i32 %[[D:.*]], 0
  // CHECK: %[[B2:.*]] = icmp ne i32 %[[N:.*]], -2147483648
  // CHECK: %[[B3:.*]] = icmp ne i32 %[[D]], -1
  // CHECK: %[[B4:.*]] = or i1 %[[B2]], %[[B3]]
  // CHECK: br i1 %[[B1]], label %[[L1:[0-9a-z_.]*]], label %[[L2:[0-9a-z_.]*]]

  // {{^|:}} used to match both Debug form of the captured label
  // cont:
  // and Release form
  // ; <label>:14
  // But avoids false matches inside other numbers such as [114 x i8].
  // CHECK: {{^|:}}[[L2]]
  // CHECK-NEXT: call void @llvm.ubsantrap(i8 3)
  // CHECK-NEXT: unreachable

  // CHECK: {{^|:}}[[L1]]
  // CHECK-NEXT: br i1 %[[B4]], label %[[L3:[0-9a-z_.]*]], label %[[L4:[0-9a-z_.]*]]

  // CHECK: {{^|:}}[[L4]]
  // CHECK-NEXT: zext
  // CHECK-NEXT: zext
  // CHECK-NEXT: __ubsan_handle_divrem_overflow

  // CHECK: {{^|:}}[[L3]]
  // CHECK-NEXT: sdiv i32 %[[N]], %[[D]]
  return x / y;
}
