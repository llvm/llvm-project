// A signed division or remainder only overflows for INT_MIN / -1, so a
// constant operand that rules that out needs no signed-integer-overflow check.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsanitize=signed-integer-overflow -emit-llvm -o - %s | FileCheck %s

int div_by_4(int a) { return a / 4; }
// CHECK-LABEL: define {{.*}} @div_by_4
// CHECK-NOT: __ubsan_handle_divrem_overflow
// CHECK: ret i32

int div_by_minus_1(int a) { return a / -1; }
// CHECK-LABEL: define {{.*}} @div_by_minus_1
// CHECK: call void @__ubsan_handle_divrem_overflow

int rem_of_7(int b) { return 7 % b; }
// CHECK-LABEL: define {{.*}} @rem_of_7
// CHECK-NOT: __ubsan_handle_divrem_overflow
// CHECK: ret i32

int rem_of_min(int b) { return (-2147483647 - 1) % b; }
// CHECK-LABEL: define {{.*}} @rem_of_min
// CHECK: call void @__ubsan_handle_divrem_overflow
