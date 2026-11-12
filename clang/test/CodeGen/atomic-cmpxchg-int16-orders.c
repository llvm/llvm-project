// RUN: %clang_cc1 %s -emit-llvm -o - -triple=msp430 | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=avr -target-cpu atmega2560 | FileCheck %s

// On these targets int has 16 bits, and the orders known only at run time are
// switched on as 32-bit values, with 32-bit cases.

// CHECK-LABEL: @run_time_orders(
// CHECK:      [[SUCCESS:%.*]] = zext i16 {{%.*}} to i32
// CHECK-NEXT: switch i32 [[SUCCESS]], label %{{.*}} [
// CHECK:      [[FAILURE:%.*]] = zext i16 {{%.*}} to i32
// CHECK-NEXT: switch i32 [[FAILURE]], label %{{.*}} [
// CHECK-NEXT:   i32 1, label %[[ACQUIRE:.*]]
// CHECK-NEXT:   i32 2, label %[[ACQUIRE]]
// CHECK-NEXT:   i32 5, label %{{.*}}
// CHECK-NEXT: ]
_Bool run_time_orders(int *p, int *e, int d, int s, int f) {
  return __atomic_compare_exchange_n(p, e, d, 0, s, f);
}
