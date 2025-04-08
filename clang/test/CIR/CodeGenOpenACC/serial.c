// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_serial(void) {
  // CHECK: cir.func @acc_serial() {
#pragma acc serial
  {}

  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT:acc.yield
  // CHECK-NEXT:}

#pragma acc serial
  while(1){}
  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT: cir.scope {
  // CHECK-NEXT: cir.while {
  // CHECK-NEXT: %[[INT:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[CAST:.*]] = cir.cast(int_to_bool, %[[INT]] :
  // CHECK-NEXT: cir.condition(%[[CAST]])
  // CHECK-NEXT: } do {
  // CHECK-NEXT: cir.yield
  // cir.while do end:
  // CHECK-NEXT: }
  // cir.scope end:
  // CHECK-NEXT: }
  // CHECK-NEXT:acc.yield
  // CHECK-NEXT:}

  // CHECK-NEXT: cir.return
}
