// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_kernels(void) {
  // CHECK: cir.func @acc_kernels() {
#pragma acc kernels
  {}

  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT:}

#pragma acc kernels default(none)
  {}
  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc kernels default(present)
  {}
  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

#pragma acc kernels
  while(1){}
  // CHECK-NEXT: acc.kernels {
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
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT:}

  // CHECK-NEXT: cir.return
}
