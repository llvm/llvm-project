// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_data(void) {
  // CHECK: cir.func @acc_data() {

#pragma acc data default(none)
  {
    int i = 0;
    ++i;
  }
  // CHECK-NEXT: acc.data {
  // CHECK-NEXT: cir.alloca
  // CHECK-NEXT: cir.const
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: cir.load
  // CHECK-NEXT: cir.unary
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(present)
  {
    int i = 0;
    ++i;
  }
  // CHECK-NEXT: acc.data {
  // CHECK-NEXT: cir.alloca
  // CHECK-NEXT: cir.const
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: cir.load
  // CHECK-NEXT: cir.unary
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

  // CHECK-NEXT: cir.return
}
