// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_init(void) {
  // CHECK: cir.func @acc_init() {
#pragma acc init
// CHECK-NEXT: acc.init loc(#{{[a-zA-Z0-9]+}}){{$}}
}
