// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_shutdown(void) {
  // CHECK: cir.func @acc_shutdown() {
#pragma acc shutdown
// CHECK-NEXT: acc.shutdown loc(#{{[a-zA-Z0-9]+}}){{$}}
}
