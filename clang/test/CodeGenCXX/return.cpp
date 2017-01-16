// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -O -o - %s | FileCheck %s --check-prefix=CHECK-OPT

// CHECK:     @_Z9no_return
// CHECK-OPT: @_Z9no_return
int no_return() {
  // CHECK:      call void @llvm.trap
  // CHECK-NEXT: unreachable

  // CHECK-OPT-NOT: call void @llvm.trap
  // CHECK-OPT:     unreachable
}
