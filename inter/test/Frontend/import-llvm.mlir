// RUN: inter-opt %s --inter-import-llvm | FileCheck %s

module {
  llvm.func spir_kernelcc @branch_args(%condition: i1, %value: i32) {
    llvm.cond_br %condition, ^then, ^merge(%value : i32)
  ^then:
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^merge(%one : i32)
  ^merge(%result: i32):
    llvm.return
  }
}

// CHECK-LABEL: func.func @branch_args
// CHECK-SAME: attributes {
// CHECK-SAME: xw.kernel
// CHECK: cf.cond_br {{.*}}, ^bb1, ^bb2({{.*}} : i32)
// CHECK: cf.br ^bb2({{.*}} : i32)
// CHECK: return
