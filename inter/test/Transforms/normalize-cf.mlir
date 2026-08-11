// inter-normalize-cf: llvm.func -> func.func, llvm branches -> cf branches.
// RUN: inter-opt %s --inter-normalize-cf | FileCheck %s

module attributes {
    llvm.data_layout = "e-p:64:64-p1:64:64-i64:64-G1"} {
  // CHECK: module attributes {llvm.data_layout = "e-p:64:64-p1:64:64-i64:64-G1"}
  // CHECK-NOT: llvm.func @k
  // CHECK: func.func @k(%{{.*}}: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias})
  // CHECK-SAME: attributes {
  // CHECK-SAME: marker = "preserved"
  // CHECK-SAME: xemachine.kernel
  // CHECK-SAME: xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>]
  // CHECK-SAME: xemachine.llvm_func_properties
  llvm.func spir_kernelcc @k(
      %arg0: !llvm.ptr<1> {llvm.align = 64 : i64, llvm.noalias})
      attributes {marker = "preserved", intel_reqd_sub_group_size = 16 : i32} {
    %c = llvm.mlir.constant(true) : i1
    // CHECK: cf.cond_br
    // CHECK-SAME: weights([80, 20])
    // CHECK-NOT: llvm.cond_br
    llvm.cond_br %c weights([80, 20]), ^bb1, ^bb2
  ^bb1:
    // CHECK: cf.br
    // CHECK-NOT: llvm.br
    llvm.br ^bb2
  ^bb2:
    // CHECK: return
    // CHECK-NOT: llvm.return
    llvm.return
  }
  // The declaration must survive untouched.
  // CHECK: llvm.func {{.*}}@declared(
  llvm.func spir_funccc @declared(i32) -> i64

  // CHECK-LABEL: func.func @layout
  // CHECK-SAME: xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>, #xemachine.kernel_arg<kind = by_value, offset = 32, size = 4>, #xemachine.kernel_arg<kind = by_value, offset = 40, size = 8>]
  llvm.func spir_kernelcc @layout(%pointer: !llvm.ptr<1>, %scalar: i32,
                                  %wide: i64) {
    llvm.return
  }
}
