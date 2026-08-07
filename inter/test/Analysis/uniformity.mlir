// Uniformity analysis drives uniform_if vs exec_if selection.
// RUN: inter-opt %s --inter-normalize-cf --lift-cf-to-scf --inter-select-to-machine | FileCheck %s

module {
  // Condition reads only a kernel arg and a constant: uniform.
  // CHECK-LABEL: func.func @k_uniform
  // CHECK: xemachine.uniform_if
  // CHECK-NOT: xemachine.exec_if
  llvm.func spir_kernelcc @k_uniform(%out: !llvm.ptr<1>, %a: !llvm.ptr<1>,
                                     %b: !llvm.ptr<1>, %t: i32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%c0) : (i32) -> i64
    %cond = llvm.icmp "ugt" %t, %c3 : i32
    llvm.cond_br %cond, ^then, ^else
  ^then:
    %pa = llvm.getelementptr %a[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %va = llvm.load %pa : !llvm.ptr<1> -> i32
    llvm.br ^merge(%va : i32)
  ^else:
    %pb = llvm.getelementptr %b[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %vb = llvm.load %pb : !llvm.ptr<1> -> i32
    llvm.br ^merge(%vb : i32)
  ^merge(%v: i32):
    %po = llvm.getelementptr %out[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %v, %po : i32, !llvm.ptr<1>
    llvm.return
  }

  // Condition reads a per-lane load: divergent.
  // CHECK-LABEL: func.func @k_varying
  // CHECK: xemachine.exec_if
  // CHECK-NOT: xemachine.uniform_if
  llvm.func spir_kernelcc @k_varying(%out: !llvm.ptr<1>, %a: !llvm.ptr<1>,
                                     %b: !llvm.ptr<1>, %t: i32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%c0) : (i32) -> i64
    %pa = llvm.getelementptr %a[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %va = llvm.load %pa : !llvm.ptr<1> -> i32
    %cond = llvm.icmp "ugt" %va, %t : i32
    llvm.cond_br %cond, ^then, ^else
  ^then:
    %pb = llvm.getelementptr %b[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %vb = llvm.load %pb : !llvm.ptr<1> -> i32
    llvm.br ^merge(%vb : i32)
  ^else:
    llvm.br ^merge(%va : i32)
  ^merge(%v: i32):
    %po = llvm.getelementptr %out[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %v, %po : i32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64
}
