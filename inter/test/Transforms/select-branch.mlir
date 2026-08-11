// Branch selection: llvm branches lift to scf.if, then lower to cmp +
// exec_if with merged results.
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' | FileCheck %s

module {
  llvm.func spir_kernelcc @branch_kernel(%out: !llvm.ptr<1>, %a: !llvm.ptr<1>,
                                         %b: !llvm.ptr<1>, %t: i32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%c0) : (i32) -> i64
    %pa = llvm.getelementptr %a[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %va = llvm.load %pa : !llvm.ptr<1> -> i32
    %cond = llvm.icmp "ugt" %va, %t : i32
    llvm.cond_br %cond, ^then, ^merge(%c1 : i32)
  ^then:
    %pb = llvm.getelementptr %b[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %vb = llvm.load %pb : !llvm.ptr<1> -> i32
    llvm.br ^merge(%vb : i32)
  ^merge(%v: i32):
    %r = llvm.add %v, %va : i32
    %po = llvm.getelementptr %out[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %r, %po : i32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64
}

// The condition comes from a load: divergent.
// CHECK: xemachine.cmp
// CHECK: [[IF:%.*]]:2 = xemachine.exec_if
// Merge movs into the result register inside both regions.
// CHECK: xemachine.mov {{.*}}-> !xemachine.reg<32,
// CHECK: xemachine.yield {{.*}} : !xemachine.reg<32,{{.*}}>, !xemachine.mem.token
// CHECK: } otherwise {
// CHECK: xemachine.mov {{.*}}-> !xemachine.reg<32,
// CHECK: xemachine.yield {{.*}} : !xemachine.reg<32,{{.*}}>, !xemachine.mem.token
// CHECK: } -> !xemachine.reg<32,{{.*}}>, !xemachine.mem.token
// CHECK: xemachine.store_a64 {{.*}} dep [[IF]]#1
