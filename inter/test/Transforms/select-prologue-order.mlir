// The fixed dual-entry payload prologue must precede source-ordered machine
// operations even when the first thread-ID use occurs later in the kernel.
// RUN: inter-opt %s --inter-normalize-cf --inter-convert-calls --inter-convert-memory --inter-select-to-machine | FileCheck %s

module {
  llvm.func spir_kernelcc @atomic_before_id(%out: !llvm.ptr<1>,
                                            %counter: !llvm.ptr<1>) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %old = llvm.call spir_funccc @_Z10atomic_addPU3AS1Vjj(%counter, %one)
        : (!llvm.ptr<1>, i32) -> i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%zero) : (i32) -> i64
    %address = llvm.getelementptr %out[%gid]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %old, %address : i32, !llvm.ptr<1>
    llvm.return
  }

  llvm.func spir_funccc @_Z10atomic_addPU3AS1Vjj(!llvm.ptr<1>, i32) -> i32
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64
}

// CHECK-LABEL: func.func @atomic_before_id
// CHECK: [[R1:%.*]] = xemachine.archreg 1
// CHECK: xemachine.mov [[R1]] {{.*}}-> !xemachine.reg<16, 4>
// CHECK: xemachine.load_block_a32 {{.*}}words = 16
// CHECK: xemachine.sync allwr
// CHECK: xemachine.atomic_iadd_a64
