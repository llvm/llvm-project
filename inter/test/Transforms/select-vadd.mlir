// Straight-line selection: vadd lowers to the prologue, gid computation,
// A64 loads, a store, and EOT.
// RUN: inter-opt %s --inter-normalize-cf --inter-select-to-machine | FileCheck %s

module {
  // CHECK: func.func @vadd
  // CHECK-SAME: xemachine.target = #xemachine.target<chip = "bmg">
  llvm.func spir_kernelcc @vadd(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>,
                                %out: !llvm.ptr<1>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%c0) : (i32) -> i64
    %pa = llvm.getelementptr %a[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %va = llvm.load %pa : !llvm.ptr<1> -> i32
    %pb = llvm.getelementptr %b[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %vb = llvm.load %pb : !llvm.ptr<1> -> i32
    %sum = llvm.add %vb, %va : i32
    %po = llvm.getelementptr %out[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %sum, %po : i32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64
}

// Prologue: blob base and the two payload loads.
// CHECK: xemachine.and
// CHECK-COUNT-2: xemachine.load_block_a32
// CHECK: xemachine.sync allrd
// gid: mul into acc, add3 over local ids.
// CHECK: xemachine.mul
// CHECK: xemachine.add3
// Two A64 loads and one store with a data payload.
// CHECK-COUNT-2: xemachine.load_a64
// CHECK: xemachine.store_a64 {{.*}}data
// EOT via the gateway.
// CHECK: xemachine.eot
