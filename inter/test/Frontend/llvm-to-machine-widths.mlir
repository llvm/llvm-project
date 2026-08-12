// RUN: inter-opt %s '--inter-import-llvm=simd-width=8' \
// RUN:   --inter-convert-llvm-to-xw --inter-select-to-machine | \
// RUN:   FileCheck %s --check-prefix=WIDTH8
// RUN: inter-opt %s '--inter-import-llvm=simd-width=16' \
// RUN:   --inter-convert-llvm-to-xw --inter-select-to-machine | \
// RUN:   FileCheck %s --check-prefix=WIDTH16
// RUN: inter-opt %s '--inter-import-llvm=simd-width=32' \
// RUN:   --inter-convert-llvm-to-xw --inter-select-to-machine | \
// RUN:   FileCheck %s --check-prefix=WIDTH32

module {
  llvm.mlir.global internal @first() {addr_space = 3 : i32} : i32
  llvm.mlir.global internal @second() {
    addr_space = 3 : i32, alignment = 16 : i64} : i32

  llvm.func spir_kernelcc @supported(%value: i32) {
    %first = llvm.mlir.addressof @first : !llvm.ptr<3>
    %second = llvm.mlir.addressof @second : !llvm.ptr<3>
    %one = llvm.mlir.constant(1 : i32) : i32
    %axis = llvm.mlir.constant(0 : i32) : i32
    %local_id = llvm.call spir_funccc @_Z12get_local_idj(%axis)
        : (i32) -> i64
    %sum = llvm.add %value, %one : i32
    %wide = llvm.sext %sum : i32 to i64
    llvm.store %sum, %first : i32, !llvm.ptr<3>
    llvm.store %sum, %second : i32, !llvm.ptr<3>
    llvm.return
  }
  llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64
}

// WIDTH8-LABEL: func.func @supported
// WIDTH8-SAME: xemachine.simd_size = 8
// WIDTH8-SAME: xemachine.slm_size = 20
// WIDTH8: xemachine.add
// WIDTH8: src0Type = i16
// WIDTH8: signedSource
// WIDTH8: xemachine.eot

// WIDTH16-LABEL: func.func @supported
// WIDTH16-SAME: xemachine.simd_size = 16
// WIDTH16-SAME: xemachine.slm_size = 20
// WIDTH16: xemachine.add
// WIDTH16: src0Type = i16
// WIDTH16: signedSource
// WIDTH16: xemachine.eot

// WIDTH32-LABEL: func.func @supported
// WIDTH32-SAME: xemachine.simd_size = 32
// WIDTH32-SAME: xemachine.slm_size = 20
// WIDTH32: xemachine.add
// WIDTH32: src0Type = i16
// WIDTH32: signedSource
// WIDTH32: xemachine.eot
