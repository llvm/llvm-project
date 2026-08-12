// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.mlir.global internal @unsupported() {addr_space = 1 : i32} : i32
  llvm.func spir_kernelcc @bad() {
    %pointer = llvm.mlir.addressof @unsupported : !llvm.ptr<1>
    llvm.return
  }
}

// CHECK: only local-address-space LLVM globals are semantic allocations
