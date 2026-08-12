// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad(%pointer: !llvm.ptr<5>) {
    llvm.return
  }
}

// CHECK: pointer address space 5 has no XW mapping
