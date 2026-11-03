// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad(%pointer: !llvm.ptr<5>) {
    llvm.return
  }
}

// CHECK: kernel argument 0 has address space 5 outside the kernel ABI
