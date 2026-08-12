// RUN: not inter-opt %s --inter-import-llvm --inter-convert-llvm-to-xw 2>&1 | FileCheck %s

module {
  llvm.func spir_kernelcc @bad(%lhs: !llvm.ptr<1>, %rhs: !llvm.ptr<1>) {
    %value = llvm.icmp "ult" %lhs, %rhs : !llvm.ptr<1>
    llvm.return
  }
}

// CHECK: pointer comparison predicate must be eq or ne
