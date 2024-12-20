// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s

module @mod1 attributes {gpu.container} {
  gpu.module @gpu1 {
    fir.global linkonce @_QMtest_dinitE.dt.tseq constant : i8

    func.func @embox1(%arg0: !fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) {
      %0 = fir.embox %arg0() : (!fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
      return
    }
  }
}

// CHECK-LABEL: gpu.module @gpu1
// CHECK: llvm.mlir.global linkonce constant @_QMtest_dinitEXdtXtseq
// CHECK: llvm.mlir.addressof @_QMtest_dinitEXdtXtseq : !llvm.ptr

