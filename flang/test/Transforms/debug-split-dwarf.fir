// RUN: fir-opt --add-debug-info="split-dwarf-file=test.dwo"  \
// RUN:         --mlir-print-debuginfo %s -o - | FileCheck %s

module {
  func.func @test() {
    return
  } loc(#loc1)
}
#loc1 = loc("test.f90":15:1)

// CHECK: llvm.di_compile_unit
// CHECK-SAME: splitDebugFilename = "test.dwo"
