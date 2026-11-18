// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s --check-prefix=NO_DECL

module {
  // This constant is defined here and should be described.
  fir.global @_QMhelperECdefined constant : f32 {
    %cst = arith.constant 3.14159274 : f32
    fir.has_value %cst : f32
  } loc(#loc1)
  // This constant is only a declaration and should not be described.
  fir.global @_QMhelperECdeclared constant : f32 loc(#loc2)
}
#loc1 = loc("test.f90":8:26)
#loc2 = loc("test.f90":9:26)

// CHECK-DAG: #[[MOD:.*]] = #llvm.di_module<{{.*}}name = "helper"{{.*}}>
// CHECK-DAG: #llvm.di_global_variable<scope = #[[MOD]], name = "defined"{{.*}}>

// NO_DECL-NOT: name = "declared"
