// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test that a named constant (Fortran PARAMETER) declared in a module is
// described in the scope of that module.

module {
  fir.global @_QMhelperECpi constant : f32 {
    %cst = arith.constant 3.14159274 : f32
    fir.has_value %cst : f32
  } loc(#loc1)
  func.func @_QMhelperPtest() {
    return
  } loc(#loc2)
}
#loc1 = loc("test.f90":8:26)
#loc2 = loc("test.f90":12:5)

// CHECK-DAG: #[[R4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=4)", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}>
// CHECK-DAG: #[[MOD:.*]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "helper"{{.*}}>
// CHECK-DAG: #[[PI:.*]] = #llvm.di_global_variable<scope = #[[MOD]], name = "pi", linkageName = "_QMhelperECpi"{{.*}}line = 8, type = #[[R4]], isDefined = true>
// CHECK-DAG: #[[PIE:.*]] = #llvm.di_global_variable_expression<var = #[[PI]]>
