// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s


module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.global @_QMhelperEgli : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc1)
  fir.global @_QMhelperEglr : f32 {
    %0 = fir.zero_bits f32
    fir.has_value %0 : f32
  } loc(#loc2)
  func.func @_QMhelperPtest() {
    %c67_i32 = arith.constant 67 : i32
    %cst = arith.constant 1.234000e+01 : f32
    %0 = fir.address_of(@_QMhelperEgli) : !fir.ref<i32>
    %1 = fir.address_of(@_QMhelperEglr) : !fir.ref<f32>
    fir.store %cst to %1 : !fir.ref<f32>
    fir.store %c67_i32 to %0 : !fir.ref<i32>
    return
  } loc(#loc3)
}
#loc1 = loc("test.f90":12:11)
#loc2 = loc("test.f90":15:8)
#loc3 = loc("test.f90":20:5)

// CHECK-DAG: #[[I4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK-DAG: #[[R4:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}>
// CHECK-DAG: #[[MOD:.*]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "helper"{{.*}}>
// CHECK-DAG: #[[LOC1:.*]] = loc("{{.*}}test.f90":12{{.*}})
// CHECK-DAG: #[[GLI:.*]] = #llvm.di_global_variable<scope = #[[MOD]], name = "gli", linkageName = "_QMhelperEgli"{{.*}}line = 12, type = #[[I4]], isDefined = true>
// CHECK-DAG: #[[LOC2:.*]] = loc("{{.*}}test.f90":15{{.*}})
// CHECK-DAG: #[[GLR:.*]] = #llvm.di_global_variable<scope = #[[MOD]], name = "glr", linkageName = "_QMhelperEglr"{{.*}}line = 15, type = #[[R4]], isDefined = true>
// CHECK-DAG: #[[LOC3:.*]] = loc("{{.*}}test.f90":20{{.*}})
// CHECK-DAG: #[[TEST:.*]] = #llvm.di_subprogram<{{.*}}compileUnit = #[[CU]], scope = #[[MOD]], name = "test", linkageName = "_QMhelperPtest"{{.*}}line = 20, scopeLine = 20{{.*}}>
// CHECK-DAG: loc(fused<#[[GLI]]>[#[[LOC1]]])
// CHECK-DAG: loc(fused<#[[GLR]]>[#[[LOC2]]])
// CHECK-DAG: loc(fused<#[[TEST]]>[#[[LOC3]]])

