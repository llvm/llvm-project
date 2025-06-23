// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --mlir-print-debuginfo %s | FileCheck %s

module {
  fir.global @_QMhelperEgli : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc3)
  fir.global @_QMhelperEglr : f32 {
    %0 = fir.zero_bits f32
    fir.has_value %0 : f32
  } loc(#loc4)
}
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 32, encoding = DW_ATE_signed>
#di_basic_type1 = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>

#di_file = #llvm.di_file<"test.f90" in "">
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>

#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang version 19.0.0 (/home/haqadeer/work/llvm-project/flang 5d5c73cad421bdca6e43e1cc10704ff160f1a33e)", isOptimized = false, emissionKind = Full>
#di_module = #llvm.di_module<file = #di_file, scope = #di_compile_unit, name = "helper", line = 11>
#di_global_variable = #llvm.di_global_variable<scope = #di_module, name = "gli", linkageName = "_QMhelperEgli", file = #di_file, line = 12, type = #di_basic_type, isDefined = true>
#di_global_variable1 = #llvm.di_global_variable<scope = #di_module, name = "glr", linkageName = "_QMhelperEglr", file = #di_file, line = 15, type = #di_basic_type1, isDefined = true>
#di_global_variable_expression = #llvm.di_global_variable_expression<var = #di_global_variable>
#di_global_variable_expression1 = #llvm.di_global_variable_expression<var = #di_global_variable1>

#loc1 = loc("test.f90":12:11)
#loc2 = loc("test.f90":15:8)
#loc3 = loc(fused<[#di_global_variable_expression]>[#loc1])
#loc4 = loc(fused<[#di_global_variable_expression1]>[#loc2])


// CHECK-DAG: #[[GLI:.*]] = #llvm.di_global_variable<{{.*}}name = "gli", linkageName = "_QMhelperEgli"{{.*}}>
// CHECK-DAG: #[[GLR:.*]] = #llvm.di_global_variable<{{.*}}name = "glr", linkageName = "_QMhelperEglr"{{.*}}>
// CHECK-DAG: #[[GLIE:.*]] = #llvm.di_global_variable_expression<var = #[[GLI]]>
// CHECK-DAG: #[[GLRE:.*]] = #llvm.di_global_variable_expression<var = #[[GLR]]>
// CHECK-DAG: llvm.mlir.global{{.*}}@_QMhelperEgli() {{{.*}}dbg_exprs = [#[[GLIE]]]}
// CHECK-DAG: llvm.mlir.global{{.*}}@_QMhelperEglr() {{{.*}}dbg_exprs = [#[[GLRE]]]}
