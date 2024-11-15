// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func private @fn1(!fir.ref<tuple<f64, f64>>)
  func.func private @_FortranAioOutputDerivedType(!fir.ref<tuple<>>)
}

// CHECK: #[[F64:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 64, encoding = DW_ATE_float>
// CHECK: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}>
// CHECK: #[[DTY1:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "", baseType = #[[F64]], sizeInBits = 64, alignInBits = {{.*}}>
// CHECK: #[[DTY2:.*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "", baseType = #[[F64]], sizeInBits = 64, alignInBits = {{.*}}, offsetInBits = {{.*}}>
// CHECK: #[[COM_TY1:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "", file = #{{.*}}, scope = #[[CU]]{{.*}}elements = #[[DTY1]], #[[DTY2]]>
// CHECK: #[[COM_TY2:.*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "", file = #{{.*}}, scope = #[[CU]]>
// CHECK: #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_null_type, #[[COM_TY1]]>
// CHECK: #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_null_type, #[[COM_TY2]]>
