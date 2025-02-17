// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func private @_FortranAioBeginExternalListOutput(i8) -> !fir.ref<i8> loc(#loc1)
}
#loc1 = loc("test.f90":5:1)

// CHECK: #[[INT8_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 8, encoding = DW_ATE_signed>
// CHECK: #[[REF_TY:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "", baseType = #[[INT8_TY]]{{.*}}>
// CHECK: #llvm.di_subroutine_type<{{.*}}types = #[[REF_TY]], #[[INT8_TY]]>
