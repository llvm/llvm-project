// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func private @_FortranAAssign(i8) -> none loc(#loc1)
  func.func private @foo() -> !fir.ref<none>  loc(#loc2)
}
#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":15:1)

// CHECK: #[[INT8_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer", sizeInBits = 8, encoding = DW_ATE_signed>
// CHECK: #[[NONE_TY:.*]] = #llvm.di_null_type
// CHECK: #[[REFNONE_TY:.*]] = #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "", baseType = #di_null_type{{.*}}>
// CHECK: #llvm.di_subroutine_type<{{.*}}types = #[[NONE_TY]], #[[INT8_TY]]>
// CHECK: #llvm.di_subroutine_type<{{.*}}types = #[[REFNONE_TY]]>
