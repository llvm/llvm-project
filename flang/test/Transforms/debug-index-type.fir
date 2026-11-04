// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  func.func private @str(%arg0: index) -> i32 loc(#loc1)
}
#loc1 = loc("test.f90":5:1)

// CHECK: #[[INT32_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer(kind=4)", sizeInBits = 32, encoding = DW_ATE_signed>
// CHECK: #[[INT64_TY:.*]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer(kind=8)", sizeInBits = 64, encoding = DW_ATE_signed>
// CHECK: #llvm.di_subroutine_type<{{.*}}types = #[[INT32_TY]], #[[INT64_TY]]>
