
// RUN: fir-opt --add-debug-foundation --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: system-linux

// Test that there are no changes to a function with existed fused loc debug
module attributes {} {
  func.func @_QPs1() {
    return loc(#loc1)
  } loc(#loc2)
} loc(#loc)
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "void", encoding = DW_ATE_address>
#di_file = #llvm.di_file<"simple.f90" in "/home/user01/llvm-project/build_release">
#loc = loc("/home/user01/llvm-project/build_release/simple.f90":0:0)
#loc1 = loc("/home/user01/llvm-project/build_release/simple.f90":1:1)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_basic_type, #di_basic_type>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "_QPs1", linkageName = "_QPs1", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = Definition, type = #di_subroutine_type>
#loc2 = loc(fused<#di_subprogram>[#loc1])

// CHECK: #loc = loc("/home/user01/llvm-project/build_release/simple.f90":0:0)
// CHECK: #loc1 = loc("/home/user01/llvm-project/build_release/simple.f90":1:1)
// CHECK: #di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "_QPs1", linkageName = "_QPs1", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = Definition, type = #di_subroutine_type>
// CHECK: #loc2 = loc(fused<#di_subprogram>[#loc1])
