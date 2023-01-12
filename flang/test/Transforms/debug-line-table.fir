
// RUN: fir-opt --add-debug-foundation --mlir-print-debuginfo %s | FileCheck %s

module attributes { fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @_QPsb() {
    return loc(#loc_sb)
  } loc(#loc_sb)
} loc(#loc_module)
#loc_module = loc("./simple.f90":1:1)
#loc_sb = loc("./simple.f90":2:1)

// CHECK: module attributes
// CHECK:   func.func @[[SB_NAME:.*]]() {
// CHECK:     return loc(#[[SB_LOC:.*]])
// CHECK:   } loc(#[[FUSED_SB_LOC:.*]])
// CHECK: } loc(#[[MODULE_LOC:.*]])
// CHECK: #di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "void", encoding = DW_ATE_address>
// CHECK: #di_file = #llvm.di_file<"[[FILE_NAME:.*]]" in "[[DIR_NAME:.*]]">
// CHECK: #[[MODULE_LOC]] = loc("[[DIR_NAME]]/[[FILE_NAME]]":1:1)
// CHECK: #[[SB_LOC]] = loc("./simple.f90":2:1)
// CHECK: #di_compile_unit = #llvm.di_compile_unit<sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = LineTablesOnly>
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, resultType = #di_basic_type, argumentTypes = #di_basic_type>
// CHECK: #di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "[[SB_NAME]]", linkageName = "[[SB_NAME]]", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = Definition, type = #di_subroutine_type>
// CHECK: #[[FUSED_SB_LOC]] = loc(fused<#di_subprogram>[#[[SB_LOC]]])
