
// RUN: fir-opt --add-debug-foundation --mlir-print-debuginfo %s | FileCheck %s

module attributes { fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @_QPsb() {
    return loc(#loc_sb)
  } loc(#loc_sb)
  func.func private @decl() -> i32 loc(#loc_decl)
} loc(#loc_module)
#loc_module = loc("./simple.f90":1:1)
#loc_sb = loc("./simple.f90":2:1)
#loc_decl = loc("./simple.f90":10:1)

// CHECK: module attributes
// CHECK:   func.func @[[SB_NAME:.*]]() {
// CHECK:     return loc(#[[SB_LOC:.*]])
// CHECK:   } loc(#[[FUSED_SB_LOC:.*]])
// CHECK:   func.func private @[[DECL_NAME:.*]]() -> i32 loc(#[[FUSED_DECL_LOC:.*]])
// CHECK: } loc(#[[MODULE_LOC:.*]])
// CHECK: #di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "void", encoding = DW_ATE_address>
// CHECK: #di_file = #llvm.di_file<"[[FILE_NAME:.*]]" in "[[DIR_NAME:.*]]">
// CHECK: #[[MODULE_LOC]] = loc("[[DIR_NAME]]/[[FILE_NAME]]":1:1)
// CHECK: #[[SB_LOC]] = loc("./simple.f90":2:1)
// CHECK: #[[DECL_LOC:.*]] = loc("./simple.f90":10:1)
// CHECK: #di_compile_unit = #llvm.di_compile_unit<id = distinct[{{.*}}]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = LineTablesOnly>
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_basic_type, #di_basic_type>
// CHECK: #[[SB_SUBPROGRAM:.*]] = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "[[SB_NAME]]", linkageName = "[[SB_NAME]]", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
// CHECK: #[[DECL_SUBPROGRAM:.*]] = #llvm.di_subprogram<scope = #di_file, name = "[[DECL_NAME]]", linkageName = "[[DECL_NAME]]", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = Optimized, type = #di_subroutine_type>
// CHECK: #[[FUSED_SB_LOC]] = loc(fused<#[[SB_SUBPROGRAM]]>[#[[SB_LOC]]])
// CHECK: #[[FUSED_DECL_LOC]] = loc(fused<#[[DECL_SUBPROGRAM]]>[#[[DECL_LOC]]])
