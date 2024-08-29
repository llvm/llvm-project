
// RUN: fir-opt --add-debug-info="debug-level=LineTablesOnly" --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: system-linux

// Test for included functions that have a different debug location than the current file
module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPsinc() {
    return loc(#loc2)
  } loc(#loc1)
  func.func @_QQmain() {
    fir.call @_QPsinc() fastmath<contract> : () -> () loc(#loc4)
    return loc(#loc5)
  } loc(#loc3)
} loc(#loc)
#loc = loc("/home/user01/llvm-project/build_release/simple.f90":0:0)
#loc1 = loc("/home/user01/llvm-project/build_release/inc.f90":1:1)
#loc2 = loc("/home/user01/llvm-project/build_release/inc.f90":2:1)
#loc3 = loc("/home/user01/llvm-project/build_release/simple.f90":3:1)
#loc4 = loc("/home/user01/llvm-project/build_release/simple.f90":4:3)
#loc5 = loc("/home/user01/llvm-project/build_release/simple.f90":5:1)

// CHECK: module
// CHECK:   func.func @_QPsinc() {
// CHECK:   } loc(#[[FUSED_LOC_INC_FILE:.*]])
// CHECK:   func.func @_QQmain() {
// CHECK:   } loc(#[[FUSED_LOC_FILE:.*]])
// CHECK: } loc(#[[MODULE_LOC:.*]])
// CHECK: #[[DI_FILE:.*]] = #llvm.di_file<"simple.f90" in "[[DIR:.*]]">
// CHECK: #[[DI_INC_FILE:.*]] = #llvm.di_file<"inc.f90" in "[[DIR]]">
// CHECK: #[[MODULE_LOC]] = loc("{{.*}}simple.f90":0:0)
// CHECK: #[[LOC_INC_FILE:.*]] = loc("{{.*}}inc.f90":1:1)
// CHECK: #[[LOC_FILE:.*]] = loc("{{.*}}simple.f90":3:1)
// CHECK: #[[DI_CU:.*]] = #llvm.di_compile_unit<id = distinct[{{.*}}]<>, sourceLanguage = DW_LANG_Fortran95, file = #[[DI_FILE]], producer = "{{.*}}flang{{.*}}", isOptimized = false, emissionKind = LineTablesOnly>
// CHECK: #[[DI_SP_INC:.*]] = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #[[DI_CU]], scope = #[[DI_FILE]], name = "sinc", linkageName = "_QPsinc", file = #[[DI_INC_FILE]], {{.*}}>
// CHECK: #[[DI_SP:.*]] = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #[[DI_CU]], scope = #[[DI_FILE]], name = "_QQmain", linkageName = "_QQmain", file = #[[DI_FILE]], {{.*}}>
// CHECK: #[[FUSED_LOC_INC_FILE]] = loc(fused<#[[DI_SP_INC]]>[#[[LOC_INC_FILE]]])
// CHECK: #[[FUSED_LOC_FILE]] = loc(fused<#[[DI_SP]]>[#[[LOC_FILE]]])
