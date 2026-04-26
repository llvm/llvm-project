// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// `fir.module_debug_imports` supplies transitive USE metadata; AddDebugInfo
// flattens it onto the subprogram.

module {
  fir.global @_QMchildEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.global @_QMparentEy : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  fir.module_debug_imports "parent" {
    fir.use_stmt "child"
  }
  func.func @_QQmain() attributes {fir.bindc_name = "MAIN_USE"} {
    fir.use_stmt "parent"
    return
  } loc(#loc1)
}
#loc1 = loc("m.f90":1:1)

// CHECK-DAG: #[[MOD_CHILD:.+]] = #llvm.di_module<{{.*}}name = "child"{{.*}}>
// CHECK-DAG: #[[MOD_PARENT:.+]] = #llvm.di_module<{{.*}}name = "parent"{{.*}}>

// CHECK-DAG: #[[SP_REC:.+]] = #llvm.di_subprogram<recId = distinct[[[RECID:[0-9]+]]]<>, isRecSelf = true{{.*}}name = "MAIN_USE"
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD_CHILD]],{{.*}}>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SP_REC]], entity = #[[MOD_PARENT]],{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<recId = distinct[[[RECID]]]<>{{.*}}name = "MAIN_USE"{{.*}}retainedNodes = {{.+}}>
