// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test a unit that compiles a submodule alone. Only that submodule is defined
// here; the ancestor it imports and a module that is merely used are both
// declarations.

module {
  // Marks `subkid` as compiled here, as lowering does for every submodule.
  fir.module_debug_imports "subkid" in "subpar" parent "subpar" {
  } loc(#loc0)
  func.func @_QMsubparPhello() attributes {fir.defining_submodule = "subkid"} {
    return
  } loc(#loc1)
  func.func private @_QMelsewherePthere() loc(#loc2)
}
#loc0 = loc("kid.f90":1:1)
#loc1 = loc("kid.f90":3:3)
#loc2 = loc("kid.f90":6:3)

// CHECK-DAG: #[[CU_SELF:.*]] = #llvm.di_compile_unit<recId = [[RECID:.+]], isRecSelf = true>
// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<recId = [[RECID]], id = {{.*}}>
// CHECK-DAG: #[[SUBPAR:.*]] = #llvm.di_module<name = "subpar", isDecl = true>
// CHECK-DAG: #[[ELSEWHERE:.*]] = #llvm.di_module<name = "elsewhere", isDecl = true>

// CHECK-DAG: #[[SUBKID:.*]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "subpar.subkid"{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SUBKID]], name = "hello", linkageName = "_QMsubparPhello"{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[ELSEWHERE]], name = "there", linkageName = "_QMelsewherePthere"{{.*}}>

// CHECK-DAG: #[[SUBKID_S:.*]] = #llvm.di_module<{{.*}}scope = #[[CU_SELF]], name = "subpar.subkid"{{.*}}>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[SUBKID_S]], entity = #[[SUBPAR]]
