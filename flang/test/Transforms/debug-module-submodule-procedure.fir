// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  // Marks `subkid` as compiled here, as lowering does for every submodule.
  fir.module_debug_imports "subkid" {
  } loc(#loc0)
  func.func @_QMsubparPhello() {
    return
  } loc(#loc1)
  func.func private @_QMelsewherePthere() loc(#loc2)
}
#loc0 = loc("kid.f90":1:1)
#loc1 = loc("kid.f90":3:3)
#loc2 = loc("kid.f90":6:3)

// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}>
// CHECK-DAG: #[[SUBPAR:.*]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "subpar"{{.*}}>
// CHECK-DAG: #[[ELSEWHERE:.*]] = #llvm.di_module<name = "elsewhere", isDecl = true>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SUBPAR]], name = "hello", linkageName = "_QMsubparPhello"{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[ELSEWHERE]], name = "there", linkageName = "_QMelsewherePthere"{{.*}}>
