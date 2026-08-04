// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s \
// RUN:   --check-prefix=NO_ANCESTOR

// A unit that compiles a submodule alone defines that submodule and nothing
// else. The ancestor module is not described here at all, because the separate
// module procedure is scoped in the submodule that defines it. A module that is
// merely used stays a declaration.

module {
  // Marks `subkid` as compiled here, as lowering does for every submodule.
  fir.module_debug_imports "subkid" in "subpar" {
  } loc(#loc0)
  func.func @_QMsubparPhello() attributes {fir.defining_submodule = "subkid"} {
    return
  } loc(#loc1)
  func.func private @_QMelsewherePthere() loc(#loc2)
}
#loc0 = loc("kid.f90":1:1)
#loc1 = loc("kid.f90":3:3)
#loc2 = loc("kid.f90":6:3)

// CHECK-DAG: #[[CU:.*]] = #llvm.di_compile_unit<{{.*}}>
// CHECK-DAG: #[[SUBKID:.*]] = #llvm.di_module<{{.*}}scope = #[[CU]], name = "subpar.subkid"{{.*}}>
// CHECK-DAG: #[[ELSEWHERE:.*]] = #llvm.di_module<name = "elsewhere", isDecl = true>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SUBKID]], name = "hello", linkageName = "_QMsubparPhello"{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[ELSEWHERE]], name = "there", linkageName = "_QMelsewherePthere"{{.*}}>

// Nothing describes the ancestor module on its own.
// NO_ANCESTOR-NOT: name = "subpar"
