// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s -o - | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QMhelperPmod_sub(%arg0: !fir.ref<i32> {fir.bindc_name = "a"} ) {
    return
  } loc(#loc1)
  func.func private @_QMhelperFmod_subPchild1(%arg0: !fir.ref<i32> {fir.bindc_name = "b"} ) attributes {fir.host_symbol = @_QMhelperPmod_sub, llvm.linkage = #llvm.linkage<internal>} {
    return
  } loc(#loc2)
  func.func @global_sub_(%arg0: !fir.ref<i32> {fir.bindc_name = "n"} ) attributes {fir.internal_name = "_QPglobal_sub"} {
    return
  } loc(#loc3)
  func.func private @_QFglobal_subPchild2(%arg0: !fir.ref<i32> {fir.bindc_name = "c"}) attributes {fir.host_symbol = @global_sub_, llvm.linkage = #llvm.linkage<internal>} {
    return
  } loc(#loc4)
}

#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":15:1)
#loc3 = loc("test.f90":25:1)
#loc4 = loc("test.f90":35:1)

// CHECK-DAG: #[[SP1:.*]] = #llvm.di_subprogram<{{.*}}name = "mod_sub"{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SP1]], name = "child1"{{.*}}>
// CHECK-DAG: #[[SP2:.*]] = #llvm.di_subprogram<{{.*}}linkageName = "global_sub_"{{.*}}>
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[SP2]], name = "child2"{{.*}}>
