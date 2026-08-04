// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// A submodule is named after the module at the root of its ancestry, which the
// uniqued name of a module entity also records.

module {
  fir.module_debug_imports "shapes" {
  } loc(#loc_shapes)
  fir.global @_QMshapesEparent_var : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_parent_var)

  fir.module_debug_imports "impl" in "shapes" {
  } loc(#loc_impl)
  fir.global @_QMshapesSimplEsub_var : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_sub_var)

  fir.module_debug_imports "deep" in "shapes" {
  } loc(#loc_deep)
  fir.global @_QMshapesSimplSdeepEdeep_var : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_deep_var)
  func.func @_QMshapesSimplSdeepPdeep_helper() {
    return
  } loc(#loc_helper)
}
#loc_shapes = loc("shapes.f90":5:1)
#loc_parent_var = loc("shapes.f90":7:14)
#loc_impl = loc("impl.f90":6:1)
#loc_sub_var = loc("impl.f90":8:14)
#loc_deep = loc("deep.f90":5:1)
#loc_deep_var = loc("deep.f90":7:14)
#loc_helper = loc("deep.f90":9:3)

// Each one is described in the file that defines it, not in the file of the
// ancestor module.
// CHECK-DAG: #[[F_SHAPES:.+]] = #llvm.di_file<"shapes.f90" in {{.*}}>
// CHECK-DAG: #[[F_IMPL:.+]] = #llvm.di_file<"impl.f90" in {{.*}}>
// CHECK-DAG: #[[F_DEEP:.+]] = #llvm.di_file<"deep.f90" in {{.*}}>

// CHECK-DAG: #[[SHAPES:.+]] = #llvm.di_module<file = #[[F_SHAPES]], {{.*}}name = "shapes", line = 5>
// CHECK-DAG: #[[IMPL:.+]] = #llvm.di_module<file = #[[F_IMPL]], {{.*}}name = "shapes.impl", line = 6>
// A nested submodule is named after the root module, not after "shapes.impl".
// CHECK-DAG: #[[DEEP:.+]] = #llvm.di_module<file = #[[F_DEEP]], {{.*}}name = "shapes.deep", line = 5>

// CHECK-DAG: #llvm.di_global_variable<scope = #[[SHAPES]], name = "parent_var"
// CHECK-DAG: #llvm.di_global_variable<scope = #[[IMPL]], name = "sub_var"
// CHECK-DAG: #llvm.di_global_variable<scope = #[[DEEP]], name = "deep_var"

// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[DEEP]], name = "deep_helper"
