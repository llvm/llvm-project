// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test how a submodule is described: its name, where it is described, what is
// scoped in it, and the module it imports.

module {
  fir.module_debug_imports "shapes" {
  } loc(#loc_shapes)
  fir.global @_QMshapesEparent_var : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_parent_var)

  fir.module_debug_imports "impl" in "shapes" parent "shapes" {
  } loc(#loc_impl)
  fir.global @_QMshapesSimplEsub_var : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_sub_var)
  func.func @_QMshapesPsquare() attributes {fir.defining_submodule = "impl"} {
    return
  } loc(#loc_square)

  fir.module_debug_imports "deep" in "shapes" parent "impl" {
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
#loc_square = loc("impl.f90":10:3)
#loc_deep = loc("deep.f90":12:1)
#loc_deep_var = loc("deep.f90":14:14)
#loc_helper = loc("deep.f90":16:3)

// Each module is described in the file that defines it.
// CHECK-DAG: #[[F_SHAPES:.+]] = #llvm.di_file<"shapes.f90" in {{.*}}>
// CHECK-DAG: #[[F_IMPL:.+]] = #llvm.di_file<"impl.f90" in {{.*}}>
// CHECK-DAG: #[[F_DEEP:.+]] = #llvm.di_file<"deep.f90" in {{.*}}>

// The compile unit is built as a placeholder and then holding the imports, so
// a module that carries one is checked against each form.
// CHECK-DAG: #[[CU_SELF:.+]] = #llvm.di_compile_unit<recId = [[RECID:.+]], isRecSelf = true>
// CHECK-DAG: #[[CU:.+]] = #llvm.di_compile_unit<recId = [[RECID]], id = {{.*}}>

// A submodule is named after the module at the root of its ancestry, nested or
// not.
// CHECK-DAG: #[[SHAPES:.+]] = #llvm.di_module<file = #[[F_SHAPES]], scope = #[[CU]], name = "shapes", line = 5>
// CHECK-DAG: #[[IMPL:.+]] = #llvm.di_module<file = #[[F_IMPL]], scope = #[[CU]], name = "shapes.impl", line = 6>
// CHECK-DAG: #[[DEEP:.+]] = #llvm.di_module<file = #[[F_DEEP]], scope = #[[CU]], name = "shapes.deep", line = 12>

// Each submodule imports the one directly above it.
// CHECK-DAG: #[[SHAPES_S:.+]] = #llvm.di_module<file = #[[F_SHAPES]], scope = #[[CU_SELF]], name = "shapes", line = 5>
// CHECK-DAG: #[[IMPL_S:.+]] = #llvm.di_module<file = #[[F_IMPL]], scope = #[[CU_SELF]], name = "shapes.impl", line = 6>
// CHECK-DAG: #[[DEEP_S:.+]] = #llvm.di_module<file = #[[F_DEEP]], scope = #[[CU_SELF]], name = "shapes.deep", line = 12>
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[IMPL_S]], entity = #[[SHAPES_S]]
// CHECK-DAG: #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #[[DEEP_S]], entity = #[[IMPL_S]]

// An entity is scoped in the submodule that declares it. A separate module
// procedure goes by its attribute, anything else by its uniqued name.
// CHECK-DAG: #llvm.di_global_variable<scope = #[[SHAPES]], name = "parent_var"
// CHECK-DAG: #llvm.di_global_variable<scope = #[[IMPL]], name = "sub_var"
// CHECK-DAG: #llvm.di_global_variable<scope = #[[DEEP]], name = "deep_var"
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[IMPL]], name = "square"
// CHECK-DAG: #llvm.di_subprogram<{{.*}}scope = #[[DEEP]], name = "deep_helper"
