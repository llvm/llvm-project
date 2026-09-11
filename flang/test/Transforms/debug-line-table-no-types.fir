// Test that a compile unit which emits no type information gets an empty
// DISubroutineType.

// RUN: fir-opt --add-debug-info="debug-level=Full" --mlir-print-debuginfo %s | FileCheck %s --check-prefix=FULL
// RUN: fir-opt --add-debug-info="debug-level=LineTablesOnly" --mlir-print-debuginfo %s | FileCheck %s --check-prefix=MINIMAL
// RUN: fir-opt --add-debug-info="debug-level=DebugDirectivesOnly" --mlir-print-debuginfo %s | FileCheck %s --check-prefix=MINIMAL

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPfn(%arg0: !fir.ref<!fir.type<_QFTpoint{x:f32,y:f32}>>, %arg1: !fir.box<!fir.array<?xf32>>, %arg2: !fir.ref<i32>) -> i32 {
    %0 = fir.undefined i32
    return %0 : i32
  } loc(#loc_fn)
} loc(#loc_module)
#loc_module = loc("test.f90":1:1)
#loc_fn = loc("test.f90":5:1)

// At Full the signature drags in a type for the result and for every argument.
// FULL-DAG: #[[INT4:.+]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer(kind=4)", sizeInBits = 32, encoding = DW_ATE_signed>
// FULL-DAG: #[[REAL4:.+]] = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real(kind=4)", sizeInBits = 32, encoding = DW_ATE_float>
// FULL-DAG: #[[POINT:.+]] = #llvm.di_composite_type<{{.*}}tag = DW_TAG_structure_type, name = "point"
// FULL-DAG: #[[ARRAY:.+]] = #llvm.di_composite_type<tag = DW_TAG_array_type
// FULL-DAG: #[[SUBTY:.+]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #[[INT4]], #[[POINT]], #[[ARRAY]], #[[INT4]]>
// FULL-DAG: #llvm.di_subprogram<{{.*}}name = "fn"{{.*}}type = #[[SUBTY]]>

// MINIMAL-NOT: #llvm.di_basic_type
// MINIMAL-NOT: #llvm.di_composite_type
// MINIMAL-NOT: #llvm.di_derived_type
// MINIMAL-NOT: #llvm.di_string_type
// MINIMAL-NOT: #llvm.di_null_type
// MINIMAL: #[[SUBTY:.+]] = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
// MINIMAL: #llvm.di_subprogram<{{.*}}name = "fn"{{.*}}type = #[[SUBTY]]>
