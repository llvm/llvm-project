// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test that an assumed-shape array of assumed-length characters gets a
// string element type whose length is read from the descriptor.

module {
  func.func @_QFPshow(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "akeys"}) {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFFshowEakeys"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x!fir.char<1,?>>> loc(#loc1)
    return
  } loc(#loc2)
}
#loc1 = loc("string.f90":16:1)
#loc2 = loc("string.f90":15:1)

// CHECK: #[[STR:.*]] = #llvm.di_string_type<tag = DW_TAG_string_type, name = "", stringLengthExp = <[DW_OP_push_object_address, DW_OP_plus_uconst(8)]>, encoding = DW_ATE_ASCII>
// CHECK: #[[ARR:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}baseType = #[[STR]]
// CHECK-SAME: dataLocation = <[DW_OP_push_object_address, DW_OP_deref]>
// CHECK: #llvm.di_local_variable<{{.*}}name = "akeys"{{.*}}type = #[[ARR]]>
