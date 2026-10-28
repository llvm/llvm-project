// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>, llvm.target_triple = "amdgcn-amd-amdhsa"} {
  func.func @_QPshp(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "s"}) {
    %c3 = arith.constant 3 : index
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 origin %c3 dummy_scope %0 uniq_name("_QFshpEx") : (!fir.box<!fir.array<?xi32>>, index, !fir.dscope) -> !fir.box<!fir.array<?xi32>> loc(#loc1)
    %2 = fircg.ext_declare %arg1 dummy_scope %0 uniq_name("_QFshpEs") : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.dscope) -> !fir.box<!fir.array<?x!fir.char<1,?>>> loc(#loc2)
    %3 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QFshpEa"}
    %4 = fircg.ext_declare %3 uniq_name("_QFshpEa") : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> loc(#loc3)
    return
  } loc(#loc4)
  func.func @_QPrnk(%arg0: !fir.box<!fir.array<*:i32>> {fir.bindc_name = "r"}) {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 uniq_name("_QFrnkEr") : (!fir.box<!fir.array<*:i32>>, !fir.dscope) -> !fir.box<!fir.array<*:i32>> loc(#loc5)
    return
  } loc(#loc6)
}
#loc1 = loc("test.f90":2:3)
#loc2 = loc("test.f90":3:3)
#loc3 = loc("test.f90":4:3)
#loc4 = loc("test.f90":1:3)
#loc5 = loc("test.f90":8:3)
#loc6 = loc("test.f90":7:3)

// CHECK-DAG: #[[STR:.*]] = #llvm.di_string_type<{{.*}}stringLengthExp = <[DW_OP_push_object_address, DW_OP_constu(8), DW_OP_LLVM_user(4)]>
// CHECK-DAG: #[[X_TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}elements = #llvm.di_subrange<count = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_constu(32), DW_OP_LLVM_user(4), DW_OP_deref]>, lowerBound = 3 : i64, stride = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_constu(40), DW_OP_LLVM_user(4), DW_OP_deref]>>>
// CHECK-DAG: #[[S_TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #[[STR]]
// CHECK-DAG: #[[A_TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}allocated = {{.*}}lowerBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_constu(24), DW_OP_LLVM_user(4), DW_OP_deref]>
// CHECK-DAG: #[[R_TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, {{.*}}rank = <[DW_OP_push_object_address, DW_OP_constu(20), DW_OP_LLVM_user(4), DW_OP_deref_size(1)]>, elements = #llvm.di_generic_subrange<count = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_over, DW_OP_constu(24), DW_OP_mul, DW_OP_plus_uconst(32), DW_OP_LLVM_user(4), DW_OP_deref]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "x"{{.*}}type = #[[X_TY]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "s"{{.*}}type = #[[S_TY]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "a"{{.*}}type = #[[A_TY]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "r"{{.*}}type = #[[R_TY]]>
