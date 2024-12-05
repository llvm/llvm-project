// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QFPfn(%arg0: !fir.box<!fir.array<*:i32>> )  {
    %1 = fir.undefined !fir.dscope
    %2 = fircg.ext_declare %arg0 dummy_scope %1 {uniq_name = "_QFFfnEx"} : (!fir.box<!fir.array<*:i32>>, !fir.dscope) -> !fir.box<!fir.array<*:i32>> loc(#loc2)
    return
  } loc(#loc1)
}
#loc1 = loc("test1.f90":1:1)
#loc2 = loc("test1.f90":3:16)

// CHECK: #[[TY:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}elements = #llvm.di_generic_subrange<count = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_over, DW_OP_constu(24), DW_OP_mul, DW_OP_plus_uconst(32), DW_OP_plus, DW_OP_deref]>, lowerBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_over, DW_OP_constu(24), DW_OP_mul, DW_OP_plus_uconst(24), DW_OP_plus, DW_OP_deref]>, stride = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_over, DW_OP_constu(24), DW_OP_mul, DW_OP_plus_uconst(40), DW_OP_plus, DW_OP_deref]>>, dataLocation = <[DW_OP_push_object_address, DW_OP_deref]>, rank = <[DW_OP_push_object_address, DW_OP_plus_uconst(20), DW_OP_deref_size(1)]>>
// CHECK: #llvm.di_local_variable<{{.*}}name = "x"{{.*}}type = #[[TY]]{{.*}}>
