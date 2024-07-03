// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
  func.func @ff_(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "arr"} ) {
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0 dummy_scope %0 {uniq_name = "_QFffEarr"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>> loc(#loc1)
    return
  } loc(#loc2)
}
#loc1 = loc("test1.f90":1:1)
#loc2 = loc("test1.f90":3:16)

// CHECK: #llvm.di_composite_type<tag = DW_TAG_array_type
// CHECK-SAME: elements = #llvm.di_subrange<lowerBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(24), DW_OP_deref]>, upperBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(32), DW_OP_deref]>>
// CHECK-SAME: #llvm.di_subrange<lowerBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(48), DW_OP_deref]>, upperBound = #llvm.di_expression<[DW_OP_push_object_address, DW_OP_plus_uconst(56), DW_OP_deref]>>
// CHECK-SAME: dataLocation = <[DW_OP_push_object_address, DW_OP_deref]>>
