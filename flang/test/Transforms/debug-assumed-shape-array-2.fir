// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test assumed shape array with variable lower bound.

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func private @_QFPfn(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {} {
    %c23_i32 = arith.constant 23 : i32
    %c6_i32 = arith.constant 6 : i32
    %c20_i32 = arith.constant 20 : i32
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg1 dummy_scope %0 {uniq_name = "_QFFfnEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc1)
    %2 = fir.load %1 : !fir.ref<i32>
    %3 = fir.convert %2 : (i32) -> index
    %4 = fircg.ext_declare %arg0 origin %3 dummy_scope %0 {uniq_name = "_QFFfnEb"} : (!fir.box<!fir.array<?xi32>>, index, !fir.dscope) -> !fir.box<!fir.array<?xi32>> loc(#loc2)
    return
  } loc(#loc3)
}
#loc1 = loc("test1.f90":1:1)
#loc2 = loc("test1.f90":3:16)
#loc3 = loc("test1.f90":4:16)

// CHECK: #[[VAR:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFFfnEb1"{{.*}}flags = Artificial>
// CHECK: #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}elements = #llvm.di_subrange<count = #llvm.di_expression<[{{.*}}]>, lowerBound = #[[VAR]], stride = #llvm.di_expression<[{{.*}}]>>, dataLocation = <[DW_OP_push_object_address, DW_OP_deref]>>
