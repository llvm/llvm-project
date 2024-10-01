// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QMhelperPfn(%arg0: !fir.ref<!fir.array<5x?xi32>> {fir.bindc_name = "a1"}, %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a2"}, %arg2: !fir.ref<!fir.array<2x?xi32>> {fir.bindc_name = "a3"}) {
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c-1 = arith.constant -1 : index
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg0(%c5, %c-1) dummy_scope %0 {uniq_name = "_QMhelperFfnEa1"} : (!fir.ref<!fir.array<5x?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<5x?xi32>> loc(#loc1)
    %2 = fircg.ext_declare %arg1(%c-1) origin %c2 dummy_scope %0 {uniq_name = "_QMhelperFfnEa2"} : (!fir.ref<!fir.array<?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc2)
    return
  } loc(#loc3)
}
#loc3 = loc("test.f90":1:1)
#loc1 = loc("test.f90":3:1)
#loc2 = loc("test.f90":4:1)

// CHECK-DAG: #[[TY1:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}elements = #llvm.di_subrange<count = 5 : i64, lowerBound = 1 : i64>, #llvm.di_subrange<>>
// CHECK-DAG: #[[TY2:.*]] = #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}elements = #llvm.di_subrange<lowerBound = 2 : i64>>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "a1"{{.*}}type = #[[TY1]]>
// CHECK-DAG: #llvm.di_local_variable<{{.*}}name = "a2"{{.*}}type = #[[TY2]]>
