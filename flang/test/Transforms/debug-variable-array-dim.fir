// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @foo_(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}, %arg2: !fir.ref<i32> {fir.bindc_name = "m"}, %arg3: !fir.ref<i32> {fir.bindc_name = "p"}) attributes {fir.internal_name = "_QPfoo"} {
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2 = arith.constant 2 : index
    %c10_i32 = arith.constant 10 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = fir.undefined !fir.dscope
    %1 = fircg.ext_declare %arg1 dummy_scope %0 {uniq_name = "_QFfooEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc2)
    %2 = fircg.ext_declare %arg2 dummy_scope %0 {uniq_name = "_QFfooEm"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %3 = fircg.ext_declare %arg3 dummy_scope %0 {uniq_name = "_QFfooEp"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %4 = fir.load %1 : !fir.ref<i32>
    %5 = fir.convert %4 : (i32) -> index
    %6 = fir.load %2 : !fir.ref<i32>
    %7 = fir.convert %6 : (i32) -> index
    %8 = arith.subi %7, %5 : index
    %9 = arith.addi %8, %c1 : index
    %10 = arith.cmpi sgt, %9, %c0 : index
    %11 = arith.select %10, %9, %c0 : index
    %12 = fir.load %3 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> index
    %14 = arith.cmpi sgt, %13, %c0 : index
    %15 = arith.select %14, %13, %c0 : index
    %16 = fircg.ext_declare %arg0(%11, %15) origin %5, %c1 dummy_scope %0 {uniq_name = "_QFfooEa"} : (!fir.ref<!fir.array<?x?xi32>>, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc5)
    return
  } loc(#loc1)
}

#loc1 = loc("test.f90":5:1)
#loc2 = loc("test.f90":6:11)
#loc3 = loc("test.f90":7:11)
#loc4 = loc("test.f90":2:8)
#loc5 = loc("test.f90":8:11)

// CHECK-DAG: #[[VAR0:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFfooEa3"{{.*}}flags = Artificial>
// CHECK-DAG: #[[VAR1:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFfooEa1"{{.*}}flags = Artificial>
// CHECK-DAG: #[[VAR2:.*]] = #llvm.di_local_variable<{{.*}}name = "._QFfooEa2"{{.*}}flags = Artificial>
// CHECK-DAG: #llvm.di_composite_type<tag = DW_TAG_array_type{{.*}}elements = #llvm.di_subrange<count = #[[VAR1]], lowerBound = #[[VAR0]]>, #llvm.di_subrange<count = #[[VAR2]]>>
