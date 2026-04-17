// Test emboxing PPC vector type to a descriptor
//RUN: fir-opt -cg-rewrite --fir-to-llvm-ir %s | FileCheck %s

func.func @test1() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.vector<4:i32>>>
  %1 = fir.zero_bits !fir.heap<!fir.vector<4:i32>>
  %2 = fir.embox %1 : (!fir.heap<!fir.vector<4:i32>>) -> !fir.box<!fir.heap<!fir.vector<4:i32>>>
  fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.vector<4:i32>>>>
  return
}

//CHECK-LABEL: llvm.func @test1()
//CHECK: %[[TYPE_VAL:.*]] = llvm.mlir.constant(42 : i32) : i32
//CHECK: %[[ELEM_LEN_VAL:.*]] = llvm.mlir.constant(16 : i64) : i64
//CHECK: %[[DSC_1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK: %[[DSC_2:.*]] = llvm.insertvalue %[[ELEM_LEN_VAL]], %[[DSC_1]][1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK: %[[RANK_VAL:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK: %[[DSC_3:.*]] = llvm.trunc %[[RANK_VAL]] : i32 to i8
//CHECK: %[[DSC_4:.*]] = llvm.insertvalue %[[DSC_3]], %11[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK: %[[DSC_5:.*]] = llvm.trunc %[[TYPE_VAL]] : i32 to i8
//CHECK: %[[DSC_6:.*]] = llvm.insertvalue %[[DSC_5]], %[[DSC_4]][4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>

func.func @test2() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.vector<2:f64>>>
  %1 = fir.zero_bits !fir.heap<!fir.vector<2:f64>>
  %2 = fir.embox %1 : (!fir.heap<!fir.vector<2:f64>>) -> !fir.box<!fir.heap<!fir.vector<2:f64>>>
  fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.vector<2:f64>>>>
  return
}

//CHECK-LABEL: llvm.func @test2()
//CHECK: %[[TYPE_VAL:.*]] = llvm.mlir.constant(42 : i32) : i32
//CHECK: %[[ELEM_LEN_VAL:.*]] = llvm.mlir.constant(16 : i64) : i64
//CHECK: %[[DSC_1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK: %[[DSC_2:.*]] = llvm.insertvalue %[[ELEM_LEN_VAL]], %[[DSC_1]][1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK: %[[RANK_VAL:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK: %[[DSC_3:.*]] = llvm.trunc %[[RANK_VAL]] : i32 to i8
//CHECK: %[[DSC_4:.*]] = llvm.insertvalue %[[DSC_3]], %11[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
//CHECK: %[[DSC_5:.*]] = llvm.trunc %[[TYPE_VAL]] : i32 to i8
//CHECK: %[[DSC_6:.*]] = llvm.insertvalue %[[DSC_5]], %[[DSC_4]][4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
