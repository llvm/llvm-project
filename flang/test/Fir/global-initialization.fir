// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s
// RUN: tco --fir-to-llvm-ir %s | FileCheck %s

fir.global internal @_QEmask : !fir.array<32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32xi32>
  %2 = fir.insert_on_range %0, %c0_i32 from (0) to (31) : (!fir.array<32xi32>, i32) -> !fir.array<32xi32>
  fir.has_value %2 : !fir.array<32xi32>
}

// CHECK: llvm.mlir.global internal @_QEmask() : !llvm.array<32 x i32> {
// CHECK:   [[VAL0:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   [[VAL1:%.*]] = llvm.mlir.undef : !llvm.array<32 x i32>
// CHECK:   [[VAL2:%.*]] = llvm.mlir.constant(dense<1> : vector<32xi32>) : !llvm.array<32 x i32>
// CHECK:   llvm.return [[VAL2]] : !llvm.array<32 x i32>
// CHECK: }

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  %2 = fir.insert_on_range %0, %c0_i32 from (0,0) to (31,31) : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// CHECK: llvm.mlir.global internal @_QEmultiarray() : !llvm.array<32 x array<32 x i32>> {
// CHECK:   [[VAL0:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   [[VAL1:%.*]] = llvm.mlir.undef : !llvm.array<32 x array<32 x i32>>
// CHECK:   [[VAL2:%.*]] = llvm.mlir.constant(dense<1> : vector<32x32xi32>) : !llvm.array<32 x array<32 x i32>>
// CHECK:   llvm.return [[VAL2]] : !llvm.array<32 x array<32 x i32>>
// CHECK: }

fir.global internal @_QEmasklogical : !fir.array<32768x!fir.logical<4>> {
  %true = arith.constant true
  %0 = fir.undefined !fir.array<32768x!fir.logical<4>>
  %1 = fir.convert %true : (i1) -> !fir.logical<4>
  %2 = fir.insert_on_range %0, %1 from (0) to (32767) : (!fir.array<32768x!fir.logical<4>>, !fir.logical<4>) -> !fir.array<32768x!fir.logical<4>>
  fir.has_value %2 : !fir.array<32768x!fir.logical<4>>
}

// CHECK: llvm.mlir.global internal @_QEmasklogical() : !llvm.array<32768 x i32> {
// CHECK:   [[VAL0:%.*]] = llvm.mlir.constant(true) : i1
// CHECK:   [[VAL1:%.*]] = llvm.mlir.undef : !llvm.array<32768 x i32>
// CHECK:   [[VAL2:%.*]] = llvm.zext [[VAL0]] : i1 to i32
// CHECK:   [[VAL3:%.*]] = llvm.mlir.constant(dense<true> : vector<32768xi1>) : !llvm.array<32768 x i32>
// CHECK:   llvm.return [[VAL3]] : !llvm.array<32768 x i32>
// CHECK: }

fir.global internal @_QElookforme : !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}> {
  %c2_i32 = arith.constant 2 : i32
  %c52_i32 = arith.constant 52 : i32
  %0 = fir.undefined !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>
  %1 = fir.undefined !fir.array<500xi32>
  %2 = fir.insert_on_range %1, %c2_i32 from (0) to (499) : (!fir.array<500xi32>, i32) -> !fir.array<500xi32>
  %3 = fir.insert_value %0, %2, ["i", !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>] :(!fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>, !fir.array<500xi32>) -> !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>
  %4 = fir.insert_on_range %1, %c52_i32 from (0) to (499) : (!fir.array<500xi32>, i32) -> !fir.array<500xi32>
  %5 = fir.insert_value %3, %4, ["j", !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>] : (!fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>, !fir.array<500xi32>) -> !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>
  fir.has_value %5 : !fir.type<_QTt{i:!fir.array<500xi32>,j:!fir.array<500xi32>}>
}

// CHECK: llvm.mlir.global internal @_QElookforme() : !llvm.struct<"_QTt", (array<500 x i32>, array<500 x i32>)> {
// CHECK:   [[CST0:%.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:   [[CST1:%.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK:   [[STRUCT:%.*]] = llvm.mlir.undef : !llvm.struct<"_QTt", (array<500 x i32>, array<500 x i32>)>
// CHECK:   [[ARR1:%.*]] = llvm.mlir.undef : !llvm.array<500 x i32>
// CHECK:   [[DENSE1:%.*]] = llvm.mlir.constant(dense<2> : vector<500xi32>) : !llvm.array<500 x i32>
// CHECK:   [[STRUCT1:%.*]] = llvm.insertvalue [[DENSE1]], [[STRUCT]][0 : i32] : !llvm.struct<"_QTt", (array<500 x i32>, array<500 x i32>)>
// CHECK:   [[DENSE2:%.*]] = llvm.mlir.constant(dense<52> : vector<500xi32>) : !llvm.array<500 x i32>
// CHECK:   [[STRUCT2:%.*]] = llvm.insertvalue [[DENSE2]], [[STRUCT1]][1 : i32] : !llvm.struct<"_QTt", (array<500 x i32>, array<500 x i32>)>
// CHECK:   llvm.return [[STRUCT2]] : !llvm.struct<"_QTt", (array<500 x i32>, array<500 x i32>)>
// CHECK: }
