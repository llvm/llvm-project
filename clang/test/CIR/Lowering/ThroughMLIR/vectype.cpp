// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

typedef int vi4 __attribute__((vector_size(16)));

void vector_int_test(int x) {

  // CHECK: %[[ALLOC1:.*]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
  // CHECK: %[[ALLOC2:.*]] = memref.alloca() {alignment = 16 : i64} : memref<vector<4xi32>>
  // CHECK: %[[ALLOC3:.*]] = memref.alloca() {alignment = 16 : i64} : memref<vector<4xi32>>
  // CHECK: %[[ALLOC4:.*]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
  // CHECK: memref.store %arg0, %[[ALLOC1]][] : memref<i32>

  vi4 a = { 1, 2, 3, 4 };

  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: %[[C2:.*]] = arith.constant 2 : i32
  // CHECK: %[[C3:.*]] = arith.constant 3 : i32
  // CHECK: %[[C4:.*]] = arith.constant 4 : i32
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<4xi32>
  // CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
  // CHECK: %[[VEC0:.*]] = vector.insertelement %[[C1]], %[[CST]][%[[C0_I64]] : i64] : vector<4xi32>
  // CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  // CHECK: %[[VEC1:.*]] = vector.insertelement %[[C2]], %[[VEC0]][%[[C1_I64]] : i64] : vector<4xi32>
  // CHECK: %[[C2_I64:.*]] = arith.constant 2 : i64
  // CHECK: %[[VEC2:.*]] = vector.insertelement %[[C3]], %[[VEC1]][%[[C2_I64]] : i64] : vector<4xi32>
  // CHECK: %[[C3_I64:.*]] = arith.constant 3 : i64
  // CHECK: %[[VEC3:.*]] = vector.insertelement %[[C4]], %[[VEC2]][%[[C3_I64]] : i64] : vector<4xi32>
  // CHECK: memref.store %[[VEC3]], %[[ALLOC2]][] : memref<vector<4xi32>>
  
  vi4 b = {x, 5, 6, x + 1};

  // CHECK: %[[VAL1:.*]] = memref.load %[[ALLOC1]][] : memref<i32>
  // CHECK: %[[C5:.*]] = arith.constant 5 : i32
  // CHECK: %[[C6:.*]] = arith.constant 6 : i32
  // CHECK: %[[VAL2:.*]] = memref.load %[[ALLOC1]][] : memref<i32>
  // CHECK: %[[C1_I32_2:.*]] = arith.constant 1 : i32
  // CHECK: %[[SUM:.*]] = arith.addi %[[VAL2]], %[[C1_I32_2]] : i32
  // CHECK: %[[CST2:.*]] = arith.constant dense<0> : vector<4xi32>
  // CHECK: %[[C0_I64_2:.*]] = arith.constant 0 : i64
  // CHECK: %[[VEC4:.*]] = vector.insertelement %[[VAL1]], %[[CST2]][%[[C0_I64_2]] : i64] : vector<4xi32>
  // CHECK: %[[C1_I64_2:.*]] = arith.constant 1 : i64
  // CHECK: %[[VEC5:.*]] = vector.insertelement %[[C5]], %[[VEC4]][%[[C1_I64_2]] : i64] : vector<4xi32>
  // CHECK: %[[C2_I64_2:.*]] = arith.constant 2 : i64
  // CHECK: %[[VEC6:.*]] = vector.insertelement %[[C6]], %[[VEC5]][%[[C2_I64_2]] : i64] : vector<4xi32>
  // CHECK: %[[C3_I64_2:.*]] = arith.constant 3 : i64
  // CHECK: %[[VEC7:.*]] = vector.insertelement %[[SUM]], %[[VEC6]][%[[C3_I64_2]] : i64] : vector<4xi32>
  // CHECK: memref.store %[[VEC7]], %[[ALLOC3]][] : memref<vector<4xi32>>  

  a[x] = x;
  
  // CHECK: %[[VAL3:.*]] = memref.load %[[ALLOC1]][] : memref<i32>
  // CHECK: %[[VAL4:.*]] = memref.load %[[ALLOC1]][] : memref<i32>
  // CHECK: %[[VEC8:.*]] = memref.load %[[ALLOC2]][] : memref<vector<4xi32>>
  // CHECK: %[[VEC9:.*]] = vector.insertelement %[[VAL3]], %[[VEC8]][%[[VAL4]] : i32] : vector<4xi32>
  // CHECK: memref.store %[[VEC9]], %[[ALLOC2]][] : memref<vector<4xi32>>

  int c = a[x];

  // CHECK: %[[VEC10:.*]] = memref.load %[[ALLOC2]][] : memref<vector<4xi32>>
  // CHECK: %[[VAL5:.*]] = memref.load %[[ALLOC1]][] : memref<i32>
  // CHECK: %[[EXTRACT:.*]] = vector.extractelement %[[VEC10]][%[[VAL5]] : i32] : vector<4xi32>
  // CHECK: memref.store %[[EXTRACT]], %[[ALLOC4]][] : memref<i32>

  // CHECK: return
}