// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int sum() {
  int s = 0;
  int i = 0;
  do {
    s += i;
    ++i;
  } while (i <= 10);
  return s;
}

void nestedDoWhile() {
  int a = 0;
  do {
    a++;
    int b = 0;
    while(b < 2) {
      b++;
    }
  }while(a < 2);
}

// CHECK: func.func @sum() -> i32 {
// CHECK: %[[ALLOC:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK: %[[ALLOC0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK: %[[ALLOC1:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK: %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK: memref.store %[[C0_I32]], %[[ALLOC0]][] : memref<i32>
// CHECK: %[[C0_I32_2:.+]] = arith.constant 0 : i32
// CHECK: memref.store %[[C0_I32_2]], %[[ALLOC1]][] : memref<i32>
// CHECK: memref.alloca_scope {
// CHECK:   scf.while : () -> () {
// CHECK:     memref.alloca_scope {
// CHECK:       %[[VAR1:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
// CHECK:       %[[VAR2:.+]] = memref.load %[[ALLOC0]][] : memref<i32>
// CHECK:       %[[ADD:.+]] = arith.addi %[[VAR2]], %[[VAR1]] : i32
// CHECK:       memref.store %[[ADD]], %[[ALLOC0]][] : memref<i32>
// CHECK:       %[[VAR3:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
// CHECK:       %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK:       %[[ADD1:.+]] = arith.addi %[[VAR3]], %[[C1_I32]] : i32
// CHECK:       memref.store %[[ADD1]], %[[ALLOC1]][] : memref<i32>
// CHECK:     }
// CHECK:     %[[VAR4:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
// CHECK:     %[[C10_I32:.+]] = arith.constant 10 : i32
// CHECK:     %[[CMP:.+]] = arith.cmpi sle, %[[VAR4]], %[[C10_I32]] : i32
// CHECK:     scf.condition(%[[CMP]])
// CHECK:   } do {
// CHECK:     scf.yield
// CHECK:   }
// CHECK: }
// CHECK: %[[LOAD:.+]] = memref.load %[[ALLOC0]][] : memref<i32>
// CHECK: memref.store %[[LOAD]], %[[ALLOC]][] : memref<i32>
// CHECK: %[[RET:.+]] = memref.load %[[ALLOC]][] : memref<i32>
// CHECK: return %[[RET]] : i32

// CHECK: func.func @nestedDoWhile() {
// CHECK:     %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK:     %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:     memref.store %[[C0_I32]], %[[alloca]][] : memref<i32>
// CHECK:     memref.alloca_scope  {
// CHECK:       scf.while : () -> () {
// CHECK:         memref.alloca_scope {
// CHECK:           %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK:           %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32>
// CHECK:           %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK:           %[[ONE:.+]] = arith.addi %[[ZERO]], %[[C1_I32]] : i32
// CHECK:           memref.store %[[ONE]], %[[alloca]][] : memref<i32>
// CHECK:           %[[C0_I32_1:.+]] = arith.constant 0 : i32
// CHECK:           memref.store %[[C0_I32_1]], %[[alloca_0]][] : memref<i32>
// CHECK:           memref.alloca_scope  {
// CHECK:             scf.while : () -> () {
// CHECK:               %[[EIGHT:.+]] = memref.load %[[alloca_0]][] : memref<i32>
// CHECK:               %[[C2_I32_3:.+]] = arith.constant 2 : i32
// CHECK:               %[[NINE:.+]] = arith.cmpi slt, %[[EIGHT]], %[[C2_I32_3]] : i32
// CHECK:               scf.condition(%[[NINE]])
// CHECK:             } do {
// CHECK:               %[[EIGHT]] = memref.load %[[alloca_0]][] : memref<i32>
// CHECK:               %[[C1_I32_3:.+]] = arith.constant 1 : i32
// CHECK:               %[[NINE]] = arith.addi %[[EIGHT]], %[[C1_I32_3]] : i32
// CHECK:               memref.store %[[NINE]], %[[alloca_0]][] : memref<i32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         %[[TWO:.+]] = memref.load %[[alloca]][] : memref<i32>
// CHECK:         %[[C2_I32:.+]] = arith.constant 2 : i32
// CHECK:         %[[THREE:.+]] = arith.cmpi slt, %[[TWO]], %[[C2_I32]] : i32
// CHECK:         scf.condition(%[[THREE]])
// CHECK:       } do {
// CHECK:         scf.yield
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
