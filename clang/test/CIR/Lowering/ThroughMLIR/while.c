// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void singleWhile() {
  int a = 0;
  while(a < 2) {
    a++;
  }
}

void nestedWhile() {
  int a = 0;
  while(a < 2) {
    int b = 0;
    while(b < 2) {
      b++;
    }
    a++;
  }
}

//CHECK: func.func @singleWhile() {
//CHECK:   %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32
//CHECK:   memref.store %[[C0_I32]], %[[alloca]][] : memref<i32>
//CHECK:   memref.alloca_scope  {
//CHECK:     scf.while : () -> () {
//CHECK:       %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32>
//CHECK:       %[[C2_I32:.+]] = arith.constant 2 : i32
//CHECK:       %[[ONE:.+]] = arith.cmpi slt, %[[ZERO:.+]], %[[C2_I32]] : i32
//CHECK:       scf.condition(%[[ONE]])
//CHECK:     } do {
//CHECK:       memref.alloca_scope {
//CHECK:         %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32>
//CHECK:         %[[C1_I32:.+]] = arith.constant 1 : i32
//CHECK:         %[[ONE:.+]] = arith.addi %0, %[[C1_I32:.+]] : i32
//CHECK:         memref.store %[[ONE:.+]], %[[alloca]][] : memref<i32>
//CHECK:       }
//CHECK:       scf.yield
//CHECK:     }
//CHECK:  }
//CHECK:   return
//CHECK: }

//CHECK: func.func @nestedWhile() {
//CHECK:   %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32
//CHECK:   memref.store %[[C0_I32]], %[[alloca]][] : memref<i32>
//CHECK:   memref.alloca_scope  {
//CHECK:     scf.while : () -> () {
//CHECK:       %[[ZERO:.+]] = memref.load %alloca[] : memref<i32>
//CHECK:       %[[C2_I32:.+]] = arith.constant 2 : i32
//CHECK:       %[[ONE:.+]] = arith.cmpi slt, %[[ZERO]], %[[C2_I32]] : i32
//CHECK:       scf.condition(%[[ONE]])
//CHECK:     } do {
//CHECK:       memref.alloca_scope {
//CHECK:         %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:         %[[C0_I32_1:.+]] = arith.constant 0 : i32
//CHECK:         memref.store %[[C0_I32_1]], %[[alloca_0]][] : memref<i32>
//CHECK:         memref.alloca_scope  {
//CHECK:           scf.while : () -> () {
//CHECK:             %{{.*}} = memref.load %[[alloca_0]][] : memref<i32>
//CHECK:             %[[C2_I32]] = arith.constant 2 : i32
//CHECK:             %[[SEVEN:.*]] = arith.cmpi slt, %{{.*}}, %[[C2_I32]] : i32
//CHECK:             scf.condition(%[[SEVEN]])
//CHECK:           } do {
//CHECK:             %{{.*}} = memref.load %[[alloca_0]][] : memref<i32>
//CHECK:             %[[C1_I32_2:.+]] = arith.constant 1 : i32
//CHECK:             %{{.*}} = arith.addi %{{.*}}, %[[C1_I32_2]] : i32
//CHECK:             memref.store %{{.*}}, %[[alloca_0]][] : memref<i32>
//CHECK:             scf.yield
//CHECK:           }
//CHECK:         }
//CHECK:         %[[ZERO]] = memref.load %[[alloca]][] : memref<i32>
//CHECK:         %[[C1_I32:.+]] = arith.constant 1 : i32
//CHECK:         %[[ONE]] = arith.addi %[[ZERO]], %[[C1_I32]] : i32
//CHECK:         memref.store %[[ONE]], %[[alloca]][] : memref<i32>
//CHECK:       }
//CHECK:       scf.yield
//CHECK:     }
//CHECK:   }
//CHECK:   return
//CHECK: }
