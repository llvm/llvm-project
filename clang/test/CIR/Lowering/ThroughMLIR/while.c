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
//CHECK:       %[[TWO:.+]] = arith.extui %[[ONE:.+]] : i1 to i32 
//CHECK:       %[[C0_I32_0:.+]] = arith.constant 0 : i32 
//CHECK:       %[[THREE:.+]] = arith.cmpi ne, %[[TWO:.+]], %[[C0_I32_0]] : i32 
//CHECK:       %[[FOUR:.+]] = arith.extui %[[THREE:.+]] : i1 to i8 
//CHECK:       %[[FIVE:.+]] = arith.trunci %[[FOUR:.+]] : i8 to i1 
//CHECK:       scf.condition(%[[FIVE]]) 
//CHECK:     } do {
//CHECK:       %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32> 
//CHECK:       %[[C1_I32:.+]] = arith.constant 1 : i32 
//CHECK:       %[[ONE:.+]] = arith.addi %0, %[[C1_I32:.+]] : i32 
//CHECK:       memref.store %[[ONE:.+]], %[[alloca]][] : memref<i32> 
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
//CHECK:     %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:     scf.while : () -> () {
//CHECK:       %[[ZERO:.+]] = memref.load %alloca[] : memref<i32> 
//CHECK:       %[[C2_I32:.+]] = arith.constant 2 : i32 
//CHECK:       %[[ONE:.+]] = arith.cmpi slt, %[[ZERO]], %[[C2_I32]] : i32 
//CHECK:       %[[TWO:.+]] = arith.extui %[[ONE]] : i1 to i32 
//CHECK:       %[[C0_I32_1:.+]] = arith.constant 0 : i32 
//CHECK:       %[[THREE:.+]] = arith.cmpi ne, %[[TWO]], %[[C0_I32_1]] : i32 
//CHECK:       %[[FOUR:.+]] = arith.extui %[[THREE]] : i1 to i8 
//CHECK:       %[[FIVE:.+]] = arith.trunci %[[FOUR]] : i8 to i1 
//CHECK:       scf.condition(%[[FIVE]]) 
//CHECK:     } do {
//CHECK:         %[[C0_I32_1]] = arith.constant 0 : i32 
//CHECK:         memref.store %[[C0_I32_1]], %[[alloca_0]][] : memref<i32> 
//CHECK:         memref.alloca_scope  {
//CHECK:           scf.while : () -> () {
//CHECK:             %[[TWO]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:             %[[C2_I32]] = arith.constant 2 : i32 
//CHECK:             %[[THREE]] = arith.cmpi slt, %[[TWO]], %[[C2_I32]] : i32 
//CHECK:             %[[FOUR]] = arith.extui %[[THREE]] : i1 to i32 
//CHECK:             %[[C0_I32_2:.+]] = arith.constant 0 : i32 
//CHECK:             %[[FIVE]] = arith.cmpi ne, %[[FOUR]], %[[C0_I32_2]] : i32 
//CHECK:             %[[SIX:.+]] = arith.extui %[[FIVE]] : i1 to i8 
//CHECK:             %[[SEVEN:.+]] = arith.trunci %[[SIX]] : i8 to i1 
//CHECK:             scf.condition(%[[SEVEN]]) 
//CHECK:           } do {
//CHECK:             %[[TWO]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:             %[[C1_I32_2:.+]] = arith.constant 1 : i32 
//CHECK:             %[[THREE]] = arith.addi %[[TWO]], %[[C1_I32_2]] : i32 
//CHECK:             memref.store %[[THREE]], %[[alloca_0]][] : memref<i32> 
//CHECK:             scf.yield 
//CHECK:           } 
//CHECK:         } 
//CHECK:         %[[ZERO]] = memref.load %[[alloca]][] : memref<i32> 
//CHECK:         %[[C1_I32:.+]] = arith.constant 1 : i32 
//CHECK:         %[[ONE]] = arith.addi %[[ZERO]], %[[C1_I32]] : i32 
//CHECK:         memref.store %[[ONE]], %[[alloca]][] : memref<i32> 
//CHECK:         scf.yield 
//CHECK:       } 
//CHECK:     } 
//CHECK:     return 
//CHECK:   } 
//CHECK: } 