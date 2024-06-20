// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void foo() {
  int a = 0;
  while(a < 2) {
    a++;
  }
}

//CHECK: func.func @foo() {
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