// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void foo() {
  int a = 2;
  int b = 0;
  if (a > 0) {
    b++;
  } else {
    b--;
  }
}

//CHECK: func.func @foo() {
//CHECK:   %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:   %[[C2_I32:.+]] = arith.constant 2 : i32 
//CHECK:   memref.store %[[C2_I32]], %[[alloca]][] : memref<i32> 
//CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32 
//CHECK:   memref.store %[[C0_I32]], %[[alloca_0]][] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32> 
//CHECK:     %[[C0_I32_1:.+]] = arith.constant 0 : i32 
//CHECK:     %[[ONE:.+]] = arith.cmpi sgt, %[[ZERO]], %[[C0_I32_1]] : i32 
//CHECK:     %[[TWO:.+]] = arith.extui %[[ONE]] : i1 to i32 
//CHECK:     %[[C0_I32_2:.+]] = arith.constant 0 : i32 
//CHECK:     %[[THREE:.+]] = arith.cmpi ne, %[[TWO]], %[[C0_I32_2]] : i32 
//CHECK:     %[[FOUR:.+]] = arith.extui %[[THREE]] : i1 to i8 
//CHECK:     %[[FIVE:.+]] = arith.trunci %[[FOUR]] : i8 to i1 
//CHECK:     scf.if %[[FIVE]] {
//CHECK:       %[[SIX:.+]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:       %[[C1_I32:.+]] = arith.constant 1 : i32 
//CHECK:       %[[SEVEN:.+]] = arith.addi %[[SIX]], %[[C1_I32]] : i32 
//CHECK:       memref.store %[[SEVEN]], %[[alloca_0]][] : memref<i32> 
//CHECK:     } else {
//CHECK:       %[[SIX:.+]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:       %[[C1_I32:.+]] = arith.constant 1 : i32 
//CHECK:       %[[SEVEN:.+]] = arith.subi %[[SIX]], %[[C1_I32]] : i32 
//CHECK:       memref.store %[[SEVEN]], %[[alloca_0]][] : memref<i32> 
//CHECK:     } 
//CHECK:   } 
//CHECK:   return 
//CHECK: } 

void foo2() {
  int a = 2;
  int b = 0;
  if (a < 3) {
    b++;
  }
}

//CHECK: func.func @foo2() {
//CHECK:   %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:   %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:   %[[C2_I32:.+]] = arith.constant 2 : i32 
//CHECK:   memref.store %[[C2_I32]], %[[alloca]][] : memref<i32> 
//CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32 
//CHECK:   memref.store %[[C0_I32]], %[[alloca_0]][] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32> 
//CHECK:     %[[C3_I32:.+]] = arith.constant 3 : i32 
//CHECK:     %[[ONE:.+]] = arith.cmpi slt, %[[ZERO]], %[[C3_I32]] : i32 
//CHECK:     %[[TWO:.+]] = arith.extui %[[ONE]] : i1 to i32 
//CHECK:     %[[C0_I32_1]] = arith.constant 0 : i32 
//CHECK:     %[[THREE:.+]] = arith.cmpi ne, %[[TWO]], %[[C0_I32_1]] : i32 
//CHECK:     %[[FOUR:.+]] = arith.extui %[[THREE]] : i1 to i8 
//CHECK:     %[[FIVE]] = arith.trunci %[[FOUR]] : i8 to i1 
//CHECK:     scf.if %[[FIVE]] {
//CHECK:       %[[SIX:.+]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:       %[[C1_I32:.+]] = arith.constant 1 : i32 
//CHECK:       %[[SEVEN:.+]] = arith.addi %[[SIX]], %[[C1_I32]] : i32 
//CHECK:       memref.store %[[SEVEN]], %[[alloca_0]][] : memref<i32> 
//CHECK:     } 
//CHECK:   } 
//CHECK:   return 
//CHECK: } 

void foo3() {
  int a = 2;
  int b = 0;
  if (a < 3) {
    int c = 1;
    if (c > 2) {
      b++;
    } else {
      b--;
    }
  }
}


//CHECK: func.func @foo3() {
//CHECK:   %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %[[C2_I32:.+]] = arith.constant 2 : i32 
//CHECK:   memref.store %[[C2_I32]], %[[alloca]][] : memref<i32> 
//CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32 
//CHECK:   memref.store %[[C0_I32]], %[[alloca_0]][] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32> 
//CHECK:     %[[C3_I32:.+]] = arith.constant 3 : i32 
//CHECK:     %[[ONE:.+]] = arith.cmpi slt, %[[ZERO]], %[[C3_I32]] : i32 
//CHECK:     %[[TWO:.+]] = arith.extui %[[ONE]] : i1 to i32 
//CHECK:     %[[C0_I32_1:.+]] = arith.constant 0 : i32 
//CHECK:     %[[THREE:.+]] = arith.cmpi ne, %[[TWO:.+]], %[[C0_I32_1]] : i32 
//CHECK:     %[[FOUR:.+]] = arith.extui %[[THREE]] : i1 to i8 
//CHECK:     %[[FIVE]] = arith.trunci %[[FOUR]] : i8 to i1 
//CHECK:     scf.if %[[FIVE]] {
//CHECK:       %[[alloca_2:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:       %[[C1_I32:.+]] = arith.constant 1 : i32 
//CHECK:       memref.store %[[C1_I32]], %[[alloca_2]][] : memref<i32> 
//CHECK:       memref.alloca_scope  {
//CHECK:         %[[SIX:.+]] = memref.load %[[alloca_2]][] : memref<i32> 
//CHECK:         %[[C2_I32_3:.+]] = arith.constant 2 : i32 
//CHECK:         %[[SEVEN:.+]] = arith.cmpi sgt, %[[SIX]], %[[C2_I32_3]] : i32 
//CHECK:         %[[EIGHT:.+]] = arith.extui %[[SEVEN]] : i1 to i32 
//CHECK:         %[[C0_I32_4:.+]] = arith.constant 0 : i32 
//CHECK:         %[[NINE:.+]] = arith.cmpi ne, %[[EIGHT]], %[[C0_I32_4]] : i32 
//CHECK:         %[[TEN:.+]] = arith.extui %[[NINE]] : i1 to i8 
//CHECK:         %[[ELEVEN:.+]] = arith.trunci %[[TEN]] : i8 to i1 
//CHECK:         scf.if %[[ELEVEN]] {
//CHECK:           %[[TWELVE:.+]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:           %[[C1_I32_5:.+]] = arith.constant 1 : i32 
//CHECK:           %[[THIRTEEN:.+]] = arith.addi %[[TWELVE]], %[[C1_I32_5]] : i32 
//CHECK:           memref.store %[[THIRTEEN]], %[[alloca_0]][] : memref<i32> 
//CHECK:         } else {
//CHECK:           %[[TWELVE:.+]] = memref.load %[[alloca_0]][] : memref<i32> 
//CHECK:           %[[C1_I32_5:.+]] = arith.constant 1 : i32 
//CHECK:           %[[THIRTEEN:.+]] = arith.subi %[[TWELVE]], %[[C1_I32_5]] : i32 
//CHECK:           memref.store %[[THIRTEEN]], %[[alloca_0]][] : memref<i32> 
//CHECK:         } 
//CHECK:       } 
//CHECK:     } 
//CHECK:   } 
//CHECK:   return 
//CHECK: } 
