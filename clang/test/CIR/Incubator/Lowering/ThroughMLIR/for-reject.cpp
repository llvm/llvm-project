// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void f() {}

void reject_test1() {
  for (int i = 0; i < 100; i++, f());
  // CHECK: %[[ALLOCA:.+]] = memref.alloca
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: memref.store %[[ZERO]], %[[ALLOCA]]
  // CHECK: %[[HUNDRED:.+]] = arith.constant 100
  // CHECK: scf.while : () -> () {
  // CHECK:   %[[TMP:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[TMP1:.+]] = arith.cmpi slt, %0, %[[HUNDRED]]
  // CHECK:   scf.condition(%[[TMP1]])
  // CHECK: } do {
  // CHECK:   %[[TMP2:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE:.+]] = arith.constant 1
  // CHECK:   %[[TMP3:.+]] = arith.addi %[[TMP2]], %[[ONE]]
  // CHECK:   memref.store %[[TMP3]], %[[ALLOCA]]
  // CHECK:   func.call @_Z1fv()
  // CHECK:   scf.yield
  // CHECK: }
}

void reject_test2() {
  for (int i = 0; i < 100; i++, i++);
  // CHECK: %[[ALLOCA:.+]] = memref.alloca
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: memref.store %[[ZERO]], %[[ALLOCA]]
  // CHECK: %[[HUNDRED:.+]] = arith.constant 100
  // CHECK: scf.while : () -> () {
  // CHECK:   %[[TMP:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[TMP2:.+]] = arith.cmpi slt, %[[TMP]], %[[HUNDRED]]
  // CHECK:   scf.condition(%[[TMP2]])
  // CHECK: } do {
  // CHECK:   %[[TMP3:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE:.+]] = arith.constant 1
  // CHECK:   %[[ADD:.+]] = arith.addi %[[TMP3]], %[[ONE]]
  // CHECK:   memref.store %[[ADD]], %[[ALLOCA]]
  // CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE2:.+]] = arith.constant 1
  // CHECK:   %[[ADD2:.+]] = arith.addi %[[LOAD]], %[[ONE2]]
  // CHECK:   memref.store %[[ADD2]], %[[ALLOCA]]
  // CHECK:   scf.yield
  // CHECK: }
}

void reject_test3() {
  int i;
  for (i = 0; i < 100; i++);
  i += 10;
  // CHECK: %[[ALLOCA:.+]] = memref.alloca()
  // CHECK: memref.alloca_scope  {
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: memref.store %[[ZERO]], %[[ALLOCA]]
  // CHECK: %[[HUNDRED:.+]] = arith.constant 100
  // CHECK: scf.while : () -> () {
  // CHECK:   %[[TMP:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[TMP2:.+]] = arith.cmpi slt, %[[TMP]], %[[HUNDRED]]
  // CHECK:   scf.condition(%[[TMP2]])
  // CHECK: } do {
  // CHECK:   %[[TMP3:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE:.+]] = arith.constant 1
  // CHECK:   %[[ADD:.+]] = arith.addi %[[TMP3]], %[[ONE]]
  // CHECK:   memref.store %[[ADD]], %[[ALLOCA]]
  // CHECK:   scf.yield
  // CHECK: }
  // CHECK: }
  // CHECK: %[[TEN:.+]] = arith.constant 10
  // CHECK: %[[TMP4:.+]] = memref.load %[[ALLOCA]]
  // CHECK: %[[TMP5:.+]] = arith.addi %[[TMP4]], %[[TEN]]
  // CHECK: memref.store %[[TMP5]], %[[ALLOCA]]
}
