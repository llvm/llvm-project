// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void for_continue() {
  for (int i = 0; i < 100; i++)
    continue;

  // CHECK: scf.while : () -> () {
  // CHECK:   %[[IV:.+]] = memref.load %alloca[{{%c0(_[0-9]+)?}}]
  // CHECK:   %[[CMP:.+]] = arith.cmpi slt, %[[IV]], %c100_i32
  // CHECK:   scf.condition(%[[CMP]])
  // CHECK: } do {
  // CHECK:   %[[IV2:.+]] = memref.load %alloca[{{%c0(_[0-9]+)?}}]
  // CHECK:   %[[ONE:.+]] = arith.constant 1
  // CHECK:   %[[CMP2:.+]] = arith.addi %[[IV2]], %[[ONE]]
  // CHECK:   memref.store %[[CMP2]], %alloca[{{%c0(_[0-9]+)?}}]
  // CHECK:   scf.yield
  // CHECK: }
}
