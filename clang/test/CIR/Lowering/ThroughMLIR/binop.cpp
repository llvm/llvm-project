// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void testSignedIntBinOps(int a, int b) {
  int x = a * b;
  x = x / b;
  x = x % b;
  x = x + b;
  x = x - b;
  x = x >> b;
  x = x << b;
  x = x & b;
  x = x ^ b;
  x = x | b;
}

// CHECK: func.func @_Z19testSignedIntBinOpsii
// CHECK:   %[[VAR2:.*]] = arith.muli %[[VAR0:.*]], %[[VAR1:.*]] : i32
// CHECK:   %[[VAR5:.*]] = arith.divsi %[[VAR3:.*]], %[[VAR4:.*]] : i32
// CHECK:   %[[VAR8:.*]] = arith.remsi %[[VAR6:.*]], %[[VAR7:.*]] : i32
// CHECK:   %[[VAR11:.*]] = arith.addi %[[VAR9:.*]], %[[VAR10:.*]] : i32
// CHECK:   %[[VAR14:.*]] = arith.subi %[[VAR12:.*]], %[[VAR13:.*]] : i32
// CHECK:   %[[VAR18:.*]] = arith.shrsi %[[VAR15:.*]], %[[VAR16:.*]] : i32
// CHECK:   %[[VAR22:.*]] = arith.shli %[[VAR19:.*]], %[[VAR20:.*]] : i32
// CHECK:   %[[VAR25:.*]] = arith.andi %[[VAR23:.*]], %[[VAR24:.*]] : i32
// CHECK:   %[[VAR28:.*]] = arith.xori %[[VAR26:.*]], %[[VAR27:.*]] : i32
// CHECK:   %[[VAR31:.*]] = arith.ori %[[VAR29:.*]], %[[VAR30:.*]] : i32
// CHECK: }

void testUnSignedIntBinOps(unsigned a, unsigned b) {
  unsigned x = a * b;
  x = x / b;
  x = x % b;
  x = x + b;
  x = x - b;
  x = x >> b;
  x = x << b;
  x = x & b;
  x = x ^ b;
  x = x | b;
}

// CHECK: func.func @_Z21testUnSignedIntBinOpsjj
// CHECK:   %[[VAR2:.*]] = arith.muli %[[VAR0:.*]], %[[VAR1:.*]] : i32
// CHECK:   %[[VAR5:.*]] = arith.divui %[[VAR3:.*]], %[[VAR4:.*]] : i32
// CHECK:   %[[VAR8:.*]] = arith.remui %[[VAR6:.*]], %[[VAR7:.*]] : i32
// CHECK:   %[[VAR11:.*]] = arith.addi %[[VAR9:.*]], %[[VAR10:.*]] : i32
// CHECK:   %[[VAR14:.*]] = arith.subi %[[VAR12:.*]], %[[VAR13:.*]] : i32
// CHECK:   %[[VAR18:.*]] = arith.shrui %[[VAR15:.*]], %[[VAR16:.*]] : i32
// CHECK:   %[[VAR22:.*]] = arith.shli %[[VAR19:.*]], %[[VAR20:.*]] : i32
// CHECK:   %[[VAR25:.*]] = arith.andi %[[VAR23:.*]], %[[VAR24:.*]] : i32
// CHECK:   %[[VAR28:.*]] = arith.xori %[[VAR26:.*]], %[[VAR27:.*]] : i32
// CHECK:   %[[VAR31:.*]] = arith.ori %[[VAR29:.*]], %[[VAR30:.*]] : i32
// CHECK: }

void testFloatingPointBinOps(float a, float b, double c, double d) {
  float e = a * b;
  e = a / b;
  e = a + b;
  e = a - b;

  double f = a * b;
  f = c * d;
  f = c / d;
  f = c + d;
  f = c - d;
}

// CHECK: func.func @_Z23testFloatingPointBinOpsffdd
// CHECK:   %[[VAR2:.*]] = arith.mulf %[[VAR0:.*]], %[[VAR1:.*]] : f32
// CHECK:   %[[VAR5:.*]] = arith.divf %[[VAR3:.*]], %[[VAR4:.*]] : f32
// CHECK:   %[[VAR8:.*]] = arith.addf %[[VAR6:.*]], %[[VAR7:.*]] : f32
// CHECK:   %[[VAR11:.*]] = arith.subf %[[VAR9:.*]], %[[VAR10:.*]] : f32
// CHECK:   %[[VAR14:.*]] = arith.mulf %[[VAR12:.*]], %[[VAR13:.*]] : f64
// CHECK:   %[[VAR18:.*]] = arith.divf %[[VAR16:.*]], %[[VAR17:.*]] : f64
// CHECK:   %[[VAR22:.*]] = arith.addf %[[VAR20:.*]], %[[VAR21:.*]] : f64
// CHECK:   %[[VAR26:.*]] = arith.subf %[[VAR24:.*]], %[[VAR25:.*]] : f64
