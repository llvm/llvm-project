// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -mmlir -mlir-print-ir-before=cir-canonicalize -Wno-unused-value -Wno-constant-conversion -Wno-literal-conversion -fclangir -emit-cir %s -o %t.cir 2> %t-before.cir
// Note: The "before" check intentionally uses the same CIR check prefixes to
//       verify that the IR is folded during code generation and not during
//       canonicalization. In a few cases, it is necessary to match an unused
//       duplicate constant that will be erased during canonicalization.
// RUN: FileCheck --input-file=%t-before.cir %s -check-prefixes=CIR,CIR-DUP,ALL
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,ALL
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -Wno-constant-conversion -Wno-literal-conversion -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM_OGCG,ALL
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -Wno-constant-conversion -Wno-literal-conversion -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM_OGCG,ALL

void fold_int_not() {
// ALL: fold_int_not
  int n;
  short s;
  unsigned u;

  n = ~0;
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i32 -1, ptr %{{.*}}

  n = ~1;
  // CIR: %[[MINUS_TWO:.*]] = cir.const #cir.int<-2> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_TWO]], %{{.*}}
  // LLVM_OGCG: store i32 -2, ptr %{{.*}}

  n = ~0xFFFFFFFE;
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i32 1, ptr %{{.*}}
}

void fold_int_plus() {
// ALL: fold_int_plus
  int n;
  short s;
  unsigned u;

  n = +1;
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i32 1, ptr %{{.*}}

  n = +2;
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[TWO]], %{{.*}}
  // LLVM_OGCG: store i32 2, ptr %{{.*}}

  n = +(-3);
  // CIR: %[[MINUS_THREE:.*]] = cir.const #cir.int<-3> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_THREE]], %{{.*}}
  // LLVM_OGCG: store i32 -3, ptr %{{.*}}

  n = +(0x1FFFFFFFF);
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i32 -1, ptr %{{.*}}

  s = +1;
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i16 1, ptr %{{.*}}

  s = +2;
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[TWO]], %{{.*}}
  // LLVM_OGCG: store i16 2, ptr %{{.*}}

  s = +(-3);
  // CIR: %[[MINUS_THREE:.*]] = cir.const #cir.int<-3> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_THREE]], %{{.*}}
  // LLVM_OGCG: store i16 -3, ptr %{{.*}}

  s = +(0x1FFFF);
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i16 -1, ptr %{{.*}}

  u = +1;
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i32 1, ptr %{{.*}}

  u = +2;
  // CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[TWO]], %{{.*}}
  // LLVM_OGCG: store i32 2, ptr %{{.*}}

  u = +(-3);
  // CIR: %[[MINUS_THREE:.*]] = cir.const #cir.int<4294967293> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_THREE]], %{{.*}}
  // LLVM_OGCG: store i32 -3, ptr %{{.*}}

  u = +(0x1FFFFFFFF);
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<4294967295> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i32 -1, ptr %{{.*}}
}

void fold_int_minus() {
// ALL: fold_int_minus
  int n;
  short s;
  unsigned u;

  n = -1;
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i32 -1, ptr %{{.*}}

  n = -2;
  // CIR: %[[MINUS_TWO:.*]] = cir.const #cir.int<-2> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_TWO]], %{{.*}}
  // LLVM_OGCG: store i32 -2, ptr %{{.*}}

  n = -(-3);
  // CIR-DUP: %[[UNUSED_THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[THREE]], %{{.*}}
  // LLVM_OGCG: store i32 3, ptr %{{.*}}

  n = -(0x1FFFFFFFF);
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i32 1, ptr %{{.*}}

  s = -1;
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i16 -1, ptr %{{.*}}

  s = -2;
  // CIR: %[[MINUS_TWO:.*]] = cir.const #cir.int<-2> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_TWO]], %{{.*}}
  // LLVM_OGCG: store i16 -2, ptr %{{.*}}

  s = -(-3);
  // CIR-DUP: %[[UNUSED_THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[THREE]], %{{.*}}
  // LLVM_OGCG: store i16 3, ptr %{{.*}}

  s = -(0x1FFFF);
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s16i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i16 1, ptr %{{.*}}

  u = -1;
  // CIR: %[[MINUS_ONE:.*]] = cir.const #cir.int<4294967295> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_ONE]], %{{.*}}
  // LLVM_OGCG: store i32 -1, ptr %{{.*}}

  u = -2;
  // CIR: %[[MINUS_TWO:.*]] = cir.const #cir.int<4294967294> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_TWO]], %{{.*}}
  // LLVM_OGCG: store i32 -2, ptr %{{.*}}

  u = -(-3);
  // CIR-DUP: %[[UNUSED_THREE:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[THREE:.*]] = cir.const #cir.int<3> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[THREE]], %{{.*}}
  // LLVM_OGCG: store i32 3, ptr %{{.*}}

  u = -(0x1FFFFFFFF);
  // CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[ONE]], %{{.*}}
  // LLVM_OGCG: store i32 1, ptr %{{.*}}
}

void fold_float_plus() {
// ALL: fold_float_plus
  float f;
  double d;

  f = +2.0f;
  // CIR: %[[TWO:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[TWO]], %{{.*}}
  // LLVM_OGCG: store float 2.000000e+00, ptr %{{.*}}

  f = +(-3.0f);
  // CIR: %[[MINUS_THREE:.*]] = cir.const #cir.fp<-3.000000e+00> : !cir.float
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_THREE]], %{{.*}}
  // LLVM_OGCG: store float -3.000000e+00, ptr %{{.*}}

  d = +2.0;
  // CIR: %[[TWO:.*]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[TWO]], %{{.*}}
  // LLVM_OGCG: store double 2.000000e+00, ptr %{{.*}}

  d = +(-3.0);
  // CIR: %[[MINUS_THREE:.*]] = cir.const #cir.fp<-3.000000e+00> : !cir.double
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_THREE]], %{{.*}}
  // LLVM_OGCG: store double -3.000000e+00, ptr %{{.*}}
}

void fold_float_minus() {
// ALL: fold_float_minus
  float f;
  double d;

  f = -2.0f;
  // CIR: %[[MINUS_TWO:.*]] = cir.const #cir.fp<-2.000000e+00> : !cir.float
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_TWO]], %{{.*}}
  // LLVM_OGCG: store float -2.000000e+00, ptr %{{.*}}

  f = -(-3.0f);
  // CIR-DUP: %[[UNUSED_THREE:.*]] = cir.const #cir.fp<3.000000e+00> : !cir.float
  // CIR: %[[THREE:.*]] = cir.const #cir.fp<3.000000e+00> : !cir.float
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[THREE]], %{{.*}}
  // LLVM_OGCG: store float 3.000000e+00, ptr %{{.*}}

  d = -2.0;
  // CIR: %[[MINUS_TWO:.*]] = cir.const #cir.fp<-2.000000e+00> : !cir.double
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[MINUS_TWO]], %{{.*}}
  // LLVM_OGCG: store double -2.000000e+00, ptr %{{.*}}

  d = -(-3.0);
  // CIR-DUP: %[[UNUSED_THREE:.*]] = cir.const #cir.fp<3.000000e+00> : !cir.double
  // CIR: %[[THREE:.*]] = cir.const #cir.fp<3.000000e+00> : !cir.double
  // CIR-NOT: cir.unary
  // CIR: cir.store{{.*}} %[[THREE]], %{{.*}}
  // LLVM_OGCG: store double 3.000000e+00, ptr %{{.*}}
}
