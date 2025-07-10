// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef union {
  int x;
  int y : 4;
  int z : 8;
} demo;

// CIR:  !rec_demo = !cir.record<union "demo" {!s32i, !u8i, !u8i}>
// LLVM: %union.demo = type { i32 }
// OGCG: %union.demo = type { i32 }

typedef union {
  int x;
  int y : 3;
  int   : 0;
  int z : 2;
} zero_bit;

// CIR:  !rec_zero_bit = !cir.record<union "zero_bit" {!s32i, !u8i, !u8i}>
// LLVM: %union.zero_bit = type { i32 }
// OGCG: %union.zero_bit = type { i32 }

demo d;
zero_bit z;
