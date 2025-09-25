// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mms-bitfields -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mms-bitfields -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mms-bitfields -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct s1 {
  int       f32 : 2;
  long long f64 : 30;
} s1;

// CIR-DAG: !rec_s1 = !cir.record<struct "s1" {!s32i, !s64i}>
// LLVM-DAG: %struct.s1 = type { i32, i64 }
// OGCG-DAG: %struct.s1 = type { i32, i64 }

struct s2 {
    int a : 24;
    char b;
    int c : 30;
} Clip;

// CIR-DAG: !rec_s2 = !cir.record<struct "s2" {!s32i, !s8i, !s32i}>
// LLVM-DAG: %struct.s2 = type { i32, i8, i32 }
// OGCG-DAG: %struct.s2 = type { i32, i8, i32 }

struct s3 {
    int a : 18;
    int   :  0;
    int c : 14;
} zero_bit;

// CIR-DAG:  !rec_s3 = !cir.record<struct "s3" {!s32i, !s32i}>
// LLVM-DAG: %struct.s3 = type { i32, i32 }
// OGCG-DAG: %struct.s3 = type { i32, i32 }

#pragma pack (push,1)

struct Inner {
  unsigned int A :  1;
  unsigned int B :  1;
  unsigned int C :  1;
  unsigned int D : 30;
} Inner;

#pragma pack (pop)

// CIR-DAG: !rec_Inner = !cir.record<struct "Inner" {!u32i, !u32i}>
// LLVM-DAG: %struct.Inner = type { i32, i32 }
// OGCG-DAG: %struct.Inner = type { i32, i32 }

#pragma pack(push, 1)

union HEADER {
  struct A {
    int                                         :  3;  // Bits 2:0
    int a                                       :  9;  // Bits 11:3
    int                                         :  12;  // Bits 23:12
    int b                                       :  17;  // Bits 40:24
    int                                         :  7;  // Bits 47:41
    int c                                       :  4;  // Bits 51:48
    int                                         :  4;  // Bits 55:52
    int d                                       :  3;  // Bits 58:56
    int                                         :  5;  // Bits 63:59
  } Bits;
} HEADER;

#pragma pack(pop)

// CIR-DAG: !rec_A = !cir.record<struct "A" {!s32i, !s32i, !s32i}>
// CIR-DAG: !rec_HEADER = !cir.record<union "HEADER" {!rec_A}>
// LLVM-DAG: %struct.A = type { i32, i32, i32 }
// LLVM-DAG: %union.HEADER = type { %struct.A }
// OGCG-DAG: %struct.A = type { i32, i32, i32 }
// OGCG-DAG: %union.HEADER = type { %struct.A }
