// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Globals whose layout includes anonymous nested aggregates. The constant
// emitter has to recurse into the anonymous record's CIR layout and produce a
// nested const_record whose member list matches that anonymous record's CIR
// type.

// Anonymous struct as a field.
struct StructWithAnon {
  struct { int x; int y; } inner;
  int z;
};
struct StructWithAnon g_anon_struct = { {1, 2}, 3 };

// CIR: cir.global external @g_anon_struct
// CIR-SAME: #cir.int<1> : !s32i, #cir.int<2> : !s32i
// CIR-SAME: #cir.int<3> : !s32i
// LLVM: @g_anon_struct ={{.*}}i32 1, i32 2{{.*}}i32 3
// OGCG: @g_anon_struct ={{.*}}i32 1, i32 2{{.*}}i32 3

// Anonymous union as a field, initialized via designator.  Because CIR's
// UnionType carries a single storage member at the LLVM level, the active
// float value is encoded as the bit-equivalent integer (0x3FC00000 = 1.5f).
struct StructWithAnonUnion {
  union { int i; float f; } u;
  int n;
};
struct StructWithAnonUnion g_anon_union = { {.f = 1.5f}, 42 };

// LLVM: @g_anon_union ={{.*}}i32 1069547520{{.*}}i32 42
// OGCG: @g_anon_union ={{.*}}float 1.500000e+00{{.*}}i32 42

// Anonymous struct holding bitfields, surrounded by a regular field.
struct StructWithAnonBitfields {
  int leading;
  struct { unsigned a : 4; unsigned b : 4; } bits;
  int trailing;
};
struct StructWithAnonBitfields g_anon_bits = { 7, {0xA, 0x5}, 9 };

// a=0xA in bits[0..3], b=0x5 in bits[4..7] -> packed = 0x5A = 90.
// LLVM: @g_anon_bits ={{.*}}i32 7{{.*}}i8 90{{.*}}i32 9
// OGCG: @g_anon_bits ={{.*}}i32 7{{.*}}i8 90{{.*}}i32 9
