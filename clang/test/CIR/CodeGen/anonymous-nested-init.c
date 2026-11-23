// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Anonymous struct as a field.
struct StructWithAnon {
  struct { int x; int y; } inner;
  int z;
};
struct StructWithAnon g_anon_struct = { {1, 2}, 3 };
// CIR: cir.global external @g_anon_struct = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i}> : !rec_anon2E0, #cir.int<3> : !s32i}> : !rec_StructWithAnon
// LLVM: @g_anon_struct = global %struct.StructWithAnon { %struct.anon{{.*}} { i32 1, i32 2 }, i32 3 }

struct StructWithAnonUnion {
  union { int i; float f; } u;
  int n;
};
struct StructWithAnonUnion g_anon_union = { {.f = 1.5f}, 42 };

// CIR: cir.global external @g_anon_union = #cir.const_record<{#cir.const_record<{#cir.fp<1.500000e+00> : !cir.float}> : !rec_anon2E1, #cir.int<42> : !s32i}> : !rec_StructWithAnonUnion
// Union lowering chooses to lower this as an int instead of a float since the storage type is 'int', but active type is 'float'.
// LLVM:    @g_anon_union = global { { float }, i32 } { { float } { float 1.500000e+00 }, i32 42 }

struct StructWithAnonBitfields {
  int leading;
  struct { unsigned a : 4; unsigned b : 4; } bits;
  int trailing;
};
struct StructWithAnonBitfields g_anon_bits = { 7, {0xA, 0x5}, 9 };

// CIR: cir.global external @g_anon_bits = #cir.const_record<{#cir.int<7> : !s32i, #cir.const_record<{#cir.int<90> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_anon2E2, #cir.int<9> : !s32i}> : !rec_StructWithAnonBitfields
// LLVM: @g_anon_bits = global %struct.StructWithAnonBitfields { i32 7, %struct.anon{{.*}} { i8 90, [3 x i8] zeroinitializer }, i32 9 }
