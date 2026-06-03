// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// CIR uses the logical CIR record type for the const_record; alignment-induced
// padding bytes between fields are implicit (no separate padding member). OG
// emits an anonymous struct with explicit padding members; the resulting byte
// content is the same.

struct padding_after_field {
  char c;
  int i;
};

struct bitfield_with_padding {
  unsigned int a : 3;
  unsigned int b : 5;
  int c;
};

struct tail_padding {
  int a;
  char b;
};

struct multiple_padding {
  char a;
  short b;
  long long c;
};

void test_zero_init_padding(void) {
  static const struct padding_after_field paf = {1, 42};
  static const struct bitfield_with_padding bfp = {1, 2, 99};
  static const struct tail_padding tp = {10, 20};
  static const struct multiple_padding mp = {5, 10, 100};
}

// CIR-DAG: cir.global "private" constant internal dso_local @test_zero_init_padding.paf = #cir.const_record<{#cir.int<1> : !s8i, #cir.int<42> : !s32i}> : !rec_padding_after_field
// CIR-DAG: cir.global "private" constant internal dso_local @test_zero_init_padding.bfp = #cir.const_record<{#cir.int<17> : !u8i, #cir.int<99> : !s32i}> : !rec_bitfield_with_padding
// CIR-DAG: cir.global "private" constant internal dso_local @test_zero_init_padding.tp = #cir.const_record<{#cir.int<10> : !s32i, #cir.int<20> : !s8i}> : !rec_tail_padding
// CIR-DAG: cir.global "private" constant internal dso_local @test_zero_init_padding.mp = #cir.const_record<{#cir.int<5> : !s8i, #cir.int<10> : !s16i, #cir.int<100> : !s64i}> : !rec_multiple_padding

// CIR-LABEL: cir.func {{.*}}@test_zero_init_padding
// CIR:   cir.return

// LLVM-DAG: @test_zero_init_padding.paf = internal constant %struct.padding_after_field { i8 1, i32 42 }
// LLVM-DAG: @test_zero_init_padding.bfp = internal constant %struct.bitfield_with_padding { i8 17, i32 99 }
// LLVM-DAG: @test_zero_init_padding.tp = internal constant %struct.tail_padding { i32 10, i8 20 }
// LLVM-DAG: @test_zero_init_padding.mp = internal constant %struct.multiple_padding { i8 5, i16 10, i64 100 }

// LLVM-LABEL: define{{.*}} void @test_zero_init_padding
// LLVM:   ret void

// OGCG-DAG: @test_zero_init_padding.paf = internal constant { i8, [3 x i8], i32 } { i8 1, [3 x i8] zeroinitializer, i32 42 }
// OGCG-DAG: @test_zero_init_padding.bfp = internal constant { i8, [3 x i8], i32 } { i8 17, [3 x i8] zeroinitializer, i32 99 }
// OGCG-DAG: @test_zero_init_padding.tp = internal constant { i32, i8, [3 x i8] } { i32 10, i8 20, [3 x i8] zeroinitializer }
// OGCG-DAG: @test_zero_init_padding.mp = internal constant { i8, i8, i16, [4 x i8], i64 } { i8 5, i8 0, i16 10, [4 x i8] zeroinitializer, i64 100 }

// OGCG-LABEL: define{{.*}} void @test_zero_init_padding
// OGCG:   ret void
