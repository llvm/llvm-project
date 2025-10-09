// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

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

// Type definitions for anonymous structs with padding
// CIR-DAG: !rec_anon_struct = !cir.record<struct  {!s8i, !u8i, !s16i, !cir.array<!u8i x 4>, !s64i}>
// CIR-DAG: !rec_anon_struct1 = !cir.record<struct  {!s32i, !s8i, !cir.array<!u8i x 3>}>
// CIR-DAG: !rec_anon_struct2 = !cir.record<struct  {!u8i, !cir.array<!u8i x 3>, !s32i}>
// CIR-DAG: !rec_anon_struct3 = !cir.record<struct  {!s8i, !cir.array<!u8i x 3>, !s32i}>

// paf: char + 3 bytes padding + int -> uses !rec_anon_struct3
// CIR-DAG: cir.global "private" internal dso_local @test_zero_init_padding.paf = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<1> : !s8i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>,
// CIR-DAG-SAME:   #cir.int<42> : !s32i
// CIR-DAG-SAME: }> : !rec_anon_struct3

// bfp: unsigned bitfield byte + 3 bytes padding + int -> uses !rec_anon_struct2
// CIR-DAG: cir.global "private" internal dso_local @test_zero_init_padding.bfp = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<17> : !u8i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>,
// CIR-DAG-SAME:   #cir.int<99> : !s32i
// CIR-DAG-SAME: }> : !rec_anon_struct2

// tp: int + char + 3 bytes tail padding -> uses !rec_anon_struct1
// CIR-DAG: cir.global "private" internal dso_local @test_zero_init_padding.tp = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<10> : !s32i,
// CIR-DAG-SAME:   #cir.int<20> : !s8i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>
// CIR-DAG-SAME: }> : !rec_anon_struct1

// mp: char + 1 byte padding + short + 4 bytes padding + long long -> uses !rec_anon_struct
// CIR-DAG: cir.global "private" internal dso_local @test_zero_init_padding.mp = #cir.const_record<{
// CIR-DAG-SAME:   #cir.int<5> : !s8i,
// CIR-DAG-SAME:   #cir.zero : !u8i,
// CIR-DAG-SAME:   #cir.int<10> : !s16i,
// CIR-DAG-SAME:   #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 4>,
// CIR-DAG-SAME:   #cir.int<100> : !s64i
// CIR-DAG-SAME: }> : !rec_anon_struct

// CIR-LABEL: cir.func {{.*}}@test_zero_init_padding
// CIR:   cir.return

// LLVM-DAG: @test_zero_init_padding.paf = internal global { i8, [3 x i8], i32 } { i8 1, [3 x i8] zeroinitializer, i32 42 }
// LLVM-DAG: @test_zero_init_padding.bfp = internal global { i8, [3 x i8], i32 } { i8 17, [3 x i8] zeroinitializer, i32 99 }
// LLVM-DAG: @test_zero_init_padding.tp = internal global { i32, i8, [3 x i8] } { i32 10, i8 20, [3 x i8] zeroinitializer }
// LLVM-DAG: @test_zero_init_padding.mp = internal global { i8, i8, i16, [4 x i8], i64 } { i8 5, i8 0, i16 10, [4 x i8] zeroinitializer, i64 100 }

// LLVM-LABEL: define{{.*}} void @test_zero_init_padding
// LLVM:   ret void

// OGCG-DAG: @test_zero_init_padding.paf = internal constant { i8, [3 x i8], i32 } { i8 1, [3 x i8] zeroinitializer, i32 42 }
// OGCG-DAG: @test_zero_init_padding.bfp = internal constant { i8, [3 x i8], i32 } { i8 17, [3 x i8] zeroinitializer, i32 99 }
// OGCG-DAG: @test_zero_init_padding.tp = internal constant { i32, i8, [3 x i8] } { i32 10, i8 20, [3 x i8] zeroinitializer }
// OGCG-DAG: @test_zero_init_padding.mp = internal constant { i8, i8, i16, [4 x i8], i64 } { i8 5, i8 0, i16 10, [4 x i8] zeroinitializer, i64 100 }

// OGCG-LABEL: define{{.*}} void @test_zero_init_padding
// OGCG:   ret void
