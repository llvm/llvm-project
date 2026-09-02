// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

// CIR: !rec_S = !cir.struct<"S" {!s32i, !cir.array<!s8i x 0>}>
// CIR: !rec_T = !cir.struct<"T" {!cir.ptr<!s32i>, !cir.array<!s32i x 0>}>

// 's1' lowers via the bulk constant-record path (LowerToLLVM.cpp ~line 2585):
// every member can be lowered to a constant attribute.
struct S { int n; char data[]; };
struct S s1 = { 3, { 'a', 'b', 'c' } };

// CIR: cir.global external @s1 = #cir.const_record<{#cir.int<3> : !s32i, #cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i]> : !cir.array<!s8i x 3>}> : !rec_S
// LLVM: @s1 = global <{ i32, [3 x i8] }> <{ i32 3, [3 x i8] c"abc" }>

struct S s2 = { 3, { 'a', 'b', 'c', 0, 0, 0, 0 } };
// CIR: cir.global external @s2 = #cir.const_record<{#cir.int<3> : !s32i, #cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i], trailing_zeros> : !cir.array<!s8i x 7>}> : !rec_S
// LLVM: @s2 = global <{ i32, [7 x i8] }> <{ i32 3, [7 x i8] c"abc\00\00\00\00" }>

struct S s3 = {3};
// CIR: cir.global external @s3 = #cir.const_record<{#cir.int<3> : !s32i, #cir.zero : !cir.array<!s8i x 0>}> : !rec_S
// LLVM: @s3 = global %struct.S { i32 3, [0 x i8] zeroinitializer }

int arr[4];

struct T { int *p; int data[]; };
struct T t1 = { &arr[2], { 1, 2, 3 } };

// CIR: cir.global external @t1 = #cir.const_record<{#cir.global_view<@arr, [2 : i32]> : !cir.ptr<!s32i>, #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>}> : !rec_T
// LLVM: @t1 = global <{ ptr, [3 x i32] }> <{ ptr getelementptr {{.*}}(i8, ptr @arr, i64 8), [3 x i32] [i32 1, i32 2, i32 3] }>

struct T t2 = { &arr[2], { 1, 2, 3, 0, 0, 0, 0, 0 } };
// CIR: cir.global external @t2 = #cir.const_record<{#cir.global_view<@arr, [2 : i32]> : !cir.ptr<!s32i>, #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i], trailing_zeros> : !cir.array<!s32i x 8>}> : !rec_T
// LLVM: @t2 = global { ptr, [8 x i32] } { ptr getelementptr {{.*}}(i8, ptr @arr, i64 8), [8 x i32] [i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0] }

struct T t3 = { &arr[2], { 0, 0, 0, 0, 0, 0, 0, 0 } };
// CIR: cir.global external @t3 = #cir.const_record<{#cir.global_view<@arr, [2 : i32]> : !cir.ptr<!s32i>, #cir.zero : !cir.array<!s32i x 8>}> : !rec_T
// LLVM: @t3 = global { ptr, [8 x i32] } { ptr getelementptr {{.*}}(i8, ptr @arr, i64 8), [8 x i32] zeroinitializer }

struct V { int tag; int count; int data[]; };
struct V v1 = {1, 3, {10, 20, 30}};
// CIR: cir.global external @v1 = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<3> : !s32i, #cir.const_array<[#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.int<30> : !s32i]> : !cir.array<!s32i x 3>}> : !rec_V
// LLVM: @v1 = global { i32, i32, [3 x i32] } { i32 1, i32 3, [3 x i32] [i32 10, i32 20, i32 30] }


struct Padded { char c; int n; int data[]; };
struct Padded padded1 = {'x', 10, {'a', 'b'}};
// CIR: cir.global external @padded1 = #cir.const_record<{#cir.int<120> : !s8i, #cir.int<10> : !s32i, #cir.const_array<[#cir.int<97> : !s32i, #cir.int<98> : !s32i]> : !cir.array<!s32i x 2>}> : !rec_Padded
// These differ because classic-codegen has put the padding in place.  However,
// they should still match anyway, since llvm aligns each field itself.
// LLVMCIR: @padded1 = global { i8, i32, [2 x i32] } { i8 120, i32 10, [2 x i32] [i32 97, i32 98] }
// OGCG:    @padded1 = global { i8, [3 x i8], i32, [2 x i32] } { i8 120, [3 x i8] zeroinitializer, i32 10, [2 x i32] [i32 97, i32 98] }

struct __attribute__((packed)) Packed { int n; char data[]; };
struct Packed packed1 = {1, {'a', 'b'}};
// CIR: cir.global external @packed1 = #cir.const_record<{#cir.int<1> : !s32i, #cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i]> : !cir.array<!s8i x 2>}> : !rec_Packed
// LLVM: @packed1 = global <{ i32, [2 x i8] }> <{ i32 1, [2 x i8] c"ab" }>
