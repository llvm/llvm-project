// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

// Fewer than 8 nonzero leading elements: individual scalar fields + zero tail.
int sparse[100] = {1, 2, 3};

// Eight or more nonzero leading elements: a dense leading array field.
int dense[100] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// Pointer-into-array constants must stay in bounds after the split: a view into
// the zero tail and a view into the leading init region.
int *tail = &sparse[50];
int *head = &sparse[2];

// The split composes through a record: the nested array splits and the record
// type follows.
struct S {
  int arr[100];
};
struct S nested = {{1, 2, 3}};

// An array of such records: uniform element types keep an array.
struct S aos[2] = {{{1, 2, 3}}, {{4, 5, 6}}};

// CIR: cir.global external @sparse = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i], trailing_zeros> : !cir.array<!s32i x 100>
// CIR: cir.global external @dense = #cir.const_array<[#cir.int<1> : !s32i, {{.*}}#cir.int<10> : !s32i], trailing_zeros> : !cir.array<!s32i x 100>
// CIR: cir.global external @tail = #cir.global_view<@sparse, [50 : i32]> : !cir.ptr<!s32i>
// CIR: cir.global external @head = #cir.global_view<@sparse, [2 : i32]> : !cir.ptr<!s32i>
// CIR: cir.global external @nested = #cir.const_record<{#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i], trailing_zeros> : !cir.array<!s32i x 100>}> : !rec_S
// CIR: cir.global external @aos = #cir.const_array<[#cir.const_record<{{.*}}> : !cir.array<!rec_S x 2>

// LLVM: @sparse = global <{ i32, i32, i32, [97 x i32] }> <{ i32 1, i32 2, i32 3, [97 x i32] zeroinitializer }>, align 16
// LLVM: @dense = global <{ [10 x i32], [90 x i32] }> <{ [10 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10], [90 x i32] zeroinitializer }>, align 16

// LLVM-CIR: @tail = global ptr getelementptr inbounds nuw (i8, ptr @sparse, i64 200), align 8
// OGCG: @tail = global ptr getelementptr (i8, ptr @sparse, i64 200), align 8
// LLVM-CIR: @head = global ptr getelementptr inbounds nuw (i8, ptr @sparse, i64 8), align 8
// OGCG: @head = global ptr getelementptr (i8, ptr @sparse, i64 8), align 8

// LLVM: @nested = global { <{ i32, i32, i32, [97 x i32] }> } { <{ i32, i32, i32, [97 x i32] }> <{ i32 1, i32 2, i32 3, [97 x i32] zeroinitializer }> }, align 4
// LLVM: @aos = global [2 x { <{ i32, i32, i32, [97 x i32] }> }] [{{.*}}i32 1, i32 2, i32 3{{.*}}i32 4, i32 5, i32 6{{.*}}], align 16
