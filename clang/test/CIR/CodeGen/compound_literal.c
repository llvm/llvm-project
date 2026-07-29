// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int *p1 = (int[]){1, 2, 3};
// CIR: cir.global "private" internal @".compoundliteral" = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3> {alignment = 4 : i64}
// CIR: cir.global external @p1 = #cir.global_view<@".compoundliteral"> : !cir.ptr<!s32i> {alignment = 8 : i64}
// LLVM: @.compoundliteral = internal global [3 x i32] [i32 1, i32 2, i32 3], align 4
// LLVM: @p1 = global ptr @.compoundliteral, align 8

int *p2 = &(int){42};
// CIR: cir.global "private" internal @".compoundliteral.1" = #cir.int<42> : !s32i {alignment = 4 : i64}
// CIR: cir.global external @p2 = #cir.global_view<@".compoundliteral.1"> : !cir.ptr<!s32i> {alignment = 8 : i64}
// LLVM: @.compoundliteral.1 = internal global i32 42, align 4
// LLVM: @p2 = global ptr @.compoundliteral.1, align 8

struct S { int x, y; };
struct S *p3 = &(struct S){5, 10};
// CIR: cir.global "private" internal @".compoundliteral.2" = #cir.const_record<{#cir.int<5> : !s32i, #cir.int<10> : !s32i}> : !rec_S {alignment = 4 : i64}
// CIR: cir.global external @p3 = #cir.global_view<@".compoundliteral.2"> : !cir.ptr<!rec_S> {alignment = 8 : i64}
// LLVM: @.compoundliteral.2 = internal global %struct.S { i32 5, i32 10 }, align 4
// LLVM: @p3 = global ptr @.compoundliteral.2, align 8

int *p4[2] = { (int[]){1, 2}, (int[]){3, 4} };
// CIR: cir.global "private" internal @".compoundliteral.3" = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2> {alignment = 4 : i64}
// CIR: cir.global "private" internal @".compoundliteral.4" = #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2> {alignment = 4 : i64}
// CIR: cir.global external @p4 = #cir.const_array<[#cir.global_view<@".compoundliteral.3"> : !cir.ptr<!s32i>, #cir.global_view<@".compoundliteral.4"> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2> {alignment = 16 : i64}
// LLVM: @.compoundliteral.3 = internal global [2 x i32] [i32 1, i32 2], align 4
// LLVM: @.compoundliteral.4 = internal global [2 x i32] [i32 3, i32 4], align 4
// LLVM: @p4 = global [2 x ptr] [ptr @.compoundliteral.3, ptr @.compoundliteral.4], align 16

struct W { int *p; };
struct W p5 = { (int[]){10, 20} };
// CIR: cir.global "private" internal @".compoundliteral.5" = #cir.const_array<[#cir.int<10> : !s32i, #cir.int<20> : !s32i]> : !cir.array<!s32i x 2> {alignment = 4 : i64}
// CIR: cir.global external @p5 = #cir.const_record<{#cir.global_view<@".compoundliteral.5"> : !cir.ptr<!s32i>}> : !rec_W {alignment = 8 : i64}
// LLVM: @.compoundliteral.5 = internal global [2 x i32] [i32 10, i32 20], align 4
// LLVM: @p5 = global %struct.W { ptr @.compoundliteral.5 }, align 8

const int *p6 = (const int[]){1, 2, 3};
// CIR: cir.global "private" constant internal @".compoundliteral.6" = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3> {alignment = 4 : i64}
// CIR: cir.global external @p6 = #cir.global_view<@".compoundliteral.6"> : !cir.ptr<!s32i> {alignment = 8 : i64}
// LLVM: @.compoundliteral.6 = internal constant [3 x i32] [i32 1, i32 2, i32 3], align 4
// LLVM: @p6 = global ptr @.compoundliteral.6, align 8

char *p7 = (char[]){"hi"};
// CIR: cir.global "private" internal @".compoundliteral.7" = #cir.const_array<"hi" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3> {alignment = 1 : i64}
// CIR: cir.global external @p7 = #cir.global_view<@".compoundliteral.7"> : !cir.ptr<!s8i> {alignment = 8 : i64}
// LLVM: @.compoundliteral.7 = internal global [3 x i8] c"hi\00", align 1
// LLVM: @p7 = global ptr @.compoundliteral.7, align 8

int *p8 = &((int[]){10, 20, 30})[1];
// CIR: cir.global "private" internal @".compoundliteral.8" = #cir.const_array<[#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.int<30> : !s32i]> : !cir.array<!s32i x 3> {alignment = 4 : i64}
// CIR: cir.global external @p8 = #cir.global_view<@".compoundliteral.8", [1 : i32]> : !cir.ptr<!s32i> {alignment = 8 : i64}
// LLVM: @.compoundliteral.8 = internal global [3 x i32] [i32 10, i32 20, i32 30], align 4
// LLVM: @p8 = global ptr getelementptr {{(inbounds nuw )?}}(i8, ptr @.compoundliteral.8, i64 4), align 8

int x;
int **p9 = (int*[]){&x, &x};
// CIR: cir.global external @x
// CIR: cir.global "private" internal @".compoundliteral.9" = #cir.const_array<[#cir.global_view<@x> : !cir.ptr<!s32i>, #cir.global_view<@x> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2> {alignment = 8 : i64}
// CIR: cir.global external @p9 = #cir.global_view<@".compoundliteral.9"> : !cir.ptr<!cir.ptr<!s32i>> {alignment = 8 : i64}
// LLVM: @x = global i32 0, align 4
// LLVM: @.compoundliteral.9 = internal global [2 x ptr] [ptr @x, ptr @x], align 8
// LLVM: @p9 = global ptr @.compoundliteral.9, align 8
