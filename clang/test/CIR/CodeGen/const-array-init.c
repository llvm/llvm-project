// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Fully initialized array.
int arr1[5] = {1, 2, 3, 4, 5};
// CIR: cir.global external @arr1 = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i]> : !cir.array<!s32i x 5>
// LLVM: @arr1 = global [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5]
// OGCG: @arr1 = global [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5]

// Partial initialization with implicit zero fill (< 8 trailing zeros).
int arr2[7] = {1, 2, 3};
// CIR: cir.global external @arr2 = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 7>
// LLVM: @arr2 = global [7 x i32] [i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0]
// OGCG: @arr2 = global [7 x i32] [i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0]

// All-zero array from {0}.
int arr3[4] = {0};
// CIR: cir.global external @arr3 = #cir.zero : !cir.array<!s32i x 4>
// LLVM: @arr3 = global [4 x i32] zeroinitializer
// OGCG: @arr3 = global [4 x i32] zeroinitializer

// Nested 2D array.
int arr4[2][3] = {{1, 2, 3}, {4, 5, 6}};
// CIR: cir.global external @arr4 = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>, #cir.const_array<[#cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 3>]> : !cir.array<!cir.array<!s32i x 3> x 2>
// LLVM: @arr4 = global [2 x [3 x i32]] {{\[}}[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6]]
// OGCG: @arr4 = global [2 x [3 x i32]] {{\[}}[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6]]

// Float array.
float arr5[3] = {1.0f, 2.0f, 3.0f};
// CIR: cir.global external @arr5 = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.float, #cir.fp<2.000000e+00> : !cir.float, #cir.fp<3.000000e+00> : !cir.float]> : !cir.array<!cir.float x 3>
// LLVM: @arr5 = global [3 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00]
// OGCG: @arr5 = global [3 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00]

// Large trailing zeros (>= 8) triggers struct packing in emitArrayConstant.
int arr6[20] = {1, 2, 3};
// CIR: cir.global external @arr6 = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.zero : !cir.array<!s32i x 17>}>
// LLVM: @arr6 = global <{ i32, i32, i32, [17 x i32] }> <{ i32 1, i32 2, i32 3, [17 x i32] zeroinitializer }>
// OGCG: @arr6 = global <{ i32, i32, i32, [17 x i32] }> <{ i32 1, i32 2, i32 3, [17 x i32] zeroinitializer }>

// Char array from string literal.
char str[] = "hello";
// CIR: cir.global external @str = #cir.const_array<"hello" : !cir.array<!s8i x 5>, trailing_zeros> : !cir.array<!s8i x 6>
// LLVM: @str = global [6 x i8] c"hello\00"
// OGCG: @str = global [6 x i8] c"hello\00"
