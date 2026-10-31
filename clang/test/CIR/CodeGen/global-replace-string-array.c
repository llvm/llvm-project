// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

extern int arr[];
struct S {
  int *p;
  char name[4];
};
struct S a = {arr, "aa"};
int arr[5] = {1, 2, 3, 4, 5};

// CIR-DAG: cir.global external @arr = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i]> : !cir.array<!s32i x 5>
// CIR-DAG: cir.global external @a = #cir.const_record<{#cir.global_view<@arr> : !cir.ptr<!s32i>, #cir.const_array<"aa" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 4>}>

// LLVM-DAG: @arr = global [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5]
// LLVM-CIR-DAG: @a = global %struct.S { ptr @arr, [4 x i8] c"aa\00\00" }
// OGCG-DAG: @a = global { ptr, [4 x i8], [4 x i8] } { ptr @arr, [4 x i8] c"aa\00\00", [4 x i8] zeroinitializer }
