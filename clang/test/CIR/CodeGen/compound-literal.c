// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM


typedef struct {
  int *arr;
} S;

S a = {
  .arr = (int[]){}
};

// CIR: cir.global "private" internal @".compoundLiteral.0" = #cir.zero : !cir.array<!s32i x 0> {alignment = 4 : i64}
// CIR: cir.global external @a = #cir.const_struct<{#cir.global_view<@".compoundLiteral.0"> : !cir.ptr<!s32i>}> : !ty_22S22

// LLVM: @.compoundLiteral.0 = internal global [0 x i32] zeroinitializer
// LLVM: @a = global %struct.S { ptr @.compoundLiteral.0 }

S b = {
  .arr = (int[]){1}
};

// CIR: cir.global "private" internal @".compoundLiteral.1" = #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1> {alignment = 4 : i64}
// CIR: cir.global external @b = #cir.const_struct<{#cir.global_view<@".compoundLiteral.1"> : !cir.ptr<!s32i>}> : !ty_22S22

// LLVM: @.compoundLiteral.1 = internal global [1 x i32] [i32 1]
// LLVM: @b = global %struct.S { ptr @.compoundLiteral.1 }
