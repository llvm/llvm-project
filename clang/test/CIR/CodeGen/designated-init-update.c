// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  int a, b, c;
};

// Basic designated init update: start from {1, 2, 3}, override .b = 20
struct S g1 = (struct S){1, 2, 3, .b = 20};

// CIR: cir.global external @g1 = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<20> : !s32i, #cir.int<3> : !s32i}> : !rec_S
// LLVM: @g1 = global %struct.S { i32 1, i32 20, i32 3 }
// OGCG: @g1 = global %struct.S { i32 1, i32 20, i32 3 }

// Multiple field overrides
struct S g2 = (struct S){10, 20, 30, .a = 100, .c = 300};

// CIR: cir.global external @g2 = #cir.const_record<{#cir.int<100> : !s32i, #cir.int<20> : !s32i, #cir.int<300> : !s32i}> : !rec_S
// LLVM: @g2 = global %struct.S { i32 100, i32 20, i32 300 }
// OGCG: @g2 = global %struct.S { i32 100, i32 20, i32 300 }

// Nested struct with designated init update
struct Outer {
  struct S inner;
  int x;
};

struct Outer g3 = (struct Outer){{1, 2, 3}, 4, .inner.b = 50};

// CIR: cir.global external @g3 = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i, #cir.int<50> : !s32i, #cir.int<3> : !s32i}> : !rec_S, #cir.int<4> : !s32i}> : !rec_Outer
// LLVM: @g3 = global %struct.Outer { %struct.S { i32 1, i32 50, i32 3 }, i32 4 }
// OGCG: @g3 = global %struct.Outer { %struct.S { i32 1, i32 50, i32 3 }, i32 4 }

// Compound literal as sub-object with later field override.
// This produces a DesignatedInitUpdateExpr AST node (unlike the cases above
// which Sema folds in-place) and exercises ConstantAggregateBuilder::split().
struct P {
  struct S s;
  int x;
};

struct P g4 = { (struct S){1, 2, 3}, 4, .s.b = 9 };

// CIR: cir.global external @g4 = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<3> : !s32i}> : !rec_S, #cir.int<4> : !s32i}> : !rec_P
// LLVM: @g4 = global %struct.P { %struct.S { i32 1, i32 9, i32 3 }, i32 4 }
// OGCG: @g4 = global %struct.P { %struct.S { i32 1, i32 9, i32 3 }, i32 4 }

// Array element override via compound literal — exercises the array path
// of emitDesignatedInitUpdater and buildFrom for cir::ArrayType.
struct Inner { int arr[4]; };
struct ArrOuter { struct Inner in; int x; };

struct ArrOuter g5 = { (struct Inner){{10, 20, 30, 40}}, 5, .in.arr[1] = 99 };

// CIR: cir.global external @g5 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.int<10> : !s32i, #cir.int<99> : !s32i, #cir.int<30> : !s32i, #cir.int<40> : !s32i]> : !cir.array<!s32i x 4>}> : !rec_Inner, #cir.int<5> : !s32i}> : !rec_ArrOuter
// LLVM: @g5 = global %struct.ArrOuter { %struct.Inner { [4 x i32] [i32 10, i32 99, i32 30, i32 40] }, i32 5 }
// OGCG: @g5 = global %struct.ArrOuter { %struct.Inner { [4 x i32] [i32 10, i32 99, i32 30, i32 40] }, i32 5 }
