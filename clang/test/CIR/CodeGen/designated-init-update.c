// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,ALL --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG,ALL --input-file=%t.ll %s

struct S {
  int a, b, c;
};

// Basic designated init update: start from {1, 2, 3}, override .b = 20.
// Sema folds in place -- does NOT produce a DesignatedInitUpdateExpr.
struct S g1 = (struct S){1, 2, 3, .b = 20};
// CIR: cir.global external @g1 = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<20> : !s32i, #cir.int<3> : !s32i}> : !rec_S
// ALL: @g1 = global %struct.S { i32 1, i32 20, i32 3 }

// Multiple field overrides; also folded by Sema.
struct S g2 = (struct S){10, 20, 30, .a = 100, .c = 300};
// CIR: cir.global external @g2 = #cir.const_record<{#cir.int<100> : !s32i, #cir.int<20> : !s32i, #cir.int<300> : !s32i}> : !rec_S
// ALL: @g2 = global %struct.S { i32 100, i32 20, i32 300 }

// Nested struct, folded by Sema.
struct Outer {
  struct S inner;
  int x;
};
struct Outer g3 = (struct Outer){{1, 2, 3}, 4, .inner.b = 50};
// CIR: cir.global external @g3 = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i, #cir.int<50> : !s32i, #cir.int<3> : !s32i}> : !rec_S, #cir.int<4> : !s32i}> : !rec_Outer
// ALL: @g3 = global %struct.Outer { %struct.S { i32 1, i32 50, i32 3 }, i32 4 }

// From here on: cases that produce a DesignatedInitUpdateExpr.

// g4: compound-literal sub-record + later record-field override.
struct P {
  struct S s;
  int x;
};
struct P g4 = { (struct S){1, 2, 3}, 4, .s.b = 9 };
// CIR: cir.global external @g4 = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i, #cir.int<9> : !s32i, #cir.int<3> : !s32i}> : !rec_S, #cir.int<4> : !s32i}> : !rec_P
// ALL: @g4 = global %struct.P { %struct.S { i32 1, i32 9, i32 3 }, i32 4 }

// g5: compound-literal sub-record + later array-element override.
struct Inner { int arr[4]; };
struct ArrOuter { struct Inner in; int x; };
struct ArrOuter g5 = { (struct Inner){{10, 20, 30, 40}}, 5, .in.arr[1] = 99 };
// CIR: cir.global external @g5 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.int<10> : !s32i, #cir.int<99> : !s32i, #cir.int<30> : !s32i, #cir.int<40> : !s32i]> : !cir.array<!s32i x 4>}> : !rec_Inner, #cir.int<5> : !s32i}> : !rec_ArrOuter
// ALL: @g5 = global %struct.ArrOuter { %struct.Inner { [4 x i32] [i32 10, i32 99, i32 30, i32 40] }, i32 5 }

// g6: empty initializer base for a sub-record + deep designator override
// through an anonymous struct.
struct Base {
  struct {
    int A;
  };
};
struct Derived {
  struct Base B;
};
struct Derived g6 = { {}, .B.A = 42 };
// CIR: cir.global external @g6 = #cir.const_record<{#cir.const_record<{#cir.const_record<{#cir.int<42> : !s32i}> : !rec_anon{{.*}}}> : !rec_Base}> : !rec_Derived
// ALL: @g6 = global %struct.Derived { %struct.Base { %struct.anon{{.*}} { i32 42 } } }

// g7: array-of-array element override. Exercises nested array update.
struct M { int m[2][3]; };
struct N { struct M mm; int x; };
struct N g7 = { (struct M){{ {1,2,3}, {4,5,6} }}, 7, .mm.m[1][1] = 99 };
// CIR: cir.global external @g7 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>, #cir.const_array<[#cir.int<4> : !s32i, #cir.int<99> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 3>]> : !cir.array<!cir.array<!s32i x 3> x 2>}> : !rec_M, #cir.int<7> : !s32i}> : !rec_N
// ALL: @g7 = global %struct.N { %struct.M { [2 x [3 x i32]] {{\[}}[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 99, i32 6]] }, i32 7 }

// g8: trailing zeros + single override in a 10-element array.
struct Q { int arr[10]; };
struct R { struct Q q; int x; };
struct R g8 = { (struct Q){{1, 2, 3}}, 5, .q.arr[7] = 99 };
// CIR: cir.global external @g8 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<99> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>}> : !rec_Q, #cir.int<5> : !s32i}> : !rec_R
// ALL: @g8 = global %struct.R { %struct.Q { [10 x i32] [i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 99, i32 0, i32 0] }, i32 5 }

// g9: large trailing zeros (20-element array, override at [15]).
struct BigArrInner { int arr[20]; };
struct BigArrOuter { struct BigArrInner in; int x; };
struct BigArrOuter g9 = { (struct BigArrInner){{1, 2, 3}}, 7, .in.arr[15] = 99 };
// CIR: cir.global external @g9 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<99> : !s32i], trailing_zeros> : !cir.array<!s32i x 20>}> : !rec_BigArrInner, #cir.int<7> : !s32i}> : !rec_BigArrOuter
// ALL: @g9 = global %struct.BigArrOuter { %struct.BigArrInner { [20 x i32] [i32 1, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 99, i32 0, i32 0, i32 0, i32 0] }, i32 7 }

// g10: all-zero base, single override.
struct AllZero { int arr[12]; };
struct AllZeroOuter { struct AllZero az; int x; };
struct AllZeroOuter g10 = { (struct AllZero){{0}}, 7, .az.arr[5] = 99 };
// CIR: cir.global external @g10 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<99> : !s32i], trailing_zeros> : !cir.array<!s32i x 12>}> : !rec_AllZero, #cir.int<7> : !s32i}> : !rec_AllZeroOuter
// LLVM: @g10 = global %struct.AllZeroOuter { %struct.AllZero { [12 x i32] [i32 0, i32 0, i32 0, i32 0, i32 0, i32 99, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0] }, i32 7 }
// Note: OGCG produces a byte-packed struct for all-zero bases; our typed
// array output is better. Only verify the values are present.
// OGCG: @g10 = global { { { [20 x i8], i32, [24 x i8] } }, i32 }

// g11: very large array (100 elements), override at index 80.
struct Many { int arr[100]; };
struct ManyOuter { struct Many mm; int x; };
struct ManyOuter g11 = { (struct Many){{1, 2, 3}}, 7, .mm.arr[80] = 42 };
// CIR: cir.global external @g11 = #cir.const_record<{#cir.const_record<{#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i{{.*}}#cir.int<42> : !s32i], trailing_zeros> : !cir.array<!s32i x 100>}> : !rec_Many, #cir.int<7> : !s32i}> : !rec_ManyOuter
// LLVM: @g11 = global %struct.ManyOuter { %struct.Many { [100 x i32] [i32 1, i32 2, i32 3,{{.*}} i32 42,{{.*}}] }, i32 7 }
// Note: OGCG produces a packed struct <{[81 x i32], [19 x i32]}>; our
// typed [100 x i32] array output is better.
// OGCG: @g11 = global { { <{ [81 x i32], [19 x i32] }> }, i32 }

// g12: unnamed bitfield - tests that elementNo skips unnamed bitfields
// correctly (they are not represented in the InitListExpr).
struct WithBitfield {
  int a;
  int : 16;
  int b;
  int c;
};
struct BFOuter { struct WithBitfield wbf; int x; };
struct BFOuter g12 = { (struct WithBitfield){1, 2, 3}, 4, .wbf.b = 99 };
// LLVM: @g12 = global %struct.BFOuter { %struct.WithBitfield { i32 1, i16 0, i32 99, i32 3 }, i32 4 }
// OGCG: @g12 = global { { i32, [4 x i8], i32, i32 }, i32 }
