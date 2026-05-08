// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Regression test for a CXXABILowering crash when a record that requires
// CXX-ABI lowering (because it contains a pointer-to-data-member field) also
// has a sibling char-array field initialized from a string literal.

struct B { int x; int y; };

struct S {
  int B::*pm;
  char buf[32];
};

const S g = { &B::y, "abc" };

const S *get() { return &g; }

// CIR: !rec_S = !cir.record<struct "S" {!s64i, !cir.array<!s8i x 32>}>
// CIR: cir.global {{.*}} @_ZL1g = #cir.const_record<{
// CIR-SAME: #cir.int<4> : !s64i,
// CIR-SAME: #cir.const_array<"abc" : !cir.array<!s8i x 3>, trailing_zeros> : !cir.array<!s8i x 32>
// CIR-SAME: }> : !rec_S

// LLVM: @_ZL1g = internal constant %struct.S { i64 4, [32 x i8] c"abc\00{{(\\00)+}}" }
