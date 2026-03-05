// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

// Test that [[no_unique_address]] empty fields are handled correctly.
// These fields are zero-sized and don't occupy space in the struct layout,
// but we still need to be able to initialize them in constructors.
// Trivial default constructors for empty fields are lowered away.

struct Empty {};

struct S {
  int x;
  [[no_unique_address]] Empty e;
  S() : x(1), e() {}
};

void test() {
  S s;
}

// The struct should only have space for 'x' (the empty field is zero-sized)
// CIR-DAG: !rec_S = !cir.record<struct "S" {!s32i}>

// CIR: cir.func {{.*}}linkonce_odr @_ZN1SC2Ev
// CIR:   cir.store {{.*}} : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return

// Trivial default constructor call is lowered away, matching OG behavior
// LLVM-LABEL: define {{.*}} @_ZN1SC2Ev
// LLVM:   store i32 1
// LLVM-NOT:   call void @_ZN5EmptyC1Ev
// LLVM:   ret void

// OGCG-LABEL: define {{.*}} @_ZN1SC2Ev
// OGCG:   store i32 1
// OGCG:   ret void
