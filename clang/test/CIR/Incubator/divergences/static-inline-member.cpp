// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Inline static member (C++17).
//
// CodeGen:
//   @_ZN1S5valueE = linkonce_odr global i32 42, comdat
//
// CIR:
//   Missing comdat

// DIFF: -@_ZN1S5valueE = linkonce_odr global i32 42, comdat
// DIFF: +@_ZN1S5valueE = linkonce_odr global i32 42

struct S {
    inline static int value = 42;
};

int test() {
    return S::value;
}
