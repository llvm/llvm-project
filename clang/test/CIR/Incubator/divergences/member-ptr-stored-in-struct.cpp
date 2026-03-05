// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Member pointer stored as struct member.
//
// CodeGen still decomposes when passing/returning:
//   (check for decomposition in invoke())
//
// CIR passes as struct

// DIFF: Check for member pointer calling convention

struct Callback {
    int (Callback::*method)();
    int impl() { return 42; }
    int invoke() { return (this->*method)(); }
};

int test() {
    Callback cb;
    cb.method = &Callback::impl;
    return cb.invoke();
}
