// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Lambda with capture by reference missing comdat.
//
// CodeGen:
//   $_ZZ4testvENK3$_0clEv = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZZ4testvENK3$_0clEv = comdat any

int test() {
    int x = 10;
    auto f = [&x]() { x *= 2; return x; };
    return f();
}
