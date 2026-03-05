// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Lambda returning struct missing comdat and may have struct return ABI issues.
//
// CodeGen:
//   $_ZZ4testvENK3$_0clEv = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZZ4testvENK3$_0clEv = comdat any

struct Result {
    int value;
};

int test() {
    auto f = []() { return Result{42}; };
    return f().value;
}
