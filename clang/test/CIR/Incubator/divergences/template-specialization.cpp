// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Template specialization missing comdat.
//
// CodeGen generates comdat for specialization:
//   $_ZN5ValueIiE3getEv = comdat any
//
// CIR missing comdat

// DIFF: -$_ZN5ValueIiE3getEv = comdat any

template<typename T>
struct Value {
    static int get() { return 1; }
};

template<>
struct Value<int> {
    static int get() { return 42; }
};

int test() {
    return Value<int>::get();
}
