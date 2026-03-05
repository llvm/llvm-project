// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Member function template missing comdat.
//
// CodeGen:
//   $_ZN1S7processIiEEiT_ = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN1S7processIiEEiT_ = comdat any

struct S {
    template<typename T>
    int process(T value) {
        return static_cast<int>(value);
    }
};

int test() {
    S s;
    return s.process(42.5);
}
