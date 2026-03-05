// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Variadic template missing comdat.
//
// CodeGen:
//   $_ZN7CounterIJidcEE5countE = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN7CounterIJidcEE5countE = comdat any

template<typename... Args>
struct Counter {
    static constexpr int count = sizeof...(Args);
};

int test() {
    return Counter<int, double, char>::count;
}
