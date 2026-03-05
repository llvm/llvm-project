// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Move constructor missing comdat.
//
// CodeGen:
//   $_ZN1SC1EOS_ = comdat any
//   $_ZN1SD1Ev = comdat any
//
// CIR:
//   Both missing comdat

// DIFF: -$_ZN1SC1EOS_ = comdat any
// DIFF: -$_ZN1SD1Ev = comdat any

struct S {
    int* ptr;
    S(int val) : ptr(new int(val)) {}
    S(S&& other) : ptr(other.ptr) { other.ptr = nullptr; }
    ~S() { delete ptr; }
    int get() const { return ptr ? *ptr : 0; }
};

int test() {
    S s1(42);
    S s2(static_cast<S&&>(s1));
    return s2.get();
}
