// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Lambda functions are missing comdat groups and other attributes.
//
// CodeGen generates comdat declarations for lambda operator():
//   $_ZZ4testvENK3$_0clEv = comdat any
//   define linkonce_odr i32 @_ZZ4testvENK3$_0clEv(...) comdat
//
// CIR omits comdat:
//   define linkonce_odr i32 @_ZZ4testvENK3$_0clEv(...)  // No comdat
//
// This affects:
// - Lambda operator() functions
// - Lambda copy/move constructors
// - Lambda destructors
//
// Impact: Potential ODR violations with multiple TUs

// DIFF: -$_ZZ4testvENK3$_0clEv = comdat any
// DIFF: -define linkonce_odr {{.*}} @_ZZ4testvENK3$_0clEv{{.*}} comdat
// DIFF: +define linkonce_odr {{.*}} @_ZZ4testvENK3$_0clEv

int test() {
    auto f = []() { return 42; };
    return f();
}

// Lambda with capture
int test_capture() {
    int x = 10;
    auto f = [x]() { return x * 2; };
    return f();
}

// Lambda taking parameters
int test_params() {
    auto f = [](int a, int b) { return a + b; };
    return f(10, 20);
}

// Mutable lambda
int test_mutable() {
    int x = 10;
    auto f = [x]() mutable { return ++x; };
    return f();
}

// Lambda returning struct
struct Result {
    int value;
};

int test_struct_return() {
    auto f = []() { return Result{42}; };
    return f().value;
}
