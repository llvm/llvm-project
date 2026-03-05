// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR crashes when using array new with default constructor arguments.
//
// Array new requires calling the constructor for each element. When the
// constructor has default arguments, CIR's lowering fails with:
//   'cir.const' op operation destroyed but still has uses
//   fatal error: error in backend: operation destroyed but still has uses
//
// The issue is in how CIR handles default argument values when generating
// the loop to initialize array elements.
//
// This affects any array new expression where the class has a constructor
// with default parameters.

struct S {
    int x;
    S(int v = 0) : x(v) {}  // Default argument triggers the bug
    ~S() {}
};

S* test_array_new() {
    return new S[10];  // Crashes during lowering
}

// LLVM: Should generate array new
// LLVM: define {{.*}} @_Z14test_array_newv()

// OGCG: Should generate array new with cookie and element loop
// OGCG: define {{.*}} @_Z14test_array_newv()
// OGCG: call {{.*}} @_Znam  // operator new[]
