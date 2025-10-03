// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

struct Other {
    int x;
};

struct Trivial {
    int x;
    double y;
    decltype(&Other::x) ptr;
};

// This case has a trivial default constructor, but can't be zero-initialized.
Trivial t;

// Since the case above isn't handled yet, we want a test that verifies that
// we're failing for the right reason.

// CHECK: error: ClangIR code gen Not Yet Implemented: tryEmitPrivateForVarInit: non-zero-initializable cxx record
