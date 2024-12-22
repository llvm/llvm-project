// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

// Verify that we print the [[clang::lifetime_capture_by(X)]] attribute.

struct S {
    void foo(int &a, int &b) [[clang::lifetime_capture_by(a, b, global)]];
};

// CHECK: CXXMethodDecl {{.*}}clang::lifetime_capture_by(a, b, global)
