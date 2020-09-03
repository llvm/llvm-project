// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-unknown-linux -std=c++17 | FileCheck %s

double a(double) noexcept;
int b(double (&)(double));

// CHECK: call i32 @_Z1bRFddE(double (double)* nonnull @_Z1ad)
int c = b(a);
