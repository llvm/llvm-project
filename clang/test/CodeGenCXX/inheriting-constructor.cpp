// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// XFAIL: *

// PR12219
struct A { A(int); virtual ~A(); };
struct B : A { using A::A; ~B(); };
B::~B() {}
// CHECK: define void @_ZN1BD0Ev
// CHECK: define void @_ZN1BD1Ev
// CHECK: define void @_ZN1BD2Ev
