// RUN: %clang_cc1 -o - -emit-interface-stubs -std=c++17 %s | FileCheck %s

// CHECK:      --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target:
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

// CXXDeductionGuideDecl
template<typename T> struct A { A(); A(T); };
A() -> A<int>;
