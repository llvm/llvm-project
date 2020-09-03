// RUN: %clang_cc1 -o - -emit-interface-stubs -std=c++17 %s | FileCheck %s

// CHECK:      --- !experimental-ifs-v2
// CHECK-NEXT: IfsVersion: 2.0
// CHECK-NEXT: Triple:
// CHECK-NEXT: ObjectFileFormat: ELF
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

// CXXDeductionGuideDecl
template<typename T> struct A { A(); A(T); };
A() -> A<int>;
