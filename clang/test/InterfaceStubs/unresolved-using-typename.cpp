// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target:
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

// UnresolvedUsingTypenameDecl
template<typename T> class C1 { using ReprType = unsigned; };
template<typename T> class C2 : public C1<T> { using typename C1<T>::Repr; };
