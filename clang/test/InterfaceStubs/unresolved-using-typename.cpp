// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !experimental-ifs-v1
// CHECK-NEXT: IfsVersion: 1.0
// CHECK-NEXT: Triple:
// CHECK-NEXT: ObjectFileFormat: ELF
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

// UnresolvedUsingTypenameDecl
template<typename T> class C1 { using ReprType = unsigned; };
template<typename T> class C2 : public C1<T> { using typename C1<T>::Repr; };
