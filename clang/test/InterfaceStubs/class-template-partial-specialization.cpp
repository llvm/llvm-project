// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !experimental-ifs-v2
// CHECK-NEXT: IfsVersion: 2.0
// CHECK-NEXT: Triple:
// CHECK-NEXT: ObjectFileFormat: ELF
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

template<typename> struct S7 { };
template<typename T> struct S7<T&> { };
