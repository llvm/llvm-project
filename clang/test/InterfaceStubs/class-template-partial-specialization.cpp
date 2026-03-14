// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target:
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

template<typename> struct S7 { };
template<typename T> struct S7<T&> { };
