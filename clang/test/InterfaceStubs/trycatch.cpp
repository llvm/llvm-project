// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target: x86_64-unknown-linux-gnu
// CHECK-NEXT: Symbols:
// CHECK-NEXT: - { Name: "_Z1fv", Type: Func }
// CHECK-NEXT: ...

class C5 {};
void f() { try {} catch(C5&){} }
