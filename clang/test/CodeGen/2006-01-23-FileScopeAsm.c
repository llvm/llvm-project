// UNSUPPORTED: target={{.*}}-zos{{.*}}
// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: module asm(target_features: "{{.*}}")
// CHECK-NEXT: "foo1"
__asm__ ("foo1");
// CHECK-NEXT: "foo2"
__asm__ ("foo2");
// CHECK-NEXT: "foo3"
__asm__ ("foo3");
// CHECK-NEXT: "foo4"
__asm__ ("foo4");
// CHECK-NEXT: "foo5"
__asm__ ("foo5");
