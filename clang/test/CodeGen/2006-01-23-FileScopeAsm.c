// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: module asm "foo1"
__asm__ ("foo1");
// CHECK: module asm "foo2"
__asm__ ("foo2");
// CHECK: module asm "foo3"
__asm__ ("foo3");
// CHECK: module asm "foo4"
__asm__ ("foo4");
// CHECK: module asm "foo5"
__asm__ ("foo5");
