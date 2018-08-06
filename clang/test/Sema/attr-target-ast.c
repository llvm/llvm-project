// RUN: %clang_cc1 -triple x86_64-linux-gnu -ast-dump %s | FileCheck %s

int __attribute__((target("arch=hiss,arch=woof"))) pine_tree() { return 4; }
// CHECK-NOT: arch=hiss
// CHECK-NOT: arch=woof
