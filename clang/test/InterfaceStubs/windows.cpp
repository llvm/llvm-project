// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64-windows-msvc -o - %s \
// RUN: -emit-interface-stubs -emit-merged-ifs | FileCheck %s

// CHECK: Symbols:
// CHECK-NEXT: ?helloWindowsMsvc@@YAHXZ
int helloWindowsMsvc();
