// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-windows-msvc -o - %s -emit-interface-stubs | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: %clang -target x86_64-windows-msvc -o - %s -emit-interface-stubs -emit-merged-ifs -S | FileCheck -check-prefix=CHECK-IFS %s
// note: -S is added here to prevent clang from invoking link.exe

// CHECK-CC1: Symbols:
// CHECK-CC1-NEXT: ?helloWindowsMsvc@@YAHXZ

// CHECK-IFS: --- !experimental-ifs-v2
// CHECK-IFS: IfsVersion: 2.0
// CHECK-IFS: Triple:
// CHECK-IFS: Symbols:
// CHECK-IFS:   - { Name: '?helloWindowsMsvc@@YAHXZ', Type: Func }
// CHECK-IFS: ...

int helloWindowsMsvc();
