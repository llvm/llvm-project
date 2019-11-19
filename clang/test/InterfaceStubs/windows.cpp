// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-windows-msvc -o - %s -emit-interface-stubs | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: %clang -target x86_64-windows-msvc -o - %s -emit-interface-stubs -emit-merged-ifs | FileCheck -check-prefix=CHECK-IFS %s

// CHECK-CC1: Symbols:
// CHECK-CC1-NEXT: ?helloWindowsMsvc@@YAHXZ

 // CHECK-IFS: --- !experimental-ifs-v1
 // CHECK-IFS: IfsVersion:      1.0
 // CHECK-IFS: Triple:
 // CHECK-IFS: Symbols:
 // CHECK-IFS:   ?helloWindowsMsvc@@YAHXZ: { Type: Func }
 // CHECK-IFS: ...

int helloWindowsMsvc();
