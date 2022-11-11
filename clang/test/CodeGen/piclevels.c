// RUN: %clang_cc1 -emit-llvm -pic-level 2 %s -o - | FileCheck %s -check-prefix=CHECK-BIGPIC -check-prefix=CHECK-NOPIE
// RUN: %clang_cc1 -emit-llvm -pic-level 1 %s -o - | FileCheck %s -check-prefix=CHECK-SMALLPIC -check-prefix=CHECK-NOPIE
// RUN: %clang_cc1 -emit-llvm -pic-level 2 -pic-is-pie %s -o - | FileCheck %s -check-prefix=CHECK-BIGPIC -check-prefix=CHECK-BIGPIE
// RUN: %clang_cc1 -emit-llvm -pic-level 1 -pic-is-pie %s -o - | FileCheck %s -check-prefix=CHECK-SMALLPIC -check-prefix=CHECK-SMALLPIE

// CHECK-BIGPIC: !llvm.module.flags = !{{{.*}}}
// CHECK-BIGPIC: ![[#]] = !{i32 8, !"PIC Level", i32 2}
// CHECK-SMALLPIC: !llvm.module.flags = !{{{.*}}}
// CHECK-SMALLPIC: ![[#]] = !{i32 8, !"PIC Level", i32 1}
// CHECK-NOPIE-NOT: PIE Level
// CHECK-BIGPIE: ![[#]] = !{i32 7, !"PIE Level", i32 2}
// CHECK-SMALLPIE: ![[#]] = !{i32 7, !"PIE Level", i32 1}
