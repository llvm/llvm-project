// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-O0
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-O1
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-O2
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O3 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-O3
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Os -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-Os
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Oz -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CHECK-Oz

void f() {}

// CHECK-O0: module attributes
// CHECK-O0-NOT: cir.opt_info
// CHECK-O1: module attributes {{.+}}cir.opt_info = #cir.opt_info<level = 1, size = 0>{{.+}}
// CHECK-O2: module attributes {{.+}}cir.opt_info = #cir.opt_info<level = 2, size = 0>{{.+}}
// CHECK-O3: module attributes {{.+}}cir.opt_info = #cir.opt_info<level = 3, size = 0>{{.+}}
// CHECK-Os: module attributes {{.+}}cir.opt_info = #cir.opt_info<level = 2, size = 1>{{.+}}
// CHECK-Oz: module attributes {{.+}}cir.opt_info = #cir.opt_info<level = 2, size = 2>{{.+}}
