// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK-O0 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK-O1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK-O2 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK-O3 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Os -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK-Os %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Oz -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CHECK-Oz %s

void foo() {}

// CHECK-O0: module
// CHECK-O0-NOT: cir.opt_info

// CHECK-O1: module
// CHECK-O1: cir.opt_info = #cir.opt_info<level = 1, size = 0>

// CHECK-O2: module
// CHECK-O2: cir.opt_info = #cir.opt_info<level = 2, size = 0>

// CHECK-O3: module
// CHECK-O3: cir.opt_info = #cir.opt_info<level = 3, size = 0>

// CHECK-Os: module
// CHECK-Os: cir.opt_info = #cir.opt_info<level = 2, size = 1>

// CHECK-Oz: module
// CHECK-Oz: cir.opt_info = #cir.opt_info<level = 2, size = 2>
