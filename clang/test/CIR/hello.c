// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s | FileCheck --allow-empty %s

// just confirm that we don't crash
// CHECK-NOT: *
void foo() {}
