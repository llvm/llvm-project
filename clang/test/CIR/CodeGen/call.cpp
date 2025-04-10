// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

void f1();
void f2() {
  f1();
}

// CHECK-LABEL: cir.func @f2
// CHECK:         cir.call @f1() : () -> ()
