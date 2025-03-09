// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int a[10];
// CHECK: cir.global external @a : !cir.array<!cir.int<s, 32> x 10>

extern int b[10];
// CHECK: cir.global external @b : !cir.array<!cir.int<s, 32> x 10>

void f() {
  int c[10];
  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.int<s, 32> x 10>, !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, ["c"]
}
