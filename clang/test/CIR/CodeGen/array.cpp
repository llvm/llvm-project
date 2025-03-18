// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int a[10];
// CHECK: cir.global external @a : !cir.array<!cir.int<s, 32> x 10>

int aa[10][5];
// CHECK: cir.global external @aa : !cir.array<!cir.array<!cir.int<s, 32> x 5> x 10>

extern int b[10];
// CHECK: cir.global external @b : !cir.array<!cir.int<s, 32> x 10>

extern int bb[10][5];
// CHECK: cir.global external @bb : !cir.array<!cir.array<!cir.int<s, 32> x 5> x 10>

void f() {
  int l[10];
  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.int<s, 32> x 10>, !cir.ptr<!cir.array<!cir.int<s, 32> x 10>>, ["l"]
}

void f2(int p[10]) {}
// CHECK: cir.func @f2(%arg0: !cir.ptr<!cir.int<s, 32>>
// CHECK: cir.alloca !cir.ptr<!cir.int<s, 32>>, !cir.ptr<!cir.ptr<!cir.int<s, 32>>>, ["p", init]

void f3(int pp[10][5]) {}
// CHECK: cir.func @f3(%arg0: !cir.ptr<!cir.array<!cir.int<s, 32> x 5>>
// CHECK: cir.alloca !cir.ptr<!cir.array<!cir.int<s, 32> x 5>>, !cir.ptr<!cir.ptr<!cir.array<!cir.int<s, 32> x 5>>>
