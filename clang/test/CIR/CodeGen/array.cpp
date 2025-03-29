// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int a[10];
// CHECK: cir.global external @a : !cir.array<!s32i x 10>

int aa[10][5];
// CHECK: cir.global external @aa : !cir.array<!cir.array<!s32i x 5> x 10>

extern int b[10];
// CHECK: cir.global external @b : !cir.array<!s32i x 10>

extern int bb[10][5];
// CHECK: cir.global external @bb : !cir.array<!cir.array<!s32i x 5> x 10>

int c[10] = {};
// CHECK: cir.global external @c = #cir.zero : !cir.array<!s32i x 10>

int d[3] = {1, 2, 3};
// CHECK: cir.global external @d = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>

int dd[3][2] = {{1, 2}, {3, 4}, {5, 6}};
// CHECK: cir.global external @dd = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 3>

int e[10] = {1, 2};
// CHECK: cir.global external @e = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>

int f[5] = {1, 2};
// CHECK: cir.global external @f = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 5>

void func() {
  int l[10];
  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["l"]
}

void func2(int p[10]) {}
// CHECK: cir.func @func2(%arg0: !cir.ptr<!s32i>
// CHECK: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]

void func3(int pp[10][5]) {}
// CHECK: cir.func @func3(%arg0: !cir.ptr<!cir.array<!s32i x 5>>
// CHECK: cir.alloca !cir.ptr<!cir.array<!s32i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>
