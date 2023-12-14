// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

typedef int int4 __attribute__((vector_size(16)));
int test_vector_basic(int x, int y, int z) {
  int4 a = { 1, 2, 3, 4 };
  int4 b = { x, y, z, x + y + z };
  int4 c = a + b;
  return c[1];
}

// CHECK:    %4 = cir.alloca !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>, ["a", init] {alignment = 16 : i64}
// CHECK:    %5 = cir.alloca !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>, ["b", init] {alignment = 16 : i64}
// CHECK:    %6 = cir.alloca !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>, ["c", init] {alignment = 16 : i64}

// CHECK:    %7 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:    %8 = cir.const(#cir.int<2> : !s32i) : !s32i
// CHECK:    %9 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:    %10 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK:    %11 = cir.vec.create(%7, %8, %9, %10 : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>
// CHECK:    cir.store %11, %4 : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %12 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:    %13 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK:    %14 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK:    %15 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:    %16 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK:    %17 = cir.binop(add, %15, %16) : !s32i
// CHECK:    %18 = cir.load %2 : cir.ptr <!s32i>, !s32i
// CHECK:    %19 = cir.binop(add, %17, %18) : !s32i
// CHECK:    %20 = cir.vec.create(%12, %13, %14, %19 : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>
// CHECK:    cir.store %20, %5 : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %21 = cir.load %4 : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %22 = cir.load %5 : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %23 = cir.binop(add, %21, %22) : !cir.vector<!s32i x 4>
// CHECK:    cir.store %23, %6 : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %24 = cir.load %6 : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %25 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:    %26 = cir.vec.extract %24[%25 : !s32i] <!s32i x 4> -> !s32i
// CHECK:    cir.store %26, %3 : !s32i, cir.ptr <!s32i>
// CHECK:    %27 = cir.load %3 : cir.ptr <!s32i>, !s32i
// CHECK:    cir.return %27 : !s32i
