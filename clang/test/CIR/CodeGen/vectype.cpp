// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

typedef int vi4 __attribute__((vector_size(16)));
typedef double vd2 __attribute__((vector_size(16)));
typedef long long vll2 __attribute__((vector_size(16)));

void vector_int_test(int x) {

  // Vector constant. Not yet implemented. Expected results will change from
  // cir.vec.create to cir.const.
  vi4 a = { 1, 2, 3, 4 };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>

  // Non-const vector initialization.
  vi4 b = { x, 5, 6, x + 1 };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>

  // Extract element
  int c = a[x];
  // CHECK: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : <!s32i x 4>

  // Insert element
  a[x] = x;
  // CHECK: %[[#LOADEDVI:]] = cir.load %[[#STORAGEVI:]] : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
  // CHECK: %[[#UPDATEDVI:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVI]][%{{[0-9]+}} : !s32i] : <!s32i x 4>
  // CHECK: cir.store %[[#UPDATEDVI]], %[[#STORAGEVI]] : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>

  // Binary arithmetic operations
  vi4 d = a + b;
  // CHECK: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 e = a - b;
  // CHECK: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 f = a * b;
  // CHECK: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 g = a / b;
  // CHECK: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 h = a % b;
  // CHECK: %{{[0-9]+}} = cir.binop(rem, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 i = a & b;
  // CHECK: %{{[0-9]+}} = cir.binop(and, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 j = a | b;
  // CHECK: %{{[0-9]+}} = cir.binop(or, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
  vi4 k = a ^ b;
  // CHECK: %{{[0-9]+}} = cir.binop(xor, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>

  // Unary arithmetic operations
  vi4 l = +a;
  // CHECK: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 m = -a;
  // CHECK: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  vi4 n = ~a;
  // CHECK: %{{[0-9]+}} = cir.unary(not, %{{[0-9]+}}) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>

  // Comparisons
  vi4 o = a == b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : <!s32i x 4>, <!s32i x 4>
  vi4 p = a != b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : <!s32i x 4>, <!s32i x 4>
  vi4 q = a < b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : <!s32i x 4>, <!s32i x 4>
  vi4 r = a > b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : <!s32i x 4>, <!s32i x 4>
  vi4 s = a <= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : <!s32i x 4>, <!s32i x 4>
  vi4 t = a >= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : <!s32i x 4>, <!s32i x 4>
}

void vector_double_test(int x, double y) {
  // Vector constant. Not yet implemented. Expected results will change from
  // cir.vec.create to cir.const.
  vd2 a = { 1.5, 2.5 };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : <!cir.double x 2>

  // Non-const vector initialization.
  vd2 b = { y, y + 1.0 };
  // CHECK: %{{[0-9]+}} = cir.vec.create(%{{[0-9]+}}, %{{[0-9]+}} : !cir.double, !cir.double) : <!cir.double x 2>

  // Extract element
  double c = a[x];
  // CHECK: %{{[0-9]+}} = cir.vec.extract %{{[0-9]+}}[%{{[0-9]+}} : !s32i] : <!cir.double x 2>

  // Insert element
  a[x] = y;
  // CHECK: %[[#LOADEDVF:]] = cir.load %[[#STORAGEVF:]] : cir.ptr <!cir.vector<!cir.double x 2>>, !cir.vector<!cir.double x 2>
  // CHECK: %[[#UPDATEDVF:]] = cir.vec.insert %{{[0-9]+}}, %[[#LOADEDVF]][%{{[0-9]+}} : !s32i] : <!cir.double x 2>
  // CHECK: cir.store %[[#UPDATEDVF]], %[[#STORAGEVF]] : !cir.vector<!cir.double x 2>, cir.ptr <!cir.vector<!cir.double x 2>>

  // Binary arithmetic operations
  vd2 d = a + b;
  // CHECK: %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 e = a - b;
  // CHECK: %{{[0-9]+}} = cir.binop(sub, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 f = a * b;
  // CHECK: %{{[0-9]+}} = cir.binop(mul, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>
  vd2 g = a / b;
  // CHECK: %{{[0-9]+}} = cir.binop(div, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>

  // Unary arithmetic operations
  vd2 l = +a;
  // CHECK: %{{[0-9]+}} = cir.unary(plus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>
  vd2 m = -a;
  // CHECK: %{{[0-9]+}} = cir.unary(minus, %{{[0-9]+}}) : !cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>

  // Comparisons
  vll2 o = a == b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(eq, %{{[0-9]+}}, %{{[0-9]+}}) : <!cir.double x 2>, <!s64i x 2>
  vll2 p = a != b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ne, %{{[0-9]+}}, %{{[0-9]+}}) : <!cir.double x 2>, <!s64i x 2>
  vll2 q = a < b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(lt, %{{[0-9]+}}, %{{[0-9]+}}) : <!cir.double x 2>, <!s64i x 2>
  vll2 r = a > b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(gt, %{{[0-9]+}}, %{{[0-9]+}}) : <!cir.double x 2>, <!s64i x 2>
  vll2 s = a <= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(le, %{{[0-9]+}}, %{{[0-9]+}}) : <!cir.double x 2>, <!s64i x 2>
  vll2 t = a >= b;
  // CHECK: %{{[0-9]+}} = cir.vec.cmp(ge, %{{[0-9]+}}, %{{[0-9]+}}) : <!cir.double x 2>, <!s64i x 2>
}
