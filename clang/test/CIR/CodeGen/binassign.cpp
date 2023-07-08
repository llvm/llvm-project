// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(int a, int b) {
  int x = a * b;
  x *= b;
  x /= b;
  x %= b;
  x += b;
  x -= b;
  x >>= b;
  x <<= b;
  x &= b;
  x ^= b;
  x |= b;
  return x;
}

// CHECK: [[Value:%[0-9]+]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK: = cir.binop(mul,
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(mul,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: cir.binop(div,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(rem,  {{.*}} loc([[SourceLocation:#loc[0-9]+]])
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(add,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(sub,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(shr,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(shl,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(and,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(xor,
// CHECK: cir.store {{.*}}[[Value]]
// CHECK: = cir.load {{.*}}[[Value]]
// CHECK: = cir.binop(or,
// CHECK: cir.store {{.*}}[[Value]]

typedef enum {
  A = 3,
} enumy;

enumy getty();

void exec() {
  enumy r;
  if ((r = getty()) < 0) {}
}

// CHECK: cir.func @_Z4execv()
// CHECK:   %0 = cir.alloca !u32i, cir.ptr <!u32i>, ["r"] {alignment = 4 : i64}
// CHECK:   cir.scope {
// CHECK:     %1 = cir.call @_Z5gettyv() : () -> !u32i
// CHECK:     cir.store %1, %0 : !u32i, cir.ptr <!u32i>
// CHECK:     %2 = cir.cast(integral, %1 : !u32i), !s32i
// CHECK:     %3 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:     %4 = cir.cmp(lt, %2, %3) : !s32i, !cir.bool
// CHECK:     cir.if %4 {

// CHECK: [[SourceLocationB:#loc[0-9]+]] = loc("{{.*}}binassign.cpp":8:8)
// CHECK: [[SourceLocationA:#loc[0-9]+]] = loc("{{.*}}binassign.cpp":8:3)
// CHECK: [[SourceLocation]] = loc(fused[[[SourceLocationA]], [[SourceLocationB]]])
