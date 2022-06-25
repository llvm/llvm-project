// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

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

// CHECK: [[Value:%[0-9]+]] = cir.alloca i32, cir.ptr <i32>, ["x", cinit] {alignment = 4 : i64}
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

// CHECK: [[SourceLocation]] = loc(fused["{{.*}}binassign.cpp":8:3, "{{.*}}binassign.cpp":8:8])
