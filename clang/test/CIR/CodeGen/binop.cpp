// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void b0(int a, int b) {
  int x = a * b;
  x = x / b;
  x = x % b;
  x = x + b;
  x = x - b;
  x = x >> b;
  x = x << b;
  x = x & b;
  x = x ^ b;
  x = x | b;
}

// CHECK: = cir.binop(mul, %3, %4) : i32
// CHECK: = cir.binop(div, %6, %7) : i32
// CHECK: = cir.binop(rem, %9, %10) : i32
// CHECK: = cir.binop(add, %12, %13) : i32
// CHECK: = cir.binop(sub, %15, %16) : i32
// CHECK: = cir.binop(shr, %18, %19) : i32
// CHECK: = cir.binop(shl, %21, %22) : i32
// CHECK: = cir.binop(and, %24, %25) : i32
// CHECK: = cir.binop(xor, %27, %28) : i32
// CHECK: = cir.binop(or, %30, %31) : i32