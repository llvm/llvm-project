// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void c0(int a, int b) {
  bool x = a > b;
  x = a < b;
  x = a <= b;
  x = a >= b;
  x = a != b;
  x = a == b;
}

// CHECK: = cir.cmp(gt, %3, %4) : i32, !cir.bool
// CHECK: = cir.cmp(lt, %6, %7) : i32, !cir.bool
// CHECK: = cir.cmp(le, %9, %10) : i32, !cir.bool
// CHECK: = cir.cmp(ge, %12, %13) : i32, !cir.bool
// CHECK: = cir.cmp(ne, %15, %16) : i32, !cir.bool
// CHECK: = cir.cmp(eq, %18, %19) : i32, !cir.bool
