// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -O1 -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void b0(int a, int b) {
  int x = a * b;
  x = x / b;
  x = x % b;
  x = x + b;
  x = x - b;
  x = x & b;
  x = x ^ b;
  x = x | b;
}

// CHECK: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) nsw : !s32i
// CHECK: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !s32i
// CHECK: %{{.+}} = cir.binop(rem, %{{.+}}, %{{.+}}) : !s32i
// CHECK: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) nsw : !s32i
// CHECK: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) nsw : !s32i
// CHECK: %{{.+}} = cir.binop(and, %{{.+}}, %{{.+}}) : !s32i
// CHECK: %{{.+}} = cir.binop(xor, %{{.+}}, %{{.+}}) : !s32i
// CHECK: %{{.+}} = cir.binop(or, %{{.+}}, %{{.+}}) : !s32i

void testFloatingPointBinOps(float a, float b) {
  a * b;
  // CHECK: cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.float
  a / b;
  // CHECK: cir.binop(div, %{{.+}}, %{{.+}}) : !cir.float
  a + b;
  // CHECK: cir.binop(add, %{{.+}}, %{{.+}}) : !cir.float
  a - b;
  // CHECK: cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.float
}
