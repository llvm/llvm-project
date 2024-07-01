// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o - %s | FileCheck --check-prefix=NONATIVE %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fnative-half-type -fclangir -emit-cir -o - %s | FileCheck --check-prefix=NATIVE %s

volatile unsigned test;
volatile int i0;
volatile _Float16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;
short s0;

void foo(void) {
  test = (h0);
  // NONATIVE: %{{.+}} = cir.cast(float_to_int, %{{.+}} : !cir.f16), !u32i
  // NATIVE: %{{.+}} = cir.cast(float_to_int, %{{.+}} : !cir.f16), !u32i

  h0 = (test);
  // NONATIVE: %{{.+}} = cir.cast(int_to_float, %{{.+}} : !u32i), !cir.f16
  // NATIVE: %{{.+}} = cir.cast(int_to_float, %{{.+}} : !u32i), !cir.f16

  test = (!h1);
  //      NONATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(bool_to_int, %[[#B]] : !cir.bool), !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NATIVE-NEXT: %[[#B:]] = cir.unary(not, %[[#A]]) : !cir.bool, !cir.bool
  // NATIVE-NEXT: %[[#C:]] = cir.cast(bool_to_int, %[[#B]] : !cir.bool), !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  h1 = -h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.float), !cir.f16
  //      NATIVE: %{{.+}} = cir.unary(minus, %{{.+}}) : !cir.f16, !cir.f16

  h1 = +h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(plus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //  NATIVE-NOT: %{{.+}} = cir.cast(floating, %{{.+}} : !cir.float), !cir.f16
  //      NATIVE: %{{.+}} = cir.unary(plus, %{{.+}}) : !cir.f16, !cir.f16

  h1++;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  ++h1;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  --h1;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  h1--;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NONATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<-1.000000e+00> : !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  h1 = h0 * h2;
  //      NONATIVE: %[[#LHS:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#RHS:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#A:]] = cir.binop(mul, %[[#LHS]], %[[#RHS]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#A]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.f16

  h1 = h0 * (_Float16) -2.0f;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.unary(minus, %[[#B]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(mul, %[[#A]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#C]]) : !cir.f16

  h1 = h0 * f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = f0 * h2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = h0 * i0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.f16

  h1 = (h0 / h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.f16

  h1 = (h0 / (_Float16) -2.0f);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.unary(minus, %[[#B]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(div, %[[#A]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#C]]) : !cir.f16

  h1 = (h0 / f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = (f0 / h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = (h0 / i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(div, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.f16

  h1 = (h2 + h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.f16

  h1 = ((_Float16)-2.0 + h0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#E:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(add, %[[#D]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  //      NATIVE: %{{.+}} = cir.binop(add, %[[#C]], %{{.+}}) : !cir.f16

  h1 = (h2 + f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = (f2 + h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = (h0 + i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(add, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16

  h1 = (h2 - h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#A]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16

  // NATIVE: %{{.+}} = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.f16

  h1 = ((_Float16)-2.0f - h0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#E:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.binop(sub, %[[#D]], %[[#E]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#F]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  //      NATIVE: %{{.+}} = cir.binop(sub, %[[#C]], %{{.+}}) : !cir.f16

  h1 = (h2 - f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = (f2 - h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.float
  // NATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16

  h1 = (h0 - i0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#A]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %{{.+}} = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %{{.+}} = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.f16

  test = (h2 < h0);
  //      NONATIVE: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(lt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  test = (h2 < (_Float16)42.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(lt, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  test = (h2 < f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (f2 < h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (i0 < h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(lt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 < i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(lt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 > h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(gt, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  test = ((_Float16)42.0 > h2);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NONATIVE: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.cmp(gt, %[[#B]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  test = (h0 > f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (f0 > h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (i0 > h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(gt, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 > i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(gt, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h2 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(le, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  test = (h2 <= (_Float16)42.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<4.200000e+01> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(le, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  test = (h2 <= f0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (f2 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (i0 <= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(le, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 <= i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(le, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 >= h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i
  // NONATIVE-NEXT: %{{.+}} = cir.get_global @test : !cir.ptr<!u32i>

  //      NATIVE: %[[#A:]] = cir.cmp(ge, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  test = (h0 >= (_Float16)-2.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#D]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.double, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.cmp(ge, %{{.+}}, %[[#C]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#D]] : !s32i), !u32i

  test = (h0 >= f2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (f0 >= h2);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (i0 >= h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(ge, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 >= i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ge, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h1 == h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(eq, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  test = (h1 == (_Float16)1.0);
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(eq, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  test = (h1 == f1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (f1 == h1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (i0 == h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(eq, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 == i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(eq, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h1 != h2);
  //      NONATIVE: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cmp(ne, %{{.+}}, %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#A]] : !s32i), !u32i

  test = (h1 != (_Float16)1.0);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cmp(ne, %{{.+}}, %[[#B]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#C]] : !s32i), !u32i

  test = (h1 != f1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (f1 != h1);
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.float, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (i0 != h0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NONATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.cmp(ne, %[[#A]], %{{.+}}) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  test = (h0 != i0);
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NONATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.cmp(ne, %{{.+}}, %[[#A]]) : !cir.f16, !s32i
  // NATIVE-NEXT: %{{.+}} = cir.cast(integral, %[[#B]] : !s32i), !u32i

  h1 = (h1 ? h2 : h0);
  //      NONATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NONATIVE-NEXT: %{{.+}} = cir.ternary(%[[#A]], true {
  //      NONATIVE:   cir.yield %{{.+}} : !cir.f16
  // NONATIVE-NEXT: }, false {
  //      NONATIVE:   cir.yield %{{.+}} : !cir.f16
  // NONATIVE-NEXT: }) : (!cir.bool) -> !cir.f16
  //      NONATIVE: %{{.+}} = cir.get_global @h1 : !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(float_to_bool, %{{.+}} : !cir.f16), !cir.bool
  // NATIVE-NEXT: %[[#B:]] = cir.ternary(%[[#A]], true {
  //      NATIVE:   cir.yield %{{.+}} : !cir.f16
  // NATIVE-NEXT: }, false {
  //      NATIVE:   cir.yield %{{.+}} : !cir.f16
  // NATIVE-NEXT: }) : (!cir.bool) -> !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  h0 = h1;
  //      NONATIVE: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @h1 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#B]], %[[#C]] : !cir.f16, !cir.ptr<!cir.f16>

  h0 = (_Float16)-2.0f;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<2.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.unary(minus, %[[#A]]) : !cir.float, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  h0 = f0;
  //      NONATIVE: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @f0 : !cir.ptr<!cir.float>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.float>, !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  h0 = i0;
  //      NONATIVE: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!s32i>, !s32i
  // NATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  i0 = h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @i0 : !cir.ptr<!s32i>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !s32i, !cir.ptr<!s32i>

  h0 += h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(add, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 += (_Float16)1.0f;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#D:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.binop(add, %[[#D]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.cast(floating, %[[#E]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#F]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.float), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(add, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 += f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  i0 += h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(add, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(add, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  h0 += i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(add, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(add, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 -= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(sub, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 -= (_Float16)1.0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(sub, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 -= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  i0 -= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(sub, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(sub, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  h0 -= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(sub, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(sub, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 *= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(mul, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(mul, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 *= (_Float16)1.0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(mul, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 *= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  i0 *= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(mul, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(mul, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  h0 *= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(mul, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(mul, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 /= h1;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.binop(div, %{{.+}}, %{{.+}}) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#A]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 /= (_Float16)1.0;
  //      NONATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#D:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.binop(div, %[[#D]], %[[#C]]) : !cir.float
  // NONATIVE-NEXT: %[[#F:]] = cir.cast(floating, %[[#E]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#F]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.const #cir.fp<1.000000e+00> : !cir.double
  // NATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.double), !cir.f16
  //      NATIVE: %[[#C:]] = cir.binop(div, %{{.+}}, %[[#B]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 /= f2;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.float
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.float), !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  i0 /= h0;
  //      NONATIVE: %[[#A:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  //      NONATIVE: %[[#B:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.float
  // NONATIVE-NEXT: %[[#C:]] = cir.binop(div, %[[#B]], %[[#A]]) : !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(float_to_int, %[[#C]] : !cir.float), !s32i
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NATIVE-NEXT: %[[#B:]] = cir.binop(div, %[[#A]], %{{.+}}) : !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(float_to_int, %[[#B]] : !cir.f16), !s32i
  // NATIVE-NEXT: cir.store volatile %[[#C]], %{{.+}} : !s32i, !cir.ptr<!s32i>

  h0 /= i0;
  //      NONATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  // NONATIVE-NEXT: %[[#B:]] = cir.cast(floating, %[[#A]] : !cir.f16), !cir.float
  //      NONATIVE: %[[#C:]] = cir.cast(floating, %{{.+}} : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.binop(div, %[[#C]], %[[#B]]) : !cir.float
  // NONATIVE-NEXT: %[[#E:]] = cir.cast(floating, %[[#D]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: cir.store volatile %[[#E]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.cast(int_to_float, %{{.+}} : !s32i), !cir.f16
  //      NATIVE: %[[#B:]] = cir.binop(div, %{{.+}}, %[[#A]]) : !cir.f16
  // NATIVE-NEXT: cir.store volatile %[[#B]], %{{.+}} : !cir.f16, !cir.ptr<!cir.f16>

  h0 = d0;
  //      NONATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  h0 = (float)d0;
  //      NONATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NONATIVE-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.double>, !cir.double
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.double), !cir.float
  // NATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.f16
  // NATIVE-NEXT: %[[#E:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.f16, !cir.ptr<!cir.f16>

  d0 = h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.double
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.double
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.double, !cir.ptr<!cir.double>

  d0 = (float)h0;
  //      NONATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NONATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.double
  // NONATIVE-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NONATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  //      NATIVE: %[[#A:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: %[[#B:]] = cir.load volatile %[[#A]] : !cir.ptr<!cir.f16>, !cir.f16
  // NATIVE-NEXT: %[[#C:]] = cir.cast(floating, %[[#B]] : !cir.f16), !cir.float
  // NATIVE-NEXT: %[[#D:]] = cir.cast(floating, %[[#C]] : !cir.float), !cir.double
  // NATIVE-NEXT: %[[#E:]] = cir.get_global @d0 : !cir.ptr<!cir.double>
  // NATIVE-NEXT: cir.store volatile %[[#D]], %[[#E]] : !cir.double, !cir.ptr<!cir.double>

  h0 = s0;
  //      NONATIVE: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // NONATIVE-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // NONATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s16i), !cir.f16
  // NONATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NONATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>

  //      NATIVE: %[[#A:]] = cir.get_global @s0 : !cir.ptr<!s16i>
  // NATIVE-NEXT: %[[#B:]] = cir.load %[[#A]] : !cir.ptr<!s16i>, !s16i
  // NATIVE-NEXT: %[[#C:]] = cir.cast(int_to_float, %[[#B]] : !s16i), !cir.f16
  // NATIVE-NEXT: %[[#D:]] = cir.get_global @h0 : !cir.ptr<!cir.f16>
  // NATIVE-NEXT: cir.store volatile %[[#C]], %[[#D]] : !cir.f16, !cir.ptr<!cir.f16>
}
