// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -emit-llvm -o - -triple arm-none-linux-gnueabi %s | FileCheck %s
typedef unsigned cond_t;

volatile cond_t test;
volatile __fp16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;

void foo(void) {
  // CHECK: define void @foo()

  // Check unary ops

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK fptoi float
  test = (h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp une float
  test = (!h1);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = -h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = +h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1++;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  ++h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  --h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1--;

  // Check binary ops with various operands
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = h0 * h2;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call i16 @llvm.convert.to.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = h0 * (__fp16) -2.0;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = h0 * f2;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = f0 * h2;

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h0 / h2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h0 / (__fp16) -2.0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h0 / f2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (f0 / h2);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h2 + h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = ((__fp16)-2.0 + h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h2 + f0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (f2 + h0);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h2 - h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = ((__fp16)-2.0 - h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h2 - f0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub float
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (f2 - h0);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp olt
  test = (h2 < h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp olt
  test = (h2 < (__fp16)42.0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp olt
  test = (h2 < f0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp olt
  test = (f2 < h0);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ogt
  test = (h0 > h2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ogt
  test = ((__fp16)42.0 > h2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ogt
  test = (h0 > f2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ogt
  test = (f0 > h2);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ole
  test = (h2 <= h0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ole
  test = (h2 <= (__fp16)42.0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ole
  test = (h2 <= f0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp ole
  test = (f2 <= h0);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oge
  test = (h0 >= h2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oge
  test = (h0 >= (__fp16)-2.0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oge
  test = (h0 >= f2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oge
  test = (f0 >= h2);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oeq
  test = (h1 == h2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oeq
  test = (h1 == (__fp16)1.0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oeq
  test = (h1 == f1);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp oeq
  test = (f1 == h1);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp une
  test = (h1 != h2);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp une
  test = (h1 != (__fp16)1.0);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp une
  test = (h1 != f1);
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp une
  test = (f1 != h1);

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fcmp une
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call i16 @llvm.convert.to.fp16
  h1 = (h1 ? h2 : h0);
  // Check assignments (inc. compound)
  h0 = h1;
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 = (__fp16)-2.0;
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 = f0;

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd float
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 += h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 += (__fp16)1.0;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fadd
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 += f2;

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 -= h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 -= (__fp16)1.0;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fsub
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 -= f2;

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 *= h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 *= (__fp16)1.0;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fmul
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 *= f2;

  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 /= h1;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 /= (__fp16)1.0;
  // CHECK: call float @llvm.convert.from.fp16
  // CHECK: fdiv
  // CHECK: call i16 @llvm.convert.to.fp16
  h0 /= f2;
}
