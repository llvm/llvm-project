// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple arm-none-linux-gnueabi %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple aarch64-none-linux-gnueabi %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple x86_64-linux-gnu %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple arm-none-linux-gnueabi -fallow-half-arguments-and-returns %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple aarch64-none-linux-gnueabi -fallow-half-arguments-and-returns %s | FileCheck %s --check-prefix=NOTNATIVE --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple arm-none-linux-gnueabi -fnative-half-type %s \
// RUN:   | FileCheck %s --check-prefix=NATIVE-HALF
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple aarch64-none-linux-gnueabi -fnative-half-type %s \
// RUN:   | FileCheck %s --check-prefix=NATIVE-HALF
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -x renderscript %s \
// RUN:   | FileCheck %s --check-prefix=NATIVE-HALF
typedef unsigned cond_t;
typedef __fp16 float16_t;

volatile cond_t test;
volatile int i0;
volatile __fp16 h0 = 0.0, h1 = 1.0, h2;
volatile float f0, f1, f2;
volatile double d0;
short s0;

void foo(void) {
  // CHECK-LABEL: define{{.*}} void @foo()

  // Check unary ops

  // NOTNATIVE: [[F16TOF32:fpext half]]
  // CHECK: fptoui float
  // NATIVE-HALF: fptoui half
  test = (h0);
  // CHECK: uitofp i32
  // NOTNATIVE: [[F32TOF16:fptrunc float]]
  // NATIVE-HALF: uitofp i32 {{.*}} to half
  h0 = (test);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fcmp une half
  test = (!h1);
  // CHECK: [[F16TOF32]]
  // CHECK: fneg float
  // NOTNATIVE: [[F32TOF16]]
  // NATIVE-HALF: fneg half
  h1 = -h1;
  // CHECK: [[F16TOF32]]
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: load volatile half
  // NATIVE-HALF-NEXT: store volatile half
  h1 = +h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h1++;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  ++h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  --h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h1--;

  // Check binary ops with various operands
  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fmul half
  h1 = h0 * h2;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fmul half
  h1 = h0 * (__fp16) -2.0f;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fmul float
  h1 = h0 * f2;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fmul float
  h1 = f0 * h2;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fmul half
  h1 = h0 * i0;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fdiv half
  h1 = (h0 / h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fdiv half
  h1 = (h0 / (__fp16) -2.0f);
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fdiv float
  h1 = (h0 / f2);
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fdiv float
  h1 = (f0 / h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fdiv half
  h1 = (h0 / i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h1 = (h2 + h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h1 = ((__fp16)-2.0 + h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fadd float
  h1 = (h2 + f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fadd float
  h1 = (f2 + h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h1 = (h0 + i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fsub half
  h1 = (h2 - h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fsub half
  h1 = ((__fp16)-2.0f - h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fsub float
  h1 = (h2 - f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fsub float
  h1 = (f2 - h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fsub half
  h1 = (h0 - i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt float
  // NATIVE-HALF: fcmp olt half
  test = (h2 < h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt float
  // NATIVE-HALF: fcmp olt half
  test = (h2 < (__fp16)42.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp olt float
  test = (h2 < f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp olt float
  test = (f2 < h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt float
  // NATIVE-HALF: fcmp olt half
  test = (i0 < h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp olt float
  // NATIVE-HALF: fcmp olt half
  test = (h0 < i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt float
  // NATIVE-HALF: fcmp ogt half
  test = (h0 > h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt float
  // NATIVE-HALF: fcmp ogt half
  test = ((__fp16)42.0 > h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp ogt float
  test = (h0 > f2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp ogt float
  test = (f0 > h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt float
  // NATIVE-HALF: fcmp ogt half
  test = (i0 > h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ogt float
  // NATIVE-HALF: fcmp ogt half
  test = (h0 > i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole float
  // NATIVE-HALF: fcmp ole half
  test = (h2 <= h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole float
  // NATIVE-HALF: fcmp ole half
  test = (h2 <= (__fp16)42.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp ole float
  test = (h2 <= f0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp ole float
  test = (f2 <= h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole float
  // NATIVE-HALF: fcmp ole half
  test = (i0 <= h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp ole float
  // NATIVE-HALF: fcmp ole half
  test = (h0 <= i0);


  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge float
  // NATIVE-HALF: fcmp oge half
  test = (h0 >= h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge float
  // NATIVE-HALF: fcmp oge half
  test = (h0 >= (__fp16)-2.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp oge float
  test = (h0 >= f2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp oge float
  test = (f0 >= h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge float
  // NATIVE-HALF: fcmp oge half
  test = (i0 >= h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oge float
  // NATIVE-HALF: fcmp oge half
  test = (h0 >= i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq float
  // NATIVE-HALF: fcmp oeq half
  test = (h1 == h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq float
  // NATIVE-HALF: fcmp oeq half
  test = (h1 == (__fp16)1.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp oeq float
  test = (h1 == f1);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp oeq float
  test = (f1 == h1);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq float
  // NATIVE-HALF: fcmp oeq half
  test = (i0 == h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp oeq float
  // NATIVE-HALF: fcmp oeq half
  test = (h0 == i0);

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fcmp une half
  test = (h1 != h2);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fcmp une half
  test = (h1 != (__fp16)1.0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp une float
  test = (h1 != f1);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fcmp une float
  test = (f1 != h1);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fcmp une half
  test = (i0 != h0);
  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // NATIVE-HALF: fcmp une half
  test = (h0 != i0);

  // CHECK: [[F16TOF32]]
  // CHECK: fcmp une float
  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fcmp une half {{.*}}, 0xH0000
  h1 = (h1 ? h2 : h0);
  // Check assignments (inc. compound)
  h0 = h1;
  // NOTNATIVE: store {{.*}} half 0xHC000
  // NATIVE-HALF: store {{.*}} half 0xHC000
  h0 = (__fp16)-2.0f;
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fptrunc float
  h0 = f0;

  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  h0 = i0;
  // CHECK: [[F16TOF32]]
  // CHECK: fptosi float {{.*}} to i32
  // NATIVE-HALF: fptosi half {{.*}} to i32
  i0 = h0;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h0 += h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fadd half
  h0 += (__fp16)1.0f;
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fadd float
  // NATIVE-HALF: fptrunc float
  h0 += f2;
  // CHECK: [[F16TOF32]]
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: fadd float
  // CHECK: fptosi float {{.*}} to i32
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fadd half
  // NATIVE-HALF: fptosi half {{.*}} to i32
  i0 += h0;
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: [[F16TOF32]]
  // CHECK: fadd float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fadd half
  h0 += i0;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fsub half
  h0 -= h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fsub half
  h0 -= (__fp16)1.0;
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fsub float
  // NATIVE-HALF: fptrunc float
  h0 -= f2;
  // CHECK: [[F16TOF32]]
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: fsub float
  // CHECK: fptosi float {{.*}} to i32
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fsub half
  // NATIVE-HALF: fptosi half {{.*}} to i32
  i0 -= h0;
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: [[F16TOF32]]
  // CHECK: fsub float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fsub half
  h0 -= i0;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fmul half
  h0 *= h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fmul half
  h0 *= (__fp16)1.0;
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fmul float
  // NATIVE-HALF: fptrunc float
  h0 *= f2;
  // CHECK: [[F16TOF32]]
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: fmul float
  // CHECK: fptosi float {{.*}} to i32
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fmul half
  // NATIVE-HALF: fptosi half {{.*}} to i32
  i0 *= h0;
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: [[F16TOF32]]
  // CHECK: fmul float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fmul half
  h0 *= i0;

  // CHECK: [[F16TOF32]]
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fdiv half
  h0 /= h1;
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fdiv half
  h0 /= (__fp16)1.0;
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: fpext half
  // NATIVE-HALF: fdiv float
  // NATIVE-HALF: fptrunc float
  h0 /= f2;
  // CHECK: [[F16TOF32]]
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: fdiv float
  // CHECK: fptosi float {{.*}} to i32
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fdiv half
  // NATIVE-HALF: fptosi half {{.*}} to i32
  i0 /= h0;
  // CHECK: sitofp i32 {{.*}} to float
  // CHECK: [[F16TOF32]]
  // CHECK: fdiv float
  // CHECK: [[F32TOF16]]
  // NATIVE-HALF: sitofp i32 {{.*}} to half
  // NATIVE-HALF: fdiv half
  h0 /= i0;

  // Check conversions to/from double
  // NOTNATIVE: fptrunc double {{.*}} to half
  // NATIVE-HALF: fptrunc double {{.*}} to half
  h0 = d0;

  // CHECK: [[MID:%.*]] = fptrunc double {{%.*}} to float
  // NOTNATIVE: fptrunc float [[MID]] to half
  // NATIVE-HALF: [[MID:%.*]] = fptrunc double {{%.*}} to float
  // NATIVE-HALF: fptrunc float {{.*}} to half
  h0 = (float)d0;

  // NOTNATIVE: fpext half {{.*}} to double
  // NATIVE-HALF: fpext half {{.*}} to double
  d0 = h0;

  // NOTNATIVE: [[MID:%.*]] = fpext half {{.*}} to float
  // CHECK: fpext float [[MID]] to double
  // NATIVE-HALF: [[MID:%.*]] = fpext half {{.*}} to float
  // NATIVE-HALF: fpext float [[MID]] to double
  d0 = (float)h0;

  // NOTNATIVE: [[V1:%.*]] = load i16, i16* @s0
  // NOTNATIVE: [[CONV:%.*]] = sitofp i16 [[V1]] to float
  // NOTNATIVE: [[TRUNC:%.*]] = fptrunc float [[CONV]] to half
  // NOTNATIVE: store volatile half [[TRUNC]], half* @h0
  h0 = s0;
}

// CHECK-LABEL: define{{.*}} void @testTypeDef(
// CHECK: %[[CONV:.*]] = fpext <4 x half> %{{.*}} to <4 x float>
// CHECK: %[[CONV1:.*]] = fpext <4 x half> %{{.*}} to <4 x float>
// CHECK: %[[ADD:.*]] = fadd <4 x float> %[[CONV]], %[[CONV1]]
// CHECK: fptrunc <4 x float> %[[ADD]] to <4 x half>

void testTypeDef() {
  __fp16 t0 __attribute__((vector_size(8)));
  float16_t t1 __attribute__((vector_size(8)));
  t1 = t0 + t1;
}
