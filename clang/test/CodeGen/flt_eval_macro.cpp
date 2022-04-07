// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point -DEXCEPT=1 \
// RUN: -fcxx-exceptions -triple x86_64-linux-gnu -emit-llvm -o - %s \
// RUN: | FileCheck -check-prefix=CHECK-SRC %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - %s -ffp-eval-method=source \
// RUN: | FileCheck -check-prefix=CHECK-SRC %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - %s -ffp-eval-method=double \
// RUN: | FileCheck -check-prefixes=CHECK-DBL %s

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - %s -ffp-eval-method=extended \
// RUN: | FileCheck -check-prefixes=CHECK-EXT-FLT %s

// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefix=CHECK-DBL-PPC

// RUN: %clang_cc1 -no-opaque-pointers -fexperimental-strict-floating-point -triple i386-linux-gnu \
// RUN: -emit-llvm -o - %s -ffp-eval-method=extended -mlong-double-80 \
// RUN: | FileCheck %s -check-prefix=CHECK-EXT-FLT

int getFEM() {
  // LABEL: define {{.*}}getFEM{{.*}}
  return __FLT_EVAL_METHOD__;
  // CHECK-SRC: ret {{.*}} 0
  // CHECK-DBL: ret {{.*}} 1
  // CHECK-DBL-PPC: ret {{.*}} 1
  // CHECK-EXT-FLT: ret {{.*}} 2
}

float func() {
  // LABEL: define {{.*}}@_Z4func{{.*}}
  float X = 100.0f;
  float Y = -45.3f;
  float Z = 393.78f;
  float temp;
#if __FLT_EVAL_METHOD__ == 0
  temp = X + Y + Z;
#elif __FLT_EVAL_METHOD__ == 1
  temp = X * Y * Z;
#elif __FLT_EVAL_METHOD__ == 2
  temp = X * Y - Z;
#endif
  // CHECK-SRC: load float, float*
  // CHECK-SRC: load float, float*
  // CHECK-SRC: fadd float
  // CHECK-SRC: load float, float*
  // CHECK-SRC: fadd float

  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float
  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float
  // CHECK-DBL: fmul double
  // CHECK-DBL: load float, float*
  // CHECK-DBL: fpext float
  // CHECK-DBL: fmul double
  // CHECK-DBL: fptrunc double

  // CHECK-EXT-FLT: load float, float*
  // CHECK-EXT-FLT: fpext float
  // CHECK-EXT-FLT: load float, float*
  // CHECK-EXT-FLT: fpext float
  // CHECK-EXT-FLT: fmul x86_fp80
  // CHECK-EXT-FLT: load float, float*
  // CHECK-EXT-FLT: fpext float
  // CHECK-EXT-FLT: fsub x86_fp80
  // CHECK-EXT-FLT: fptrunc x86_fp80

  // CHECK-DBL-PPC: load float, float*
  // CHECK-DBL-PPC: load float, float*
  // CHECK-DBL-PPC: fmul float
  // CHECK-DBL-PPC: load float, float*
  // CHECK-DBL-PPC: fmul float

  return temp;
}
