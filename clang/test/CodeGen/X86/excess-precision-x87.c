// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -target-feature -sse \
// RUN:   -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=SRC
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -target-feature -sse \
// RUN:   -ffp-eval-method=extended -O1 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=EXT

// SRC-LABEL: define {{.*}}float @expr(
// SRC: fmul float
// SRC: fmul float
// SRC: fadd float
// SRC-NOT: x86_fp80
//
// EXT-LABEL: define {{.*}}float @expr(
// EXT: fmul x86_fp80
// EXT: fmul x86_fp80
// EXT: fadd x86_fp80
// EXT: fptrunc x86_fp80 {{.*}} to float
float expr(float a, float b, float c, float d) {
  return a * b + c * d;
}

// SRC-LABEL: define {{.*}}i32 @eval_method(
// SRC: ret i32 0
//
// EXT-LABEL: define {{.*}}i32 @eval_method(
// EXT: ret i32 2
int eval_method(void) {
  return __FLT_EVAL_METHOD__;
}
