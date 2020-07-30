// RUN: %clang_cc1 -triple ve-linux-gnu -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define { float, float } @p(float %a.coerce0, float %a.coerce1, float %b.coerce0, float %b.coerce1) #0 {
float __complex__ p(float __complex__ a, float __complex__ b) {
}

// CHECK-LABEL: define { double, double } @q(double %a.coerce0, double %a.coerce1, double %b.coerce0, double %b.coerce1) #0 {
double __complex__ q(double __complex__ a, double __complex__ b) {
}

void func() {
  // CHECK-LABEL: %call = call i32 (i32, i32, i32, i32, i32, i32, i32, ...) bitcast (i32 (...)* @hoge to i32 (i32, i32, i32, i32, i32, i32, i32, ...)*)(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  hoge(1, 2, 3, 4, 5, 6, 7);
}
