// RUN: %clang_cc1 -triple x86_64 %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -ffp-exception-behavior=strict -triple x86_64 %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-STRICT

float fminf (float, float);
double fmin (double, double);
long double fminl (long double, long double);
float fmaxf (float, float);
double fmax (double, double);
long double fmaxl (long double, long double);

// CHECK: call nsz float @llvm.minnum.f32
// CHECK-STRICT: call nsz float @llvm.experimental.constrained.minnum.f32{{.*}} #2
float fmin1(float a, float b) {
        return fminf(a, b);
}
// CHECK: call nsz double @llvm.minnum.f64
// CHECK-STRICT: call nsz double @llvm.experimental.constrained.minnum.f64{{.*}} #2
float fmin2(double a, double b) {
        return fmin(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.minnum.f80
// CHECK-STRICT: call nsz x86_fp80 @llvm.experimental.constrained.minnum.f80{{.*}} #2
float fmin3(long double a, long double b) {
        return fminl(a, b);
}
// CHECK: call nsz float @llvm.maxnum.f32
// CHECK-STRICT: call nsz float @llvm.experimental.constrained.maxnum.f32{{.*}} #2
float fmax1(float a, float b) {
        return fmaxf(a, b);
}
// CHECK: call nsz double @llvm.maxnum.f64
// CHECK-STRICT: call nsz double @llvm.experimental.constrained.maxnum.f64{{.*}} #2
float fmax2(double a, double b) {
        return fmax(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.maxnum.f80
// CHECK-STRICT: call nsz x86_fp80 @llvm.experimental.constrained.maxnum.f80{{.*}} #2
float fmax3(long double a, long double b) {
        return fmaxl(a, b);
}

//CHECK-STRICT: attributes #2 = { strictfp }
