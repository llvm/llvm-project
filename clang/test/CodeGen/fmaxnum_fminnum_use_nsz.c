// RUN: %clang_cc1 -triple x86_64 %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK

float fminf (float, float);
double fmin (double, double);
long double fminl (long double, long double);
float fmaxf (float, float);
double fmax (double, double);
long double fmaxl (long double, long double);

// CHECK: call nsz float @llvm.minnum.f32
float fmin1(float a, float b) {
        return fminf(a, b);
}
// CHECK: call nsz double @llvm.minnum.f64
float fmin2(double a, double b) {
        return fmin(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.minnum.f80
float fmin3(long double a, long double b) {
        return fminl(a, b);
}
// CHECK: call nsz float @llvm.maxnum.f32
float fmax1(float a, float b) {
        return fmaxf(a, b);
}
// CHECK: call nsz double @llvm.maxnum.f64
float fmax2(double a, double b) {
        return fmax(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.maxnum.f80
float fmax3(long double a, long double b) {
        return fmaxl(a, b);
}
