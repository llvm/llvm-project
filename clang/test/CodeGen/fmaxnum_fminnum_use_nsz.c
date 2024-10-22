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
float fmin32(float a, float b) {
        return fminf(a, b);
}
// CHECK: call nsz float @llvm.minnum.f32
// CHECK-STRICT: call nsz float @llvm.experimental.constrained.minnum.f32{{.*}} #2
float fmin32b(float a, float b) {
        return __builtin_fminf(a, b);
}
// CHECK: call nsz double @llvm.minnum.f64
// CHECK-STRICT: call nsz double @llvm.experimental.constrained.minnum.f64{{.*}} #2
float fmin64(double a, double b) {
        return fmin(a, b);
}
// CHECK: call nsz double @llvm.minnum.f64
// CHECK-STRICT: call nsz double @llvm.experimental.constrained.minnum.f64{{.*}} #2
float fmin64b(double a, double b) {
        return __builtin_fmin(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.minnum.f80
// CHECK-STRICT: call nsz x86_fp80 @llvm.experimental.constrained.minnum.f80{{.*}} #2
float fmin80(long double a, long double b) {
        return fminl(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.minnum.f80
// CHECK-STRICT: call nsz x86_fp80 @llvm.experimental.constrained.minnum.f80{{.*}} #2
float fmin80b(long double a, long double b) {
        return __builtin_fminl(a, b);
}
// CHECK: call nsz float @llvm.maxnum.f32
// CHECK-STRICT: call nsz float @llvm.experimental.constrained.maxnum.f32{{.*}} #2
float fmax32(float a, float b) {
        return fmaxf(a, b);
}
// CHECK: call nsz float @llvm.maxnum.f32
// CHECK-STRICT: call nsz float @llvm.experimental.constrained.maxnum.f32{{.*}} #2
float fmax32b(float a, float b) {
        return __builtin_fmaxf(a, b);
}
// CHECK: call nsz double @llvm.maxnum.f64
// CHECK-STRICT: call nsz double @llvm.experimental.constrained.maxnum.f64{{.*}} #2
float fmax64(double a, double b) {
        return fmax(a, b);
}
// CHECK: call nsz double @llvm.maxnum.f64
// CHECK-STRICT: call nsz double @llvm.experimental.constrained.maxnum.f64{{.*}} #2
float fmax64b(double a, double b) {
        return __builtin_fmax(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.maxnum.f80
// CHECK-STRICT: call nsz x86_fp80 @llvm.experimental.constrained.maxnum.f80{{.*}} #2
float fmax3(long double a, long double b) {
        return fmaxl(a, b);
}
// CHECK: call nsz x86_fp80 @llvm.maxnum.f80
// CHECK-STRICT: call nsz x86_fp80 @llvm.experimental.constrained.maxnum.f80{{.*}} #2
float fmax80b(long double a, long double b) {
        return __builtin_fmaxl(a, b);
}

//CHECK-STRICT: attributes #2 = { strictfp }
