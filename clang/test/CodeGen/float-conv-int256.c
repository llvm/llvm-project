// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Test float <-> __int256_t conversions.

// === Signed -> Float ===

// CHECK-LABEL: define {{.*}}@int256_to_double
// CHECK: sitofp i256 %{{.*}} to double
double int256_to_double(__int256_t x) { return (double)x; }

// CHECK-LABEL: define {{.*}}@int256_to_float
// CHECK: sitofp i256 %{{.*}} to float
float int256_to_float(__int256_t x) { return (float)x; }

// CHECK-LABEL: define {{.*}}@int256_to_longdouble
// CHECK: sitofp i256 %{{.*}} to x86_fp80
long double int256_to_longdouble(__int256_t x) { return (long double)x; }

// === Unsigned -> Float ===

// CHECK-LABEL: define {{.*}}@uint256_to_double
// CHECK: uitofp i256 %{{.*}} to double
double uint256_to_double(__uint256_t x) { return (double)x; }

// CHECK-LABEL: define {{.*}}@uint256_to_float
// CHECK: uitofp i256 %{{.*}} to float
float uint256_to_float(__uint256_t x) { return (float)x; }

// CHECK-LABEL: define {{.*}}@uint256_to_longdouble
// CHECK: uitofp i256 %{{.*}} to x86_fp80
long double uint256_to_longdouble(__uint256_t x) { return (long double)x; }

// === Float -> Signed ===

// CHECK-LABEL: define {{.*}}@double_to_int256
// CHECK: fptosi double %{{.*}} to i256
__int256_t double_to_int256(double x) { return (__int256_t)x; }

// CHECK-LABEL: define {{.*}}@float_to_int256
// CHECK: fptosi float %{{.*}} to i256
__int256_t float_to_int256(float x) { return (__int256_t)x; }

// === Float -> Unsigned ===

// CHECK-LABEL: define {{.*}}@double_to_uint256
// CHECK: fptoui double %{{.*}} to i256
__uint256_t double_to_uint256(double x) { return (__uint256_t)x; }

// CHECK-LABEL: define {{.*}}@float_to_uint256
// CHECK: fptoui float %{{.*}} to i256
__uint256_t float_to_uint256(float x) { return (__uint256_t)x; }

// === Long Double -> Unsigned ===

// CHECK-LABEL: define {{.*}}@longdouble_to_uint256
// CHECK: fptoui x86_fp80 %{{.*}} to i256
__uint256_t longdouble_to_uint256(long double x) { return (__uint256_t)x; }

// === Long Double -> Signed ===

// CHECK-LABEL: define {{.*}}@longdouble_to_int256
// CHECK: fptosi x86_fp80 %{{.*}} to i256
__int256_t longdouble_to_int256(long double x) { return (__int256_t)x; }
