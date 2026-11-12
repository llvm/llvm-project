// RUN: %clang_cc1 -triple riscv32 -target-feature +d -target-feature +f -target-abi ilp32d \
// RUN:     -Wno-missing-declarations -emit-llvm %s -o - | FileCheck %s

struct empty_float2 { struct {}; float f; float g; };

// CHECK: define{{.*}} float @_Z14f_empty_float212empty_float2(float %0, float %1)
// CHECK: { [4 x i8], float, float }
float f_empty_float2(empty_float2 a) {
    return a.g;
}

struct empty_double2 { struct {}; double f; double g; };

// CHECK: define{{.*}} double @_Z15f_empty_double213empty_double2(double %0, double %1)
// CHECK: { [8 x i8], double, double }
double f_empty_double2(empty_double2 a) {
    return a.g;
}

struct empty_float_double { struct {}; float f; double g; };

// CHECK: define{{.*}} double @_Z20f_empty_float_double18empty_float_double(float %0, double %1)
// CHECK: { [4 x i8], float, double }
double f_empty_float_double(empty_float_double a) {
    return a.g;
}

struct empty_double_float { struct {}; double f; float g; };

// CHECK: define{{.*}} double @_Z20f_empty_double_float18empty_double_float(double %0, float %1)
// CHECK: { [8 x i8], double, float }
double f_empty_double_float(empty_double_float a) {
    return a.g;
}

struct empty_complex_f { struct {}; float _Complex fc; };

// CHECK: define{{.*}} float @_Z17f_empty_complex_f15empty_complex_f(float %0, float %1)
// CHECK: { [4 x i8], float, float }
float f_empty_complex_f(empty_complex_f a) {
    return __imag__ a.fc;
}

struct empty_complex_d { struct {}; double _Complex fc; };

// CHECK: define{{.*}} double @_Z17f_empty_complex_d15empty_complex_d(double %0, double %1)
// CHECK: { [8 x i8], double, double }
double f_empty_complex_d(empty_complex_d a) {
    return __imag__ a.fc;
}
