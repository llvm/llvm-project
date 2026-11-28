// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes \
// RUN:   -fopenmp -ffast-math -emit-llvm %s -o - | FileCheck %s

// Calls whose scalar signature cannot be vectorized must not get a mapping.
// Clang used to crash on the complex and variadic cases and used to attach
// mappings that name a variant with a different signature.

// A complex type cannot be a vector element type.
#pragma omp declare simd notinbranch
float _Complex cexpf(float _Complex);
float _Complex complex_call(float _Complex z) { return cexpf(z); }

// Variadic functions have no vector function type.
#pragma omp declare simd
int printf(const char *, ...);
int printf_call(void) { return printf("hi"); }

// The mapped symbol already exists with another signature.
#pragma omp declare simd notinbranch
double sin(double);
float collide(double) __asm__("_ZGVbN2v_sin");
float collide(double x) { return (float)x; }
double collision_call(double x) { return sin(x); }

// The call goes through a function pointer with a different signature.
double mismatch_call(double x) {
  return ((double (*)(double, double))sin)(x, 0.0);
}

// no_builtin on the caller disables the mapping.
__attribute__((no_builtin("sin"))) double no_builtin_sin(double x) {
  return sin(x);
}
__attribute__((no_builtin)) double no_builtin_all(double x) { return sin(x); }

// A function renamed with an asm label is not the C library function.
#pragma omp declare simd notinbranch
double cos(double) __asm__("custom_cos");
double asm_label_call(double x) { return cos(x); }

// CHECK-LABEL: define{{.*}} @complex_call(
// CHECK-NOT: "vector-function-abi-variant"
