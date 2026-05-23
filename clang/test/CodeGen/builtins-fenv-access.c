// RUN: %clang_cc1 -fexperimental-strict-floating-point -ffp-exception-behavior=strict -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fexperimental-strict-floating-point -ffp-exception-behavior=strict -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter -emit-llvm %s -o - | FileCheck %s

#pragma STDC FENV_ACCESS ON

// CHECK-LABEL: define{{.*}} double @test_ceil_inexact(
// CHECK: call double @llvm.experimental.constrained.ceil.f64(
double test_ceil_inexact(void) { return __builtin_ceil(2.1); }

// CHECK-LABEL: define{{.*}} double @test_ceil_exact(
// CHECK: ret double 2.000000e+00
double test_ceil_exact(void) { return __builtin_ceil(2.0); }

// CHECK-LABEL: define{{.*}} double @test_floor_inexact(
// CHECK: call double @llvm.experimental.constrained.floor.f64(
double test_floor_inexact(void) { return __builtin_floor(2.1); }

// CHECK-LABEL: define{{.*}} double @test_floor_exact(
// CHECK: ret double 2.000000e+00
double test_floor_exact(void) { return __builtin_floor(2.0); }

// CHECK-LABEL: define{{.*}} double @test_trunc_inexact(
// CHECK: call double @llvm.experimental.constrained.trunc.f64(
double test_trunc_inexact(void) { return __builtin_trunc(2.1); }

// CHECK-LABEL: define{{.*}} double @test_trunc_exact(
// CHECK: ret double 2.000000e+00
double test_trunc_exact(void) { return __builtin_trunc(2.0); }

// CHECK-LABEL: define{{.*}} double @test_round_inexact(
// CHECK: call double @llvm.experimental.constrained.round.f64(
double test_round_inexact(void) { return __builtin_round(2.1); }

// CHECK-LABEL: define{{.*}} double @test_round_exact(
// CHECK: ret double 2.000000e+00
double test_round_exact(void) { return __builtin_round(2.0); }

// CHECK-LABEL: define{{.*}} double @test_roundeven_inexact(
// CHECK: call double @llvm.experimental.constrained.roundeven.f64(
double test_roundeven_inexact(void) { return __builtin_roundeven(2.1); }

// CHECK-LABEL: define{{.*}} double @test_roundeven_exact(
// CHECK: ret double 2.000000e+00
double test_roundeven_exact(void) { return __builtin_roundeven(2.0); }

// CHECK-LABEL: define{{.*}} double @test_fdim_inexact(
// CHECK: call double @fdim(
double test_fdim_inexact(void) { return __builtin_fdim(3.0, 0.1); }

// CHECK-LABEL: define{{.*}} double @test_fdim_exact(
// CHECK: ret double 2.000000e+00
double test_fdim_exact(void) { return __builtin_fdim(3.0, 1.0); }

// CHECK-LABEL: define{{.*}} double @test_fma_inexact(
// CHECK: call double @llvm.experimental.constrained.fma.f64(
double test_fma_inexact(void) { return __builtin_fma(3.0, 0.1, 0.01); }

// CHECK-LABEL: define{{.*}} double @test_fma_exact(
// CHECK: ret double 1.000000e+01
double test_fma_exact(void) { return __builtin_fma(2.0, 3.0, 4.0); }

// CHECK-LABEL: define{{.*}} double @test_scalbn_inexact(
// CHECK: call double @scalbn(
double test_scalbn_inexact(void) { return __builtin_scalbn(__DBL_MAX__, 1); }

// CHECK-LABEL: define{{.*}} double @test_scalbn_exact(
// CHECK: ret double 4.000000e+00
double test_scalbn_exact(void) { return __builtin_scalbn(2.0, 1); }

// CHECK-LABEL: define{{.*}} double @test_ldexp_inexact(
// CHECK: call double @llvm.experimental.constrained.ldexp.f64.i32(
double test_ldexp_inexact(void) { return __builtin_ldexp(__DBL_MAX__, 1); }

// CHECK-LABEL: define{{.*}} double @test_ldexp_exact(
// CHECK: ret double 4.000000e+00
double test_ldexp_exact(void) { return __builtin_ldexp(2.0, 1); }
