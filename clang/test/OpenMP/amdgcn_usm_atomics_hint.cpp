// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -DCHECK_HINTS -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -DCHECK_HINTS -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK-HINTS
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -DCHECK_FLAG_UNSAFE -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -DCHECK_FLAG_UNSAFE -munsafe-fp-atomics -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK-FLAG-UNSAFE

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#define N 1000

#define amd_fast_fp_atomics 1<<19
#define amd_safe_fp_atomics 1<<20

#pragma omp requires unified_shared_memory

#if defined CHECK_HINTS

double test_amdgcn_target_atomic_hints() {
// CHECK-HINTS-LABEL: define {{.*}} @{{.*}}test_amdgcn_target_atomic_hints

  double a = 0.0;
  double b = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:a,b)
  for (int i = 0; i < N; i++) {
    // CHECK-HINTS: call {{.*}} @llvm.amdgcn.global.atomic.fadd.f64.p0f64.f64
    #pragma omp atomic hint(amd_fast_fp_atomics)
    a+=(double)i;

    // CHECK-HINTS: {{.*}} = cmpxchg
    #pragma omp atomic hint(amd_safe_fp_atomics)
    b+=(double)i;
  }
  // CHECK-HINTS: ret void
  return a+b;
}
#endif // CHECK_HINTS

#if defined CHECK_FLAG_UNSAFE

double test_amdgcn_target_atomic_unsafe_opt() {
// CHECK-FLAG-UNSAFE-LABEL: define {{.*}} @{{.*}}test_amdgcn_target_atomic_unsafe_opt
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;

  #pragma omp target teams distribute parallel for map(tofrom:a,b,c)
  for (int i = 0; i < N; i++) {
    // CHECK-FLAG-UNSAFE: call {{.*}} @llvm.amdgcn.global.atomic.fadd.f64.p0f64.f64
    #pragma omp atomic
    a+=(double)i;

    // CHECK-FLAG-UNSAFE: call {{.*}} @llvm.amdgcn.global.atomic.fadd.f64.p0f64.f64
    #pragma omp atomic hint(amd_fast_fp_atomics)
    b+=(double)i;

    // CHECK-FLAG-UNSAFE: {{.*}} = cmpxchg
    #pragma omp atomic hint(amd_safe_fp_atomics)
    c+=(double)i;
  }

  return a+b+c;
}
#endif // CHECK_FLAG_UNSAFE

#endif // HEADER
