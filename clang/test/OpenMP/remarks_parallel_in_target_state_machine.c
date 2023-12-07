// RUN: %clang_cc1                                 -verify=host -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify      -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out

// host-no-diagnostics

void baz(void) __attribute__((assume("omp_no_openmp")));

void bar(void) {
#pragma omp parallel // #1                                                                                                                                                                                                                                                                                                                                           \
                     // expected-remark@#1 {{Parallel region is used in unknown ways. Will not attempt to rewrite the state machine. [OMP101]}}
  {
  }
}

void foo(void) {
#pragma omp target teams // #2
                         // expected-remark@#2 {{Rewriting generic-mode kernel with a customized state machine. [OMP131]}}
  {
    baz();               // expected-remark {{Value has potential side effects preventing SPMD-mode execution. Add `__attribute__((assume("ompx_spmd_amenable")))` to the called function to override. [OMP121]}}
#pragma omp parallel
    {
    }
    bar();
#pragma omp parallel
    {
    }
  }
}

void spmd(void) {
  // Verify we do not emit the remarks above for "SPMD" regions.
#pragma omp target teams
#pragma omp parallel
  {
  }

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 100; ++i) {
  }
}

#pragma omp begin declare target device_type(nohost)
struct KernelEnvironmentTy;
__attribute__((weak))
extern "C" int __kmpc_target_init(struct KernelEnvironmentTy *) { // expected-remark {{Could not internalize function. Some optimizations may not be possible. [OMP140]}}
  return 0;
}
#pragma omp end declare target

// expected-remark@* {{OpenMP runtime call __kmpc_global_thread_num deduplicated. [OMP170]}}
