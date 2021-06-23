// RUN: %clang_cc1                                 -verify=host -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1                                 -verify      -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out
// RUN: %clang_cc1 -fexperimental-new-pass-manager -verify      -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out

// host-no-diagnostics

void bar(void) {
#pragma omp parallel // #1                                                                                                                                                                                                                                                                                                                                           \
                     // expected-remark@#1 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}} \
                     // expected-remark@#1 {{Parallel region is used in unknown ways; will not attempt to rewrite the state machine.}}
  {
  }
}

void foo(void) {
#pragma omp target teams // #2                                                                                                                                                                      \
                         // expected-remark@#2 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__1_wrapper, kernel ID: __omp_offloading}} \
                         // expected-remark@#2 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__2_wrapper, kernel ID: __omp_offloading}}
  {
#pragma omp parallel // #3                                                                                                                                                                                                                                                                                                                                           \
                     // expected-remark@#3 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}} \
                     // expected-remark@#3 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__1_wrapper, kernel ID: __omp_offloading}}
    {
    }
    bar();
#pragma omp parallel // #4                                                                                                                                                                                                                                                                                                                                           \
                     // expected-remark@#4 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}} \
                     // expected-remark@#4 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__2_wrapper, kernel ID: __omp_offloading}}
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

// expected-remark@* {{OpenMP runtime call __kmpc_global_thread_num moved to beginning of OpenMP region}}
// expected-remark@* 2 {{OpenMP runtime call __kmpc_global_thread_num deduplicated}}
