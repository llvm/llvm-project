// RUN: %clang_cc1                                 -verify=host      -Rpass=openmp-opt -Rpass-analysis=openmp -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1                                 -verify=all,safe  -Rpass=openmp-opt -Rpass-analysis=openmp -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out
// RUN: %clang_cc1 -fexperimental-new-pass-manager -verify=all,safe  -Rpass=openmp-opt -Rpass-analysis=openmp -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out

// host-no-diagnostics

void bar1(void) {
#pragma omp parallel // #0
                     // all-remark@#0 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // safe-remark@#0 {{Parallel region is used in unexpected ways; will not attempt to rewrite the state machine.}}
                     // force-remark@#0 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__2_wrapper, kernel ID: <NONE>}}
  {
  }
}
void bar2(void) {
#pragma omp parallel // #1
                     // all-remark@#1 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // safe-remark@#1 {{Parallel region is used in unexpected ways; will not attempt to rewrite the state machine.}}
                     // force-remark@#1 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__6_wrapper, kernel ID: <NONE>}}
  {
  }
}

void foo1(void) {
#pragma omp target teams // #2
                         // all-remark@#2 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__1_wrapper, kernel ID: __omp_offloading}}
                         // all-remark@#2 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__2_wrapper, kernel ID: __omp_offloading}}
  {
#pragma omp parallel // #3
                     // all-remark@#3 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#3 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__1_wrapper, kernel ID: __omp_offloading}}
    {
    }
    bar1();
#pragma omp parallel // #4
                     // all-remark@#4 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#4 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__2_wrapper, kernel ID: __omp_offloading}}
    {
    }
  }
}

void foo2(void) {
#pragma omp target teams // #5
                         // all-remark@#5 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__5_wrapper, kernel ID: __omp_offloading}}
                         // all-remark@#5 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__4_wrapper, kernel ID: __omp_offloading}}
  {
#pragma omp parallel // #6
                     // all-remark@#6 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#6 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__4_wrapper, kernel ID: __omp_offloading}}
    {
    }
    bar1();
    bar2();
#pragma omp parallel // #7
                     // all-remark@#7 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#7 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__5_wrapper, kernel ID: __omp_offloading}}
    {
    }
    bar1();
    bar2();
  }
}

void foo3(void) {
#pragma omp target teams // #8
                         // all-remark@#8 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__7_wrapper, kernel ID: __omp_offloading}}
                         // all-remark@#8 {{Target region containing the parallel region that is specialized. (parallel region ID: __omp_outlined__8_wrapper, kernel ID: __omp_offloading}}
  {
#pragma omp parallel // #9
                     // all-remark@#9 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#9 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__7_wrapper, kernel ID: __omp_offloading}}
    {
    }
    bar1();
    bar2();
#pragma omp parallel // #10
                     // all-remark@#10 {{Found a parallel region that is called in a target region but not part of a combined target construct nor nested inside a target construct without intermediate code. This can lead to excessive register usage for unrelated target regions in the same translation unit due to spurious call edges assumed by ptxas.}}
                     // all-remark@#10 {{Specialize parallel region that is only reached from a single target region to avoid spurious call edges and excessive register usage in other target regions. (parallel region ID: __omp_outlined__8_wrapper, kernel ID: __omp_offloading}}
    {
    }
    bar1();
    bar2();
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

// all-remark@* 5 {{OpenMP runtime call __kmpc_global_thread_num moved to beginning of OpenMP region}}
// all-remark@* 12 {{OpenMP runtime call __kmpc_global_thread_num deduplicated}}
