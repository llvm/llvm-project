// RUN: %clang_cc1                                 -verify=host      -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1                                 -verify=all,safe  -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out
// RUN: %clang_cc1 -verify=all,safe  -Rpass=openmp-opt -Rpass-analysis=openmp-opt -fopenmp -O2 -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.out

// host-no-diagnostics

void baz(void) __attribute__((assume("omp_no_openmp")));

void bar1(void) {
#pragma omp parallel // #0
                     // safe-remark@#0 {{Parallel region is used in unknown ways. Will not attempt to rewrite the state machine. [OMP101]}}
                     // all-remark@#0 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // all-remark@#0 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // all-remark@#0 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#0 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
  {
  }
}
void bar2(void) {
#pragma omp parallel // #1
                     // safe-remark@#1 {{Parallel region is used in unknown ways. Will not attempt to rewrite the state machine. [OMP101]}}
                     // all-remark@#1 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // all-remark@#1 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#1 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
  {
  }
}

void foo1(void) {
#pragma omp target teams // #2
                         // all-remark@#2 {{Rewriting generic-mode kernel with a customized state machine. [OMP131]}}

  {
    baz();           // all-remark {{Value has potential side effects preventing SPMD-mode execution. Add `__attribute__((assume("ompx_spmd_amenable")))` to the called function to override. [OMP121]}}
#pragma omp parallel // #3
                     // all-remark@#3 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#3 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
    {
    }
    bar1();
#pragma omp parallel // #4
                     // all-remark@#4 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#4 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
    {
    }
  }
}

void foo2(void) {
#pragma omp target teams // #5
                         // all-remark@#5 {{Rewriting generic-mode kernel with a customized state machine. [OMP131]}}
  {
    baz();           // all-remark {{Value has potential side effects preventing SPMD-mode execution. Add `__attribute__((assume("ompx_spmd_amenable")))` to the called function to override. [OMP121]}}
#pragma omp parallel // #6
                     // all-remark@#6 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#6 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
    {
    }
    bar1();
    bar2();
#pragma omp parallel // #7
                     // all-remark@#7 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#7 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
    {
    }
    bar1();
    bar2();
  }
}

void foo3(void) {
#pragma omp target teams // #8
                         // all-remark@#8 {{Rewriting generic-mode kernel with a customized state machine. [OMP131]}}
  {
    baz();           // all-remark {{Value has potential side effects preventing SPMD-mode execution. Add `__attribute__((assume("ompx_spmd_amenable")))` to the called function to override. [OMP121]}}
#pragma omp parallel // #9
                     // all-remark@#9 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#9 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
    {
    }
    bar1();
    bar2();
#pragma omp parallel // #10
                     // all-remark@#10 {{Value has potential side effects preventing SPMD-mode execution. [OMP121]}}
                     // safe-remark@#10 {{Replaced globalized variable with 1 byte of shared memory. [OMP111]}}
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

// all-remark@* 9 {{OpenMP runtime call __kmpc_global_thread_num deduplicated. [OMP170]}}
