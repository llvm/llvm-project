// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -debug-info-kind=line-tables-only -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc

int test() {
  int c;

#pragma omp target data map(tofrom: c)
{
  #pragma omp target nowait
  {
      c = 2;
  }
}
  return c;
}
