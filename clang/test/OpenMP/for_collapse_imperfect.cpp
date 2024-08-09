// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -verify=host -O2 -triple x86_64-unknown-unknown -Rpass-analysis=openmp-opt -fopenmp -x c++ -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-x86_64-host.bc
// RUN: %clang_cc1 -verify=analysis -O2 -triple amdgcn-amd-amdhsa -Rpass-analysis=openmp-opt -fopenmp -x c++ -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86_64-host.bc -o %t.out

// host-no-diagnostics

#define N 256

int main() {
  double arr[N][N];
  double b[N];
  float c[N];

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      arr[j][i] = 0.0;

  // These nested loops look parallelisable at a glance, but if they are
  // collapsed, iterations are no longer data-independent with respect to each
  // other.  So we emit a remark saying so.
#pragma omp target map(tofrom: arr)
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    arr[i][i] = i * 10; // #0
    // analysis-remark@#0 {{Collapsing imperfectly-nested loop may introduce unexpected data dependencies}}
    for (int j = 0; j < N; j++) {
      arr[i][j]++;
    }
  }

  // This is fine, the declaration of 'f' can't affect the array 'arr'.
#pragma omp target map(tofrom: arr)
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    double f = i * 10;
    for (int j = 0; j < N; j++) {
      arr[i][j] += (i == j) ? f : 1;
    }
  }

  // The accesses in this loop could be disambiguated, but currently aren't.
  // So this is a false positive for the remark.
#pragma omp target map(tofrom: arr, b[0:N])
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    b[i] = i; // #1
    // analysis-remark@#1 {{Collapsing imperfectly-nested loop may introduce unexpected data dependencies}}
    for (int j = 0; j < N; j++) {
      arr[i][j]++;
    }
  }

  // This is fine though, presumably TBAA takes care of it.  No remark emitted.
#pragma omp target map(tofrom: arr, c[0:N])
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    c[i] = i;
    for (int j = 0; j < N; j++) {
      arr[i][j]++;
    }
  }

  return 0;
}