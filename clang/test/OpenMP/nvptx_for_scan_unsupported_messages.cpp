// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o -

int thread_data[64];
int scan[64];
int gsum;

void foo(int n) {
#pragma omp target parallel
  {
    // expected-error@+1 {{cannot compile this 'inscan' reduction on a worksharing loop in a target region yet}}
#pragma omp for reduction(inscan, + : gsum)
    for (int i = 0; i < n; ++i) {
      gsum += thread_data[i];
#pragma omp scan inclusive(gsum)
      scan[i] = gsum;
    }
  }
}
