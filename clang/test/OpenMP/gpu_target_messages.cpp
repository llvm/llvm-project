// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple amdgcn-amd-amdhsa -emit-llvm %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple nvptx64-nvidia-cuda -emit-llvm %s

void foo() {
#pragma omp target // expected-error {{Cannot emit a '#pragma omp target' region on the GPU}}
  ;
}
