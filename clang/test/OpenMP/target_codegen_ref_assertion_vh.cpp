// REQUIRES: nvptx-registered-target
//
// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t.final
// expected-no-diagnostics

#pragma omp begin declare target
constexpr int A[] = {1,2,3,4,5};
int f(int i) { return A[i]; }
#pragma omp end declare target
