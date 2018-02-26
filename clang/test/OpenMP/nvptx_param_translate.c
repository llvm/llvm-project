// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

// CHECK: [[MAP_FN:%.+]] = load void (i8*, ...)*, void (i8*, ...)** %
// CHECK: call void (i8*, ...) [[MAP_FN]](i8* %
int main() {
  double a, b;

#pragma omp target map(tofrom      \
                       : a) map(to \
                                : b)
  {
#pragma omp taskgroup
#pragma omp task shared(a)
    a = b;
  }
  return 0;
}
