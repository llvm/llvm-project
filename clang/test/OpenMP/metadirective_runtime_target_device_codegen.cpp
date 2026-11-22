// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -o - | FileCheck %s
// expected-no-diagnostics

// A target region in a variant of a runtime metadirective gets a device
// kernel.
// CHECK: define {{.*}}void @__omp_offloading_{{.*}}_Z14target_varianti_l{{[0-9]+}}(
void target_variant(int flag) {
  int x = 0;
#pragma omp metadirective when(user={condition(flag)}: target map(tofrom: x)) otherwise(parallel)
  {
    x = 1;
  }
}
