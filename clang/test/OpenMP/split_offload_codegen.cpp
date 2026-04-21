// Split inside `#pragma omp target` — host and device IR show `.split.iv`.
//
// RUN: %clang_cc1 -DCK_SPLIT -verify -fopenmp -fopenmp-version=60 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - 2>&1 | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 -DCK_SPLIT -verify -fopenmp -fopenmp-version=60 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-split-host.bc
// RUN: %clang_cc1 -DCK_SPLIT -verify -fopenmp -fopenmp-version=60 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-split-host.bc -o - 2>&1 | FileCheck -check-prefix=DEVICE %s

// expected-no-diagnostics

#ifdef CK_SPLIT
extern "C" void body(int);

void host_split_in_target(int n) {
#pragma omp target map(to : n)
  {
#pragma omp split counts(2, omp_fill)
    for (int i = 0; i < n; ++i)
      body(i);
  }
}

// HOST: define {{.*}}void {{.*}}host_split_in_target
// HOST: .split.iv
// HOST: __tgt_target_kernel

// DEVICE: define {{.*}}void @__omp_offloading_
// DEVICE: .split.iv
#endif
