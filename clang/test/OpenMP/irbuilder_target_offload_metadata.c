// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-enable-irbuilder -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-enable-irbuilder -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-enable-irbuilder -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST

// With OMPIRBuilder enabled, this target region must compile for both the host
// and the device. Previously, the host recorded the same target region
// multiple times in its offload metadata, causing device compilation to crash.
void foo(void) {
#pragma omp target
  {}
}

// The host metadata must list this target region exactly once.
// HOST: !omp_offload.info = !{![[ENTRY:[0-9]+]]}{{$}}
// HOST: ![[ENTRY]] = !{i32 0, {{.*}}!"foo", {{.*}}}

// DEVICE-LABEL: define {{.*}}@__omp_offloading_{{.*}}_foo_l{{[0-9]+}}(
