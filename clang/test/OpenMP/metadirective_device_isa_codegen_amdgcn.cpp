// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -w -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -w -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -target-cpu gfx906 -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

int amdgcn_device_isa_selected() {
  int threadCount = 0;

#pragma omp target map(tofrom \
                       : threadCount)
  {
#pragma omp metadirective                     \
    when(device = {isa("flat-address-space")} \
         : parallel) default(single)
    threadCount++;
  }

  return threadCount;
}

// CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}amdgcn_device_isa_selected
// CHECK: user_code.entry:
// CHECK: call void @__kmpc_parallel_51
// CHECK-NOT: call i32 @__kmpc_single
// CHECK: ret void

int amdgcn_device_isa_not_selected() {
  int threadCount = 0;

#pragma omp target map(tofrom \
                       : threadCount)
  {
#pragma omp metadirective                                      \
    when(device = {isa("sse")}                                 \
         : parallel)                                           \
        when(device = {isa("another-unsupported-gpu-feature")} \
             : parallel) default(single)
    threadCount++;
  }

  return threadCount;
}
// CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}amdgcn_device_isa_not_selected
// CHECK: user_code.entry:
// CHECK: call i32 @__kmpc_single
// CHECK-NOT: call void @__kmpc_parallel_51
// CHECK: ret void

#endif
