// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu gfx900 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics


// Check that the target attributes are set on the generated kernel
void func() {
  // CHECK: amdgpu_kernel void @__omp_offloading[[HASH:.*]]_l15() #0
  // CHECK: amdgpu_kernel void @__omp_offloading[[HASH:.*]]_l17()
  // CHECK: amdgpu_kernel void @__omp_offloading[[HASH:.*]]_l19() #4

  #pragma omp target ompx_attribute([[clang::amdgpu_flat_work_group_size(10, 20)]])
  {}
  #pragma omp target teams ompx_attribute(__attribute__((launch_bounds(45, 90))))
  {}
  #pragma omp target teams distribute parallel for simd ompx_attribute([[clang::amdgpu_flat_work_group_size(3, 17)]]) device(3) ompx_attribute(__attribute__((amdgpu_waves_per_eu(3, 7))))
  for (int i = 0; i < 1000; ++i)
  {}
}

// CHECK: attributes #0
// CHECK-SAME: "amdgpu-flat-work-group-size"="10,20"
// CHECK: attributes #4
// CHECK-SAME: "amdgpu-flat-work-group-size"="3,17"
// CHECK-SAME: "amdgpu-waves-per-eu"="3,7"

// CHECK: !{ptr @__omp_offloading[[HASH]]_l17, !"maxntidx", i32 45}
// CHECK: !{ptr @__omp_offloading[[HASH]]_l17, !"minctasm", i32 90}
