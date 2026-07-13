// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=AMD
// RUN: %clang_cc1 -target-cpu gfx900 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=AMD
// RUN: %clang_cc1 -target-cpu gfx900 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -dwarf-version=5 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=AMD
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64 -fopenmp-targets=nvptx64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=NVIDIA
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple nvptx64 -fopenmp-targets=nvptx64 -emit-llvm %s -fopenmp-is-target-device -dwarf-version=5 -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=NVIDIA
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple spirv64 -fopenmp-targets=spirv64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple spirv64 -fopenmp-targets=spirv64 -emit-llvm %s -fopenmp-is-target-device -dwarf-version=5 -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix=SPIRV
// expected-no-diagnostics


// Check that the target attributes are set on the generated kernel
void func() {
  // AMD: amdgpu_kernel void @__omp_offloading[[HASH:.*]]_l28(ptr {{[^,]+}}) #0
  // AMD: amdgpu_kernel void @__omp_offloading[[HASH:.*]]_l30(ptr {{[^,]+}})
  // AMD: amdgpu_kernel void @__omp_offloading[[HASH:.*]]_l32(ptr {{[^,]+}}) #4

  // NVIDIA: ptx_kernel void @__omp_offloading[[HASH:.*]]_l28(ptr {{[^,]+}}) #[[ATTR0:[0-9]+]]
  // NVIDIA: ptx_kernel void @__omp_offloading[[HASH:.*]]_l30(ptr {{[^,]+}}) #[[ATTR1:[0-9]+]]
  // NVIDIA: ptx_kernel void @__omp_offloading[[HASH:.*]]_l32(ptr {{[^,]+}}) #[[ATTR2:[0-9]+]]

  // SPIRV: spir_kernel void @__omp_offloading[[HASH:.*]]_l28(ptr {{[^,]+}}) #0
  // SPIRV: spir_kernel void @__omp_offloading[[HASH:.*]]_l30(ptr {{[^,]+}})
  // SPIRV: spir_kernel void @__omp_offloading[[HASH:.*]]_l32(ptr {{[^,]+}}) #4

  #pragma omp target ompx_attribute([[clang::amdgpu_flat_work_group_size(10, 20)]])
  {}
  #pragma omp target teams ompx_attribute(__attribute__((launch_bounds(45, 90))))
  {}
  #pragma omp target teams distribute parallel for simd ompx_attribute([[clang::amdgpu_flat_work_group_size(3, 17)]]) device(3) ompx_attribute(__attribute__((amdgpu_waves_per_eu(3, 7))))
  for (int i = 0; i < 1000; ++i)
  {}
}

// SPIRV: attributes #0
// SPIRV-SAME: "nvvm.maxntid"="20"
// SPIRV-SAME: "omp_target_thread_limit"="20" 
// SPIRV: attributes #4
// SPIRV-SAME: "amdgpu-waves-per-eu"="3,7"
// SPIRV-SAME: "nvvm.maxntid"="17"
// SPIRV-SAME: "omp_target_thread_limit"="17"

// AMD: attributes #0
// AMD-SAME: "amdgpu-flat-work-group-size"="10,20"
// AMD-SAME: "omp_target_thread_limit"="20"
// AMD: "omp_target_thread_limit"="45"
// AMD: attributes #4
// AMD-SAME: "amdgpu-flat-work-group-size"="3,17"
// AMD-SAME: "amdgpu-waves-per-eu"="3,7"
// AMD-SAME: "omp_target_thread_limit"="17"

// It is unclear if we should use the AMD annotations for other targets, we do for now.
// NVIDIA: attributes #[[ATTR0]]
// NVIDIA-SAME: "nvvm.maxntid"="20"
// NVIDIA-SAME: "omp_target_thread_limit"="20"
// NVIDIA: attributes #[[ATTR1]]
// NVIDIA-SAME: "nvvm.maxntid"="45"
// NVIDIA-SAME: "omp_target_thread_limit"="45"
// NVIDIA: attributes #[[ATTR2]]
// NVIDIA-SAME: "nvvm.maxntid"="17"
// NVIDIA-SAME: "omp_target_thread_limit"="17"
