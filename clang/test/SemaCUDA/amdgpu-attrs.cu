// RUN: %clang_cc1 -fsyntax-only -verify %s
#include "Inputs/cuda.h"


__attribute__((amdgpu_flat_work_group_size(32, 64)))
__global__ void flat_work_group_size_32_64() {}

__attribute__((amdgpu_waves_per_eu(2)))
__global__ void waves_per_eu_2() {}

__attribute__((amdgpu_waves_per_eu(2, 4)))
__global__ void waves_per_eu_2_4() {}

__attribute__((amdgpu_num_sgpr(32)))
__global__ void num_sgpr_32() {}

__attribute__((amdgpu_num_vgpr(64)))
__global__ void num_vgpr_64() {}


__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_num_sgpr_32() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_num_vgpr_64() {}

__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
__global__ void waves_per_eu_2_num_sgpr_32() {}

__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
__global__ void waves_per_eu_2_num_vgpr_64() {}

__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
__global__ void waves_per_eu_2_4_num_sgpr_32() {}

__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
__global__ void waves_per_eu_2_4_num_vgpr_64() {}

__attribute__((amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void num_sgpr_32_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32_num_vgpr_64() {}

// expected-error@+2{{attribute 'reqd_work_group_size' can only be applied to an OpenCL kernel function}}
__attribute__((reqd_work_group_size(32, 64, 64)))
__global__ void reqd_work_group_size_32_64_64() {}

// expected-error@+2{{attribute 'work_group_size_hint' can only be applied to an OpenCL kernel function}}
__attribute__((work_group_size_hint(2, 2, 2)))
__global__ void work_group_size_hint_2_2_2() {}

// expected-error@+2{{attribute 'vec_type_hint' can only be applied to an OpenCL kernel function}}
__attribute__((vec_type_hint(int)))
__global__ void vec_type_hint_int() {}

// expected-error@+2{{attribute 'intel_reqd_sub_group_size' can only be applied to an OpenCL kernel function}}
__attribute__((intel_reqd_sub_group_size(64)))
__global__ void intel_reqd_sub_group_size_64() {}
