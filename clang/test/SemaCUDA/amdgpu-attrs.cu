// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"


// expected-error@+2 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64)))
__global__ void flat_work_group_size_32_64() {}

// expected-error@+2 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2)))
__global__ void waves_per_eu_2() {}

// expected-error@+2 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2, 4)))
__global__ void waves_per_eu_2_4() {}

// expected-error@+2 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_sgpr(32)))
__global__ void num_sgpr_32() {}

// expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_vgpr(64)))
__global__ void num_vgpr_64() {}


// expected-error@+3 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2() {}

// expected-error@+3 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4() {}

// expected-error@+3 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_num_sgpr_32() {}

// expected-error@+3 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_num_vgpr_64() {}

// expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
__global__ void waves_per_eu_2_num_sgpr_32() {}

// expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
__global__ void waves_per_eu_2_num_vgpr_64() {}

// expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
__global__ void waves_per_eu_2_4_num_sgpr_32() {}

// expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
__global__ void waves_per_eu_2_4_num_vgpr_64() {}

// expected-error@+3 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void num_sgpr_32_num_vgpr_64() {}


// expected-error@+4 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32() {}

// expected-error@+4 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_vgpr_64() {}

// expected-error@+4 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32() {}

// expected-error@+4 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+3 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_vgpr_64() {}


// expected-error@+5 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+4 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+3 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32_num_vgpr_64() {}

// expected-error@+5 {{'amdgpu_flat_work_group_size' attribute only applies to kernel functions}}
// fixme-expected-error@+4 {{'amdgpu_waves_per_eu' attribute only applies to kernel functions}}
// fixme-expected-error@+3 {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}
// fixme-expected-error@+2 {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32_num_vgpr_64() {}
