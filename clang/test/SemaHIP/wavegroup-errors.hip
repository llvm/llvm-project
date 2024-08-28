// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn-amd-amdhsa -x hip %s

#define __global__ __attribute__((global))
#define __exp_amd_wavegroup_kernel(num_wavegroups, wave_size, block_x, block_y, block_z) \
    __attribute__((amdgpu_wavegroup_kernel(num_wavegroups, wave_size, block_x, block_y, block_z)))

__exp_amd_wavegroup_kernel( // expected-error {{'amdgpu_wavegroup_kernel' attribute requires an integer constant}}
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 32, 8, "hi")
__global__ void error1() {}

__exp_amd_wavegroup_kernel( // expected-error {{number of wavegroups must be 4}}
    /* num_wavegroups */ 5,
    /* wave size */ 32,
    /* block size */ 32, 8, 1)
__global__ void error2() {}

__exp_amd_wavegroup_kernel( // expected-error {{wave size must be 32}}
    /* num_wavegroups */ 4,
    /* wave size */ 64,
    /* block size */ 32, 8, 1)
__global__ void error3() {}

// expected-error@+2 {{invalid block dimensions}}
// expected-note@+1 {{number of threads must be a multiple of 128}}
__exp_amd_wavegroup_kernel(
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 32, 7, 1)
__global__ void error4() {}

// expected-error@+2 {{invalid block dimensions}}
// expected-note@+1 {{number of threads must be non-zero}}
__exp_amd_wavegroup_kernel(
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 32, 8, 0)
__global__ void error4b() {}

// expected-error@+2 {{invalid block dimensions}}
// expected-note@+1 {{number of threads must be at most 1024}}
__exp_amd_wavegroup_kernel(
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 128, 9, 1)
__global__ void error4c() {}

__attribute__((amdgpu_wavegroup_kernel(1, 2, 3))) // expected-error {{'amdgpu_wavegroup_kernel' attribute requires exactly 5 arguments}}
__global__ void error5() {}

__exp_amd_wavegroup_kernel(
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 32, 8, 1)
void error6() {} // expected-error {{'amdgpu_wavegroup_kernel' attribute only applies to kernel functions}}
