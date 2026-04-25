// RUN: %clang_cc1 -triple amdgcn-- -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple amdgcn-- -verify=silenced -Wno-deprecated-declarations -fsyntax-only %s

// silenced-no-diagnostics

__attribute__((amdgpu_num_sgpr(32))) // expected-warning {{the 'amdgpu_num_sgpr' attribute is deprecated; use 'amdgpu_waves_per_eu' instead}}
kernel void kernel_num_sgpr_32() {}

__attribute__((amdgpu_num_vgpr(64))) // expected-warning {{the 'amdgpu_num_vgpr' attribute is deprecated; use 'amdgpu_waves_per_eu' instead}}
kernel void kernel_num_vgpr_64() {}

__attribute__((amdgpu_num_sgpr(0))) // expected-warning {{the 'amdgpu_num_sgpr' attribute is deprecated; use 'amdgpu_waves_per_eu' instead}}
kernel void kernel_num_sgpr_0() {}

__attribute__((amdgpu_num_vgpr(0))) // expected-warning {{the 'amdgpu_num_vgpr' attribute is deprecated; use 'amdgpu_waves_per_eu' instead}}
kernel void kernel_num_vgpr_0() {}
