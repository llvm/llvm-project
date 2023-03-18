// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx908 -Rpass-analysis=kernel-resource-usage -S -O0 -verify %s -o /dev/null

// expected-remark@+9 {{Function Name: foo}}
// expected-remark@+8 {{    SGPRs: 13}}
// expected-remark@+7 {{    VGPRs: 10}}
// expected-remark@+6 {{    AGPRs: 12}}
// expected-remark@+5 {{    ScratchSize [bytes/lane]: 0}}
// expected-remark@+4 {{    Occupancy [waves/SIMD]: 10}}
// expected-remark@+3 {{    SGPRs Spill: 0}}
// expected-remark@+2 {{    VGPRs Spill: 0}}
// expected-remark@+1 {{    LDS Size [bytes/block]: 0}}
__kernel void foo() {
  __asm volatile ("; clobber s8" :::"s8");
  __asm volatile ("; clobber v9" :::"v9");
  __asm volatile ("; clobber a11" :::"a11");
}
