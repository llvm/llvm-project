// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx908 -Rpass-analysis=kernel-resource-usage -S -O0 -verify %s -o /dev/null

// expected-remark@+10 {{Function Name: foo}}
// expected-remark@+9 {{    SGPRs: foo.num_sgpr+(extrasgprs(foo.uses_vcc, foo.uses_flat_scratch, 1))}}
// expected-remark@+8 {{    VGPRs: foo.num_vgpr}}
// expected-remark@+7 {{    AGPRs: foo.num_agpr}}
// expected-remark@+6 {{    ScratchSize [bytes/lane]: foo.private_seg_size}}
// expected-remark@+5 {{    Dynamic Stack: False}}
// expected-remark@+4 {{    Occupancy [waves/SIMD]: occupancy(10, 4, 256, 8, 10, max(foo.num_sgpr+(extrasgprs(foo.uses_vcc, foo.uses_flat_scratch, 1)), 1, 0), max(totalnumvgprs(foo.num_agpr, foo.num_vgpr), 1, 0))}}
// expected-remark@+3 {{    SGPRs Spill: 0}}
// expected-remark@+2 {{    VGPRs Spill: 0}}
// expected-remark@+1 {{    LDS Size [bytes/block]: 0}}
__kernel void foo() {
  __asm volatile ("; clobber s8" :::"s8");
  __asm volatile ("; clobber v9" :::"v9");
  __asm volatile ("; clobber a11" :::"a11");
}
