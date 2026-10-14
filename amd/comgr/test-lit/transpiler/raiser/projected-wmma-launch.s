; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --launch-workgroup=32,1,1 --launch-grid=64,1,1 | %FileCheck %s
; CHECK: ; launch: f16 kind=unchanged
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --launch-workgroup=16,1,1 --launch-grid=16,1,1 2>&1 | %FileCheck %s --check-prefix=PARTIAL-WAVE
; PARTIAL-WAVE: unsupported-launch in kernel 'f16' :: launch requires whole source waves
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --launch-workgroup=32,1,1 --launch-grid=48,1,1 2>&1 | %FileCheck %s --check-prefix=PARTIAL-GROUP
; PARTIAL-GROUP: unsupported-launch in kernel 'f16' :: launch requires complete workgroups

; No EXEC transition or workgroup specialization: validate unchanged geometry.
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text

.globl f16
.p2align 8
.type f16,@function
; CHECK-LABEL: define amdgpu_kernel void @f16(
f16:
  s_load_b256 s[4:11], s[0:1], 0
  v_lshlrev_b32 v30, 5, v0
  s_wait_kmcnt 0
  global_load_b128 v[0:3], v30, s[4:5]
  global_load_b128 v[4:7], v30, s[4:5] offset:16
  global_load_b128 v[8:11], v30, s[6:7]
  global_load_b128 v[12:15], v30, s[6:7] offset:16
  global_load_b128 v[16:19], v30, s[8:9]
  global_load_b128 v[20:23], v30, s[8:9] offset:16
  s_wait_loadcnt 0
; CHECK: @llvm.amdgcn.mfma.f32.16x16x16f16
  v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
  global_store_b128 v30, v[16:19], s[10:11]
  global_store_b128 v30, v[20:23], s[10:11] offset:16
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel f16
  .amdhsa_kernarg_size 32
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: f16
    .symbol: f16.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 32
    .wavefront_size: 32
...
.end_amdgpu_metadata
