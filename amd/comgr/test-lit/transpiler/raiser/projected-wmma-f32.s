; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 > %t.ll
; RUN: %FileCheck %s --check-prefix=WAVE < %t.ll
; WAVE: ; launch: f32 kind=unchanged
; RUN: %opt -passes='default<O2>' -verify-each %t.ll -o %t.bc
; RUN: %llvm-dis %t.bc -o - | %FileCheck %s --check-prefix=OPT
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.target.o
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=16,1,1 2>&1 | %FileCheck %s --check-prefix=PARTIAL
; PARTIAL: WMMA requires whole source waves
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=32,1,1 > %t.single.ll
; RUN: %FileCheck %s --check-prefix=SINGLE < %t.single.ll
; SINGLE: ; launch: f32 kind=replicated-1D-whole-wave
; SINGLE: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32
; RUN: %opt -passes='default<O2>' -verify-each %t.single.ll -o %t.single.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.single.bc -o %t.single.o
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --launch-workgroup=16,1,1 --launch-grid=16,1,1 2>&1 | %FileCheck %s --check-prefix=LAUNCH
; LAUNCH: replicated dispatch requires whole source waves
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx1250 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=TARGET
; TARGET: WMMA remapping requires wave32 to wave64
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym DENORM=1 %s -o %t.output-flush.o
; RUN: %ld.lld -shared %t.output-flush.o -o %t.output-flush.hsaco
; RUN: not %transpile_cli %t.output-flush.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=DENORM
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym DENORM=0 %s -o %t.flush.o
; RUN: %ld.lld -shared %t.flush.o -o %t.flush.hsaco
; RUN: not %transpile_cli %t.flush.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=DENORM
; DENORM: F32 WMMA requires preserved input and output denormals
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym DENORM=2 %s -o %t.input-flush.o
; RUN: %ld.lld -shared %t.input-flush.o -o %t.input-flush.hsaco
; RUN: not %transpile_cli %t.input-flush.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=DENORM
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym CASE=1 %s -o %t.mod.o
; RUN: %ld.lld -shared %t.mod.o -o %t.mod.hsaco
; RUN: not %transpile_cli %t.mod.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=MOD
; MOD: WMMA source modifiers are not supported
; RUN: not %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym CASE=2 %s -o %t.clamp.o 2>&1 | %FileCheck %s --check-prefix=CLAMP
; CLAMP: error: invalid operand for instruction
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym CASE=3 %s -o %t.exec.o
; RUN: %ld.lld -shared %t.exec.o -o %t.exec.hsaco
; RUN: not %transpile_cli %t.exec.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=EXEC
; EXEC: unproven-kernel-entry-exec: v_wmma_f32_16x16x4_f32
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym CASE=4 %s -o %t.inline.o
; RUN: %ld.lld -shared %t.inline.o -o %t.inline.hsaco
; RUN: not %transpile_cli %t.inline.hsaco --target-isa=gfx942 --emit-ir --allow-replicated-dispatch --specialize-workgroup=64,1,1 2>&1 | %FileCheck %s --check-prefix=INLINE
; INLINE: only a zero immediate WMMA accumulator is supported

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.ifndef CASE
.set CASE, 0
.endif
.ifndef DENORM
.set DENORM, 3
.endif
.text
.globl f32
.p2align 8
.type f32,@function
; OPT-LABEL: define amdgpu_kernel void @f32(
f32:
  s_load_b256 s[4:11], s[0:1], 0
  v_lshlrev_b32 v31, 3, v0
  v_lshlrev_b32 v30, 7, v0
  s_wait_kmcnt 0
  global_load_b64 v[0:1], v31, s[4:5]
  global_load_b64 v[8:9], v31, s[6:7]
.irp base,16,20
  v_mov_b32 v[\base], 1.0
  v_mov_b32 v[\base+1], 2.0
  v_mov_b32 v[\base+2], 0x40400000
  v_mov_b32 v[\base+3], 4.0
.endr
  s_wait_loadcnt 0
.if CASE == 1
  v_wmma_f32_16x16x4_f32 v[16:23], v[0:1], v[8:9], v[16:23] neg_lo:[0,0,1]
.elseif CASE == 2
  v_wmma_f32_16x16x4_f32 v[16:23], v[0:1], v[8:9], v[16:23] clamp
.elseif CASE == 3
  s_mov_b32 exec_lo, 0x55555555
  v_wmma_f32_16x16x4_f32 v[16:23], v[0:1], v[8:9], v[16:23]
.elseif CASE == 4
  v_wmma_f32_16x16x4_f32 v[16:23], v[0:1], v[8:9], 1.0
.else
; OPT-COUNT-2: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float {{%[^,]+}}, float {{%[^,]+}}, <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, i32 0, i32 0, i32 0)
  v_wmma_f32_16x16x4_f32 v[16:23], v[0:1], v[8:9], v[16:23]
.endif
  global_store_b128 v30, v[16:19], s[10:11] offset:16
  global_store_b128 v30, v[20:23], s[10:11] offset:32
.irp base,32,36
  v_mov_b32 v[\base], 0x40a00000
  v_mov_b32 v[\base+1], 0x40c00000
  v_mov_b32 v[\base+2], 0x40e00000
  v_mov_b32 v[\base+3], 0x41000000
.endr
; OPT-COUNT-2: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float {{%[^,]+}}, float {{%[^,]+}}, <4 x float> <float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00>, i32 0, i32 0, i32 0)
  v_wmma_f32_16x16x4_f32 v[16:23], v[0:1], v[8:9], v[32:39]
  global_store_b128 v30, v[16:19], s[10:11] offset:48
  global_store_b128 v30, v[20:23], s[10:11] offset:64
; OPT-COUNT-2: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float {{%[^,]+}}, float {{%[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  v_wmma_f32_16x16x4_f32 v[32:39], v[0:1], v[8:9], 0
  global_store_b128 v30, v[32:35], s[10:11] offset:80
  global_store_b128 v30, v[36:39], s[10:11] offset:96
; OPT-NOT: call <4 x float> @llvm.amdgcn.mfma
; OPT: ret void
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel f32
  .amdhsa_kernarg_size 32
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 40
  .amdhsa_next_free_sgpr 12
  .amdhsa_float_denorm_mode_32 DENORM
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: f32
    .symbol: f32.kd
    .group_segment_fixed_size: 0
    .kernarg_segment_size: 32
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 1024
    .sgpr_count: 12
    .vgpr_count: 40
    .wavefront_size: 32
...
.end_amdgpu_metadata
