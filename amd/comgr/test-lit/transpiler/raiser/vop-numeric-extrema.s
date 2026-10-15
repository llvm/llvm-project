; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %FileCheck %s < %t.ll
; RUN: %opt -passes='default<O2>' -verify-each -S %t.ll -o %t.opt.ll
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.opt.ll -o %t.target.o
; RUN: %FileCheck %s --check-prefixes=CONST,PRESERVE < %t.opt.ll
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM=1 %s -o %t.mode.o
; RUN: %ld.lld -shared %t.mode.o -o %t.mode.hsaco
; RUN: %transpile_cli %t.mode.hsaco --target-isa=gfx942 --emit-ir=extrema_constants | %opt -passes='default<O2>' -S | %FileCheck %s --check-prefixes=CONST,PRESERVE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM=2 %s -o %t.mode.o
; RUN: %ld.lld -shared %t.mode.o -o %t.mode.hsaco
; RUN: %transpile_cli %t.mode.hsaco --target-isa=gfx942 --emit-ir=extrema_constants | %opt -passes='default<O2>' -S | %FileCheck %s --check-prefixes=CONST,FLUSH
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=CLAMP=1 %s -o %t.bad.o
; RUN: %ld.lld -shared %t.bad.o -o %t.bad.hsaco
; RUN: not %transpile_cli %t.bad.hsaco --target-isa=gfx942 --emit-ir=extrema 2>&1 | %FileCheck %s --check-prefix=CLAMP
; CLAMP: unsupported-instruction-form: v_max_num_f32 [VOP3]
; CLAMP-SAME: floating-point output clamp is not supported
.ifndef DENORM
.set DENORM, 3
.endif
.amdhsa_code_object_version 6
.text
.globl extrema
.p2align 8
.type extrema,@function
; CHECK-LABEL: define amdgpu_kernel void @extrema(
extrema:
 s_load_b256 s[4:11], s[0:1], 0
 v_lshlrev_b32 v1, 2, v0
 v_lshlrev_b32 v3, 4, v0
 s_wait_kmcnt 0
 global_load_b32 v2, v1, s[4:5]
 global_load_b32 v8, v1, s[6:7]
 v_mov_b32 v4, 0x12345678
 v_mov_b32 v5, 0x12345678
 v_mov_b32 v6, 0x12345678
 v_mov_b32 v7, 0x12345678
 s_wait_loadcnt 0
 s_and_saveexec_b32 s12, 0x55555555
; CHECK: icmp sgt i32
 v_max_num_f32_e32 v4, v2, v8
; CHECK: icmp slt i32
 v_min_num_f32_e32 v5, v2, v8
; CHECK: icmp sgt i32
.ifdef CLAMP
 v_max_num_f32_e64 v6, -v2, |v8| clamp
.else
 v_max_num_f32_e64 v6, -v2, |v8|
.endif
; CHECK: icmp slt i32
 v_min_num_f32_e64 v7, -v2, |v8|
 s_mov_b32 exec_lo, s12
; CHECK: store <4 x i32> {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b128 v3, v[4:7], s[8:9] offset:16
 s_endpgm
.globl extrema_constants
.p2align 8
.type extrema_constants,@function
; CONST-LABEL: define amdgpu_kernel void @extrema_constants(
extrema_constants:
 s_load_b64 s[4:5], s[0:1], 0
 v_lshlrev_b32 v1, 5, v0
 s_wait_kmcnt 0
 v_mov_b32 v2, 0x80000000
 v_mov_b32 v8, 0
 v_max_num_f32_e32 v4, v2, v8
 v_min_num_f32_e32 v5, v2, v8
 v_mov_b32 v2, 0x7f812345
 v_mov_b32 v8, 1.0
 v_max_num_f32_e32 v6, v2, v8
 v_mov_b32 v8, 0xffcabcde
 v_min_num_f32_e32 v7, v2, v8
; CONST: store <4 x i32> <i32 0, i32 -2147483648, i32 1065353216, i32 2143363909>,
 global_store_b128 v1, v[4:7], s[4:5]
 v_mov_b32 v2, 1
 v_mov_b32 v8, 2
 v_max_num_f32_e32 v4, v2, v8
 v_min_num_f32_e32 v5, v2, v8
 v_max_num_f32_e32 v6, v2, v2
 v_mov_b32 v8, 0x80000001
 v_min_num_f32_e32 v7, v8, v8
; PRESERVE: store i32 2,
; FLUSH: store i32 0,
 global_store_b32 v1, v4, s[4:5] offset:16
; PRESERVE: store i32 1,
; FLUSH: store i32 0,
 global_store_b32 v1, v5, s[4:5] offset:20
; PRESERVE: store i32 1,
; FLUSH: store i32 0,
 global_store_b32 v1, v6, s[4:5] offset:24
; PRESERVE: store i32 -2147483647,
; FLUSH: store i32 -2147483648,
 global_store_b32 v1, v7, s[4:5] offset:28
 s_endpgm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel extrema
 .amdhsa_kernarg_size 32
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 9
 .amdhsa_next_free_sgpr 13
 .amdhsa_float_denorm_mode_32 DENORM
 .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.amdhsa_kernel extrema_constants
 .amdhsa_kernarg_size 8
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 9
 .amdhsa_next_free_sgpr 6
 .amdhsa_float_denorm_mode_32 DENORM
 .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
 - .name: extrema
   .symbol: extrema.kd
   .kernarg_segment_size: 32
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 13
   .vgpr_count: 9
   .wavefront_size: 32
 - .name: extrema_constants
   .symbol: extrema_constants.kd
   .kernarg_segment_size: 8
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 6
   .vgpr_count: 9
   .wavefront_size: 32
...
.end_amdgpu_metadata
