; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %FileCheck %s < %t.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.ll -o %t.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.bc -o %t.target.o
.amdhsa_code_object_version 6
.text
.globl conversions
.p2align 8
.type conversions,@function
; CHECK-LABEL: define amdgpu_kernel void @conversions(
conversions:
 s_load_b128 s[4:7], s[0:1], 0
 v_lshlrev_b32 v1, 2, v0
 v_lshlrev_b32 v3, 4, v0
 s_wait_kmcnt 0
 global_load_b32 v2, v1, s[4:5]
 v_mov_b32 v4, 0x12345678
 v_mov_b32 v5, 0x12345678
 v_mov_b32 v6, 0x12345678
 v_mov_b32 v7, 0x12345678
 s_wait_loadcnt 0
 s_and_saveexec_b32 s8, 0x55555555
; CHECK: fptrunc float
; CHECK: or i16 {{%.+}}, 512
; CHECK: and i32 305419896, -65536
; CHECK: [[LO:%.+]] = or i32
; CHECK: phi i32 [ [[LO]], {{%.+}} ], [ 305419896, {{%.+}} ]
 v_cvt_f16_f32_e32 v4.l, v2
; CHECK: fptrunc float
; CHECK: or i16 {{%.+}}, 512
; CHECK: and i32 305419896, 65535
; CHECK: [[HI:%.+]] = or i32
; CHECK: phi i32 [ [[HI]], {{%.+}} ], [ 305419896, {{%.+}} ]
 v_cvt_f16_f32_e32 v5.h, v2
; CHECK: fptrunc float
; CHECK: or i16 {{%.+}}, 512
; CHECK: and i32 305419896, -65536
; CHECK: [[MODLO:%.+]] = or i32
; CHECK: phi i32 [ [[MODLO]], {{%.+}} ], [ 305419896, {{%.+}} ]
 v_cvt_f16_f32_e64 v6.l, -|v2|
; CHECK: fptrunc float
; CHECK: or i16 {{%.+}}, 512
; CHECK: and i32 305419896, 65535
; CHECK: [[MODHI:%.+]] = or i32
; CHECK: phi i32 [ [[MODHI]], {{%.+}} ], [ 305419896, {{%.+}} ]
 v_cvt_f16_f32_e64 v7.h, |v2|
 s_mov_b32 exec_lo, s8
; CHECK: store <4 x i32> {{.+}}, ptr addrspace(1) {{.+}}
 global_store_b128 v3, v[4:7], s[6:7] offset:16
 s_endpgm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel conversions
 .amdhsa_kernarg_size 16
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 8
 .amdhsa_next_free_sgpr 9
 .amdhsa_float_denorm_mode_32 3
 .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
 - .name: conversions
   .symbol: conversions.kd
   .kernarg_segment_size: 16
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 9
   .vgpr_count: 8
   .wavefront_size: 32
...
.end_amdgpu_metadata
