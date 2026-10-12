; REQUIRES: comgr-has-transpiler, comgr-has-llc
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --dump-decoded | %FileCheck %s --check-prefix=DECODE
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir > %t.ll
; RUN: %FileCheck %s --check-prefix=IR < %t.ll
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa \
; RUN:   -filetype=obj %t.ll -o %t.gfx942.o
; RUN: %transpile_cli %t.hsaco \
; RUN:   --target-isa=gfx1250 --emit-ir | %llc -mtriple=amdgpu12.50-amd-amdhsa \
; RUN:     -filetype=obj -o %t.gfx1250.o
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=EXTENT=3 -defsym=INITIAL=0x12345678 -defsym=MASKED=1 -filetype=obj %S/buffer-d16.s -o %t.d16.o
; RUN: %ld.lld -shared %t.d16.o -o %t.d16.hsaco
; RUN: %transpile_cli %t.d16.hsaco --target-isa=gfx942 --emit-ir | %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj -o %t.d16.gfx942.o
; RUN: %ld.lld -shared %t.d16.gfx942.o -o %t.d16.gfx942.hsaco

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.globl buffer_variants
.p2align 8
.type buffer_variants,@function
; IR-LABEL: define amdgpu_kernel void @buffer_variants(
buffer_variants:
  s_load_b128 s[4:7], s[0:1], 0
  s_wait_kmcnt 0
  s_and_b32 s7, s7, 63
  s_mov_b32 s8, 4
  s_mov_b32 m0, 8
  v_mov_b32 v1, 0x12345678
  v_mov_b32 v2, 0
  v_mov_b32 v3, 0
  v_mov_b32 v4, 0
  v_mov_b32 v5, 0
  v_mov_b32 v6, 0
  v_mov_b32 v7, 0

; DECODE: BUFFER_LOAD_U8  buffer_load_u8 v1, off
; IR: add i64 {{.+}}, 8388607
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i8 {{.+}} to i32
  buffer_load_u8 v1, off, s[4:7], s8 offset:8388607

; DECODE: BUFFER_LOAD_U8  buffer_load_u8 v1, v0
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i8 {{.+}} to i32
  buffer_load_u8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_I8  buffer_load_i8 v1, off
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i8 {{.+}} to i32
  buffer_load_i8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_I8  buffer_load_i8 v1, v0
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i8 {{.+}} to i32
  buffer_load_i8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_U16  buffer_load_u16 v1, off
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i16 {{.+}} to i32
  buffer_load_u16 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_U16  buffer_load_u16 v1, v0
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i16 {{.+}} to i32
  buffer_load_u16 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_I16  buffer_load_i16 v1, off
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i16 {{.+}} to i32
  buffer_load_i16 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_I16  buffer_load_i16 v1, v0
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i16 {{.+}} to i32
  buffer_load_i16 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_B32  buffer_load_b32 v1, off
; IR: load i32, ptr addrspace(1) {{.+}}, align 1
; IR: phi i32 [ 0, {{.+}} ], [ {{.+}} ]
  buffer_load_b32 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_B32  buffer_load_b32 v1, v0
; IR: load i32, ptr addrspace(1) {{.+}}, align 1
; IR: phi i32 [ 0, {{.+}} ], [ {{.+}} ]
  buffer_load_b32 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_B64  buffer_load_b64 v[2:3], off
; IR-COUNT-2: load i32, ptr addrspace(1) {{.+}}, align 1
  buffer_load_b64 v[2:3], off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_B64  buffer_load_b64 v[2:3], v0
; IR-COUNT-2: load i32, ptr addrspace(1) {{.+}}, align 1
  buffer_load_b64 v[2:3], v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_B96  buffer_load_b96 v[2:4], off
; IR-COUNT-3: load i32, ptr addrspace(1) {{.+}}, align 1
  buffer_load_b96 v[2:4], off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_B96  buffer_load_b96 v[2:4], v0
; IR-COUNT-3: load i32, ptr addrspace(1) {{.+}}, align 1
  buffer_load_b96 v[2:4], v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_B128  buffer_load_b128 v[4:7], off
; IR-COUNT-4: load i32, ptr addrspace(1) {{.+}}, align 1
  buffer_load_b128 v[4:7], off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_B128  buffer_load_b128 v[4:7], v0
; IR-COUNT-4: load i32, ptr addrspace(1) {{.+}}, align 1
  buffer_load_b128 v[4:7], v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_D16_U8  buffer_load_d16_u8 v1, off
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_u8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_D16_U8  buffer_load_d16_u8 v1, v0
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_u8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_D16_I8  buffer_load_d16_i8 v1, off
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_i8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_D16_I8  buffer_load_d16_i8 v1, v0
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_i8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_D16_B16  buffer_load_d16_b16 v1, off
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_b16 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_D16_B16  buffer_load_d16_b16 v1, v0
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_b16 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_D16_HI_U8  buffer_load_d16_hi_u8 v1, off
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_hi_u8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_D16_HI_U8  buffer_load_d16_hi_u8 v1, v0
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: zext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_hi_u8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_D16_HI_I8  buffer_load_d16_hi_i8 v1, off
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_hi_i8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_D16_HI_I8  buffer_load_d16_hi_i8 v1, v0
; IR: load i8, ptr addrspace(1) {{.+}}, align 1
; IR: phi i8 [ 0, {{.+}} ], [ {{.+}} ]
; IR: sext i8 {{.+}} to i16
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_hi_i8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_LOAD_D16_HI_B16  buffer_load_d16_hi_b16 v1, off
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_hi_b16 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_LOAD_D16_HI_B16  buffer_load_d16_hi_b16 v1, v0
; IR: load i16, ptr addrspace(1) {{.+}}, align 1
; IR: phi i16 [ 0, {{.+}} ], [ {{.+}} ]
; IR: or i32 {{.+}}, {{.+}}
  buffer_load_d16_hi_b16 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_B8  buffer_store_b8 v1, off
; IR: store i8 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_B8  buffer_store_b8 v1, v0
; IR: store i8 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_B16  buffer_store_b16 v1, off
; IR: store i16 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b16 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_B16  buffer_store_b16 v1, v0
; IR: store i16 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b16 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_B32  buffer_store_b32 v1, off
; IR: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b32 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_B32  buffer_store_b32 v1, v0
; IR: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b32 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_B64  buffer_store_b64 v[2:3], off
; IR-COUNT-2: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b64 v[2:3], off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_B64  buffer_store_b64 v[2:3], v0
; IR-COUNT-2: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b64 v[2:3], v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_B96  buffer_store_b96 v[2:4], off
; IR-COUNT-3: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b96 v[2:4], off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_B96  buffer_store_b96 v[2:4], v0
; IR-COUNT-3: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b96 v[2:4], v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_B128  buffer_store_b128 v[4:7], off
; IR-COUNT-4: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b128 v[4:7], off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_B128  buffer_store_b128 v[4:7], v0
; IR-COUNT-4: store i32 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_b128 v[4:7], v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_D16_HI_B8  buffer_store_d16_hi_b8 v1, off
; IR: store i8 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_d16_hi_b8 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_D16_HI_B8  buffer_store_d16_hi_b8 v1, v0
; IR: store i8 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_d16_hi_b8 v1, v0, s[4:7], m0 offen offset:16

; DECODE: BUFFER_STORE_D16_HI_B16  buffer_store_d16_hi_b16 v1, off
; IR: store i16 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_d16_hi_b16 v1, off, s[4:7], s8 offset:16

; DECODE: BUFFER_STORE_D16_HI_B16  buffer_store_d16_hi_b16 v1, v0
; IR: store i16 {{.+}}, ptr addrspace(1) {{.+}}, align 1
  buffer_store_d16_hi_b16 v1, v0, s[4:7], m0 offen offset:16
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel buffer_variants
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 9
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_variants
    .symbol: buffer_variants.kd
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
