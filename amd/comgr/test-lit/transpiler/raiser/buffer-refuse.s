; REQUIRES: comgr-has-transpiler
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=0 -filetype=obj %s -o %t.0.o
; RUN: %ld.lld -shared %t.0.o -o %t.0.hsaco
; RUN: not %transpile_cli %t.0.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=STRIDE
; STRIDE: in kernel 'buffer_refuse'
; STRIDE-SAME: buffer stride and stride scale must be provably zero
; RUN: not %transpile_cli %t.0.hsaco --target-isa=gfx1100 --emit-ir 2>&1 | %FileCheck %s --check-prefix=TARGET
; TARGET: in kernel 'buffer_refuse'
; TARGET-SAME: buffer target memory behavior is not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=1 -filetype=obj %s -o %t.1.o
; RUN: %ld.lld -shared %t.1.o -o %t.1.hsaco
; RUN: not %transpile_cli %t.1.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=SCALE
; SCALE: in kernel 'buffer_refuse'
; SCALE-SAME: buffer stride and stride scale must be provably zero

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=2 -filetype=obj %s -o %t.2.o
; RUN: %ld.lld -shared %t.2.o -o %t.2.hsaco
; RUN: not %transpile_cli %t.2.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=SWIZZLE
; SWIZZLE: in kernel 'buffer_refuse'
; SWIZZLE-SAME: swizzled buffer descriptors are not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=3 -filetype=obj %s -o %t.3.o
; RUN: %ld.lld -shared %t.3.o -o %t.3.hsaco
; RUN: not %transpile_cli %t.3.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=STRUCTURED
; STRUCTURED: in kernel 'buffer_refuse'
; STRUCTURED-SAME: structured buffer bounds are not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=4 -filetype=obj %s -o %t.4.o
; RUN: %ld.lld -shared %t.4.o -o %t.4.hsaco
; RUN: not %transpile_cli %t.4.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=TYPE
; TYPE: in kernel 'buffer_refuse'
; TYPE-SAME: buffer descriptor type must be provably zero

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=5 -filetype=obj %s -o %t.5.o
; RUN: %ld.lld -shared %t.5.o -o %t.5.hsaco
; RUN: not %transpile_cli %t.5.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=RESERVED
; RESERVED: in kernel 'buffer_refuse'
; RESERVED-SAME: buffer descriptor reserved bits must be provably zero

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=6 -filetype=obj %s -o %t.6.o
; RUN: %ld.lld -shared %t.6.o -o %t.6.hsaco
; RUN: not %transpile_cli %t.6.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=IDXEN
; IDXEN: in kernel 'buffer_refuse'
; IDXEN-SAME: unsupported buffer opcode or addressing form

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=7 -filetype=obj %s -o %t.7.o
; RUN: %ld.lld -shared %t.7.o -o %t.7.hsaco
; RUN: not %transpile_cli %t.7.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=BOTHEN
; BOTHEN: in kernel 'buffer_refuse'
; BOTHEN-SAME: unsupported buffer opcode or addressing form

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=8 -filetype=obj %s -o %t.8.o
; RUN: %ld.lld -shared %t.8.o -o %t.8.hsaco
; RUN: not %transpile_cli %t.8.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=SCOPE
; SCOPE: in kernel 'buffer_refuse'
; SCOPE-SAME: non-default buffer cache policy is not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=9 -filetype=obj %s -o %t.9.o
; RUN: %ld.lld -shared %t.9.o -o %t.9.hsaco
; RUN: not %transpile_cli %t.9.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=TEMPORAL
; TEMPORAL: in kernel 'buffer_refuse'
; TEMPORAL-SAME: non-default buffer cache policy is not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=11 -filetype=obj %s -o %t.11.o
; RUN: %ld.lld -shared %t.11.o -o %t.11.hsaco
; RUN: not %transpile_cli %t.11.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=ATOMIC
; ATOMIC: in kernel 'buffer_refuse'
; ATOMIC-SAME: unsupported buffer opcode or addressing form

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=12 -filetype=obj %s -o %t.12.o
; RUN: %ld.lld -shared %t.12.o -o %t.12.hsaco
; RUN: not %transpile_cli %t.12.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=SCALAR
; SCALAR: in kernel 'buffer_refuse'
; SCALAR-SAME: VBUFFER scalar offset must be an SGPR, M0, or null

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=13 -filetype=obj %s -o %t.13.o
; RUN: %ld.lld -shared %t.13.o -o %t.13.hsaco
; RUN: not %transpile_cli %t.13.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=DYNAMIC
; DYNAMIC: in kernel 'buffer_refuse'
; DYNAMIC-SAME: buffer descriptor type must be provably zero

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=14 -filetype=obj %s -o %t.14.o
; RUN: %ld.lld -shared %t.14.o -o %t.14.hsaco
; RUN: not %transpile_cli %t.14.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=JOIN
; JOIN: in kernel 'buffer_refuse'
; JOIN-SAME: buffer descriptor type must be provably zero

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=15 -filetype=obj %s -o %t.15.o
; RUN: %ld.lld -shared %t.15.o -o %t.15.hsaco
; RUN: not %transpile_cli %t.15.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=LOOP
; LOOP: in kernel 'buffer_refuse'
; LOOP-SAME: swizzled buffer descriptors are not modeled

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -mattr=-sramecc -defsym=CASE=16 -filetype=obj %s -o %t.16.o
; RUN: %ld.lld -shared %t.16.o -o %t.16.hsaco
; RUN: not %transpile_cli %t.16.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=ECC
; ECC: in kernel 'buffer_refuse'
; ECC-SAME: gfx1250 source does not support disabling SRAM ECC

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -defsym=CASE=17 -filetype=obj %s -o %t.17.o
; RUN: %ld.lld -shared %t.17.o -o %t.17.hsaco
; RUN: not %transpile_cli %t.17.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefix=TFE
; TFE: in kernel 'buffer_refuse'
; TFE-SAME: unsupported buffer opcode or addressing form

.amdhsa_code_object_version 6
.text
.globl buffer_refuse
.p2align 8
.type buffer_refuse,@function
buffer_refuse:
  s_load_b128 s[4:7], s[0:1], 0
  s_wait_kmcnt 0
  s_mov_b32 s7, 0
.if CASE == 0
  s_mov_b32 s7, 0x1000
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 1
  s_mov_b32 s7, 0x04000000
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 2
  s_mov_b32 s7, 0x10000000
  buffer_store_b32 v1, off, s[4:7], null
.endif
.if CASE == 3
  s_mov_b32 s7, 0x20000000
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 4
  s_mov_b32 s7, 0x40000000
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 5
  s_mov_b32 s7, 0x40
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 6
  buffer_load_b32 v1, v0, s[4:7], null idxen
.endif
.if CASE == 7
  buffer_store_b32 v1, v[2:3], s[4:7], null idxen offen
.endif
.if CASE == 8
  buffer_store_b32 v1, v0, s[4:7], null offen offset:0 scope:SCOPE_DEV
.endif
.if CASE == 9
  buffer_load_b32 v1, v0, s[4:7], null offen th:TH_LOAD_NT
.endif
.if CASE == 11
  buffer_atomic_add_u32 v1, v0, s[4:7], null offen
.endif
.if CASE == 12
  buffer_load_b32 v1, off, s[4:7], exec_lo
.endif
.if CASE == 13
  s_load_b32 s7, s[0:1], 12
  s_wait_kmcnt 0
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 14
  s_cmp_eq_u32 s4, 0
  s_cbranch_scc1 .Ljoin
  s_mov_b32 s7, 0x40000000
.Ljoin:
  buffer_load_b32 v1, off, s[4:7], null
.endif
.if CASE == 15
  .Lloop:
  buffer_load_b32 v1, off, s[4:7], null
  s_mov_b32 s7, 0x10000000
  s_branch .Lloop
.endif
.if CASE == 16
  v_mov_b32 v1, 0x12345678
  buffer_load_d16_u8 v1, off, s[4:7], null
.endif
.if CASE == 17
  buffer_load_b32 v[2:3], off, s[4:7], null tfe
.endif
  s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel buffer_refuse
  .amdhsa_kernarg_size 16
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: buffer_refuse
    .symbol: buffer_refuse.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 32
    .sgpr_count: 8
    .vgpr_count: 4
    .wavefront_size: 32
...
.end_amdgpu_metadata
