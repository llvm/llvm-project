; REQUIRES: comgr-has-transpiler, comgr-has-llc

; Redundant writes preserve all four denormal modes, independently for F32 and
; F16/F64. Numeric maximum preserves its selected subnormal unless inputs flush.
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM32=0 --defsym=DENORM64=3 --defsym=MODE=12 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir | %opt -passes='default<O2>' -verify-each -S | %FileCheck %s --check-prefixes=CHECK,FLUSH
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM32=1 --defsym=DENORM64=2 --defsym=MODE=9 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir | %opt -passes='default<O2>' -verify-each -S | %FileCheck %s --check-prefixes=CHECK,PRESERVE
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM32=2 --defsym=DENORM64=1 --defsym=MODE=6 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir | %opt -passes='default<O2>' -verify-each -S | %FileCheck %s --check-prefixes=CHECK,FLUSH
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM32=3 --defsym=DENORM64=0 --defsym=MODE=3 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir | %opt -passes='default<O2>' -verify-each -S | %FileCheck %s --check-prefixes=CHECK,PRESERVE

; Changing either precision's mode must be refused, including enabling denormals
; when the descriptor flushes them.
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM32=0 --defsym=DENORM64=3 --defsym=MODE=15 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefixes=REFUSE,CHANGE32
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=DENORM32=3 --defsym=DENORM64=0 --defsym=MODE=15 %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco --target-isa=gfx942 --emit-ir 2>&1 | %FileCheck %s --check-prefixes=REFUSE,CHANGE64
; REFUSE: unsupported-instruction-form: s_denorm_mode [SOPP]
; CHANGE32-SAME: selects denormal mode 15 rather than kernel descriptor mode 12
; CHANGE64-SAME: selects denormal mode 15 rather than kernel descriptor mode 3

.amdhsa_code_object_version 6
.text
.globl denorm_mode_max
.p2align 8
.type denorm_mode_max,@function
; CHECK-LABEL: define amdgpu_kernel void @denorm_mode_max(
denorm_mode_max:
 s_load_b64 s[4:5], s[0:1], 0
 v_lshlrev_b32 v1, 2, v0
 s_wait_kmcnt 0
 s_denorm_mode MODE
 v_mov_b32 v2, 1
 v_mov_b32 v3, 2
 v_max_num_f32_e32 v4, v2, v3
; FLUSH: store i32 0, ptr addrspace(1)
; PRESERVE: store i32 2, ptr addrspace(1)
 global_store_b32 v1, v4, s[4:5]
 s_endpgm

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel denorm_mode_max
 .amdhsa_kernarg_size 8
 .amdhsa_user_sgpr_kernarg_segment_ptr 1
 .amdhsa_next_free_vgpr 5
 .amdhsa_next_free_sgpr 6
 .amdhsa_float_denorm_mode_32 DENORM32
 .amdhsa_float_denorm_mode_16_64 DENORM64
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
 - .name: denorm_mode_max
   .symbol: denorm_mode_max.kd
   .kernarg_segment_size: 8
   .kernarg_segment_align: 8
   .group_segment_fixed_size: 0
   .private_segment_fixed_size: 0
   .max_flat_workgroup_size: 32
   .sgpr_count: 6
   .vgpr_count: 5
   .wavefront_size: 32
...
.end_amdgpu_metadata
