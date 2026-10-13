; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=FLOW=1 %s -o %t.FLOW.o
; RUN: %ld.lld -shared %t.FLOW.o -o %t.FLOW.hsaco
; RUN: %transpile_cli %t.FLOW.hsaco --target-isa=gfx942 --emit-ir > %t.FLOW.ll
; RUN: %FileCheck %s --check-prefix=VALUE < %t.FLOW.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.FLOW.ll -o %t.FLOW.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.FLOW.bc -o \
; RUN:   %t.FLOW.target.o
; VALUE: add i32 42,

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=JOIN=1 %s -o %t.JOIN.o
; RUN: %ld.lld -shared %t.JOIN.o -o %t.JOIN.hsaco
; RUN: %transpile_cli %t.JOIN.hsaco --target-isa=gfx942 --emit-ir > %t.JOIN.ll
; RUN: %FileCheck %s --check-prefix=VALUE < %t.JOIN.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.JOIN.ll -o %t.JOIN.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.JOIN.bc -o \
; RUN:   %t.JOIN.target.o

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=TRUE32=1 %s -o %t.TRUE32.o
; RUN: %ld.lld -shared %t.TRUE32.o -o %t.TRUE32.hsaco
; RUN: %transpile_cli %t.TRUE32.hsaco --target-isa=gfx942 --emit-ir > \
; RUN:   %t.TRUE32.ll
; RUN: %FileCheck %s --check-prefix=VALUE < %t.TRUE32.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.TRUE32.ll -o %t.TRUE32.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.TRUE32.bc -o \
; RUN:   %t.TRUE32.target.o

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=TRUE64=1 %s -o %t.TRUE64.o
; RUN: %ld.lld -shared %t.TRUE64.o -o %t.TRUE64.hsaco
; RUN: %transpile_cli %t.TRUE64.hsaco --target-isa=gfx942 --emit-ir > \
; RUN:   %t.TRUE64.ll
; RUN: %FileCheck %s --check-prefix=VALUE < %t.TRUE64.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.TRUE64.ll -o %t.TRUE64.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.TRUE64.bc -o \
; RUN:   %t.TRUE64.target.o

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=FALSE_SOURCE=1 %s -o %t.FALSE_SOURCE.o
; RUN: %ld.lld -shared %t.FALSE_SOURCE.o -o %t.FALSE_SOURCE.hsaco
; RUN: %transpile_cli %t.FALSE_SOURCE.hsaco --target-isa=gfx942 --emit-ir > \
; RUN:   %t.FALSE_SOURCE.ll
; RUN: %FileCheck %s --check-prefix=VALUE < %t.FALSE_SOURCE.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.FALSE_SOURCE.ll -o \
; RUN:   %t.FALSE_SOURCE.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.FALSE_SOURCE.bc \
; RUN:   -o %t.FALSE_SOURCE.target.o

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=ADDRESS_COPY=1 %s -o %t.ADDRESS_COPY.o
; RUN: %ld.lld -shared %t.ADDRESS_COPY.o -o %t.ADDRESS_COPY.hsaco
; RUN: %transpile_cli %t.ADDRESS_COPY.hsaco --target-isa=gfx942 --emit-ir > \
; RUN:   %t.ADDRESS_COPY.ll
; RUN: %FileCheck %s --check-prefix=VALUE < %t.ADDRESS_COPY.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.ADDRESS_COPY.ll -o \
; RUN:   %t.ADDRESS_COPY.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.ADDRESS_COPY.bc \
; RUN:   -o %t.ADDRESS_COPY.target.o

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=OVERWRITE=1 %s -o %t.OVERWRITE.o
; RUN: %ld.lld -shared %t.OVERWRITE.o -o %t.OVERWRITE.hsaco
; RUN: %transpile_cli %t.OVERWRITE.hsaco --target-isa=gfx942 --emit-ir > \
; RUN:   %t.OVERWRITE.ll
; RUN: %FileCheck %s --check-prefix=OVERWRITE < %t.OVERWRITE.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.OVERWRITE.ll -o \
; RUN:   %t.OVERWRITE.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.OVERWRITE.bc -o \
; RUN:   %t.OVERWRITE.target.o
; OVERWRITE: zext i32 42 to i64
; OVERWRITE: zext i32 0 to i64
; OVERWRITE: add i64 {{.+}}, 0

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CMOV=1 %s -o %t.CMOV.o
; RUN: %ld.lld -shared %t.CMOV.o -o %t.CMOV.hsaco
; RUN: %transpile_cli %t.CMOV.hsaco --target-isa=gfx942 --emit-ir > %t.CMOV.ll
; RUN: %FileCheck %s --check-prefix=CMOV < %t.CMOV.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.CMOV.ll -o %t.CMOV.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.CMOV.bc -o \
; RUN:   %t.CMOV.target.o
; CMOV: select i1 {{.+}}, i32 7, i32 42

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=CMOV64=1 %s -o %t.CMOV64.o
; RUN: %ld.lld -shared %t.CMOV64.o -o %t.CMOV64.hsaco
; RUN: %transpile_cli %t.CMOV64.hsaco --target-isa=gfx942 --emit-ir > \
; RUN:   %t.CMOV64.ll
; RUN: %FileCheck %s --check-prefix=CMOV64 < %t.CMOV64.ll
; RUN: %opt -passes='default<O2>' -verify-each %t.CMOV64.ll -o %t.CMOV64.bc
; RUN: %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.CMOV64.bc -o \
; RUN:   %t.CMOV64.target.o
; CMOV64: [[LOW:%.+]] = zext i32 42 to i64
; CMOV64: [[HIGH:%.+]] = zext i32 0 to i64
; CMOV64: [[SHIFT:%.+]] = shl i64 [[HIGH]], 32
; CMOV64: [[OLD:%.+]] = or i64 [[LOW]], [[SHIFT]]
; CMOV64: select i1 {{.+}}, i64 7, i64 [[OLD]]

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=JOIN_DATA=1 %s -o %t.JOIN_DATA.o
; RUN: %ld.lld -shared %t.JOIN_DATA.o -o %t.JOIN_DATA.hsaco
; RUN: not %transpile_cli %t.JOIN_DATA.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS
; ADDRESS: unsupported-instruction-form: {{.+}} :: operand-read: 's{{.+}}' may hold a source code-object address

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=JOIN_DIFFERENT=1 %s -o %t.JOIN_DIFFERENT.o
; RUN: %ld.lld -shared %t.JOIN_DIFFERENT.o -o %t.JOIN_DIFFERENT.hsaco
; RUN: not %transpile_cli %t.JOIN_DIFFERENT.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=PARTIAL=1 %s -o %t.PARTIAL.o
; RUN: %ld.lld -shared %t.PARTIAL.o -o %t.PARTIAL.hsaco
; RUN: not %transpile_cli %t.PARTIAL.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=INCOMPLETE=1 %s -o %t.INCOMPLETE.o
; RUN: %ld.lld -shared %t.INCOMPLETE.o -o %t.INCOMPLETE.hsaco
; RUN: not %transpile_cli %t.INCOMPLETE.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=LOST_CARRY=1 %s -o %t.LOST_CARRY.o
; RUN: %ld.lld -shared %t.LOST_CARRY.o -o %t.LOST_CARRY.hsaco
; RUN: not %transpile_cli %t.LOST_CARRY.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=REPEATED_LOW=1 %s -o %t.REPEATED_LOW.o
; RUN: %ld.lld -shared %t.REPEATED_LOW.o -o %t.REPEATED_LOW.hsaco
; RUN: not %transpile_cli %t.REPEATED_LOW.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=UNKNOWN_OLD=1 %s -o %t.UNKNOWN_OLD.o
; RUN: %ld.lld -shared %t.UNKNOWN_OLD.o -o %t.UNKNOWN_OLD.hsaco
; RUN: not %transpile_cli %t.UNKNOWN_OLD.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=UNKNOWN_OLD64=1 %s -o %t.UNKNOWN_OLD64.o
; RUN: %ld.lld -shared %t.UNKNOWN_OLD64.o -o %t.UNKNOWN_OLD64.hsaco
; RUN: not %transpile_cli %t.UNKNOWN_OLD64.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=UNKNOWN_SOURCE=1 %s -o %t.UNKNOWN_SOURCE.o
; RUN: %ld.lld -shared %t.UNKNOWN_SOURCE.o -o %t.UNKNOWN_SOURCE.hsaco
; RUN: not %transpile_cli %t.UNKNOWN_SOURCE.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=FALSE_OLD=1 %s -o %t.FALSE_OLD.o
; RUN: %ld.lld -shared %t.FALSE_OLD.o -o %t.FALSE_OLD.hsaco
; RUN: not %transpile_cli %t.FALSE_OLD.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=TRUE_PARTIAL=1 %s -o %t.TRUE_PARTIAL.o
; RUN: %ld.lld -shared %t.TRUE_PARTIAL.o -o %t.TRUE_PARTIAL.hsaco
; RUN: not %transpile_cli %t.TRUE_PARTIAL.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=ADDRESS

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj \
; RUN:   --defsym=SCC_BACKEDGE=1 %s -o %t.SCC_BACKEDGE.o
; RUN: %ld.lld -shared %t.SCC_BACKEDGE.o -o %t.SCC_BACKEDGE.hsaco
; RUN: not %transpile_cli %t.SCC_BACKEDGE.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir 2>&1 | %FileCheck %s --check-prefix=SCC
; SCC: unsupported-instruction-form: s_cselect_b32 {{.+}} :: operand-read: 'scc' may hold a carry or overflow derived from a source code-object address

.amdhsa_code_object_version 6
.text
.globl provenance
.p2align 8
.type provenance,@function
provenance:
  s_mov_b32 s2, ttmp9
.set READ_IMAGE, 0
.set USE_LOADED, 0
.set READ_ORDINARY, 0
.ifdef SCC_BACKEDGE
  s_cmp_eq_u32 s2, 0
.Lscc_loop:
  s_cselect_b32 s10, 1, 2
  s_get_pc_i64 s[8:9]
  s_add_co_u32 s8, s8, 4
  s_add_co_ci_u32 s9, s9, 0
  s_branch .Lscc_loop
  s_endpgm
.else
  s_get_pc_i64 s[8:9]
.Lpc:
.ifdef FLOW
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_branch .Lload
.endif
.ifdef JOIN
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_cmp_eq_u32 s2, 0
  s_cbranch_scc1 .Lload
  s_branch .Lload
.endif
.ifdef JOIN_DATA
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_cmp_eq_u32 s2, 0
  s_cbranch_scc1 .Lload
  s_mov_b64 s[8:9], 0
  s_branch .Lload
.endif
.ifdef JOIN_DIFFERENT
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_cmp_eq_u32 s2, 0
  s_cbranch_scc1 .Lload
  s_get_pc_i64 s[8:9]
  s_branch .Lload
.endif
.ifdef PARTIAL
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_mov_b32 s8, 0
  s_branch .Lload
.endif
.ifdef INCOMPLETE
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_add_co_u32 s8, s8, 4
  s_branch .Lload
.endif
.ifdef LOST_CARRY
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_add_co_u32 s8, s8, 4
  s_cmp_eq_u32 s2, 0
  s_add_nc_u64 s[8:9], s[8:9], 4
  s_branch .Lload
.endif
.ifdef REPEATED_LOW
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_add_co_u32 s8, s8, 4
  s_add_co_u32 s8, s8, 4
  s_add_co_ci_u32 s9, s9, 0
  s_branch .Lload
.endif
.ifdef ADDRESS_COPY
.set USE_LOADED, 1
  s_add_nc_u64 s[8:9], s[8:9], .Ldata-.Lpc
  s_cmp_eq_u32 0, 0
  s_cmov_b64 s[10:11], s[8:9]
  s_load_b32 s12, s[10:11], 0
  s_branch .Luse
.endif
.ifdef OVERWRITE
.set READ_ORDINARY, 1
  s_mov_b32 s8, 42
  s_branch .Lhigh
.Lhigh:
  s_mov_b32 s9, 0
  s_branch .Lordinary
.endif
.ifdef CMOV
.set READ_ORDINARY, 1
  s_mov_b64 s[8:9], 42
  s_branch .Lordinary
.endif
.ifdef CMOV64
.set READ_ORDINARY, 1
  s_mov_b64 s[8:9], 42
  s_branch .Lordinary
.endif
.ifdef UNKNOWN_OLD64
.set READ_ORDINARY, 1
  s_cmp_eq_u32 s2, 0
  s_cmov_b64 s[8:9], 42
  s_branch .Lordinary
.endif
.ifdef FALSE_SOURCE
.set READ_ORDINARY, 1
  s_mov_b64 s[10:11], s[8:9]
  s_mov_b32 s8, 42
  s_cmp_eq_u32 0, 1
  s_cmov_b32 s8, s10
  s_branch .Lordinary
.endif
.ifdef TRUE32
.set READ_ORDINARY, 1
  s_cmp_eq_u32 0, 0
  s_cmov_b32 s8, 42
  s_branch .Lordinary
.endif
.ifdef TRUE64
.set READ_ORDINARY, 1
  s_cmp_eq_u32 0, 0
  s_cmov_b64 s[8:9], 42
  s_branch .Lordinary
.endif
.ifdef UNKNOWN_OLD
.set READ_ORDINARY, 1
  s_cmp_eq_u32 s2, 0
  s_cmov_b32 s8, 42
  s_branch .Lordinary
.endif
.ifdef UNKNOWN_SOURCE
  s_mov_b32 s10, 42
  s_cmp_eq_u32 s2, 0
  s_cmov_b32 s10, s8
  s_endpgm
.endif
.ifdef FALSE_OLD
.set READ_ORDINARY, 1
  s_cmp_eq_u32 0, 1
  s_cmov_b64 s[8:9], 42
  s_branch .Lordinary
.endif
.ifdef TRUE_PARTIAL
.set READ_IMAGE, 1
.set USE_LOADED, 1
  s_cmp_eq_u32 0, 0
  s_cmov_b32 s8, 42
  s_branch .Lload
.endif
.if READ_IMAGE
.Lload:
  s_add_nc_u64 s[8:9], s[8:9], .Ldata-.Lpc
  s_load_b32 s12, s[8:9], 0
.endif
.if USE_LOADED
.Luse:
  s_wait_kmcnt 0
  v_add_nc_u32 v0, s12, v0
  s_endpgm
.endif
.if READ_ORDINARY
.Lordinary:
.ifdef OVERWRITE
  s_add_nc_u64 s[8:9], s[8:9], 0
.endif
.ifdef CMOV64
  s_cmp_eq_u32 s2, 0
  s_cmov_b64 s[8:9], 7
.endif
.ifdef CMOV
  s_cmp_eq_u32 s2, 0
  s_cmov_b32 s8, 7
.endif
  v_add_nc_u32 v0, s8, v0
  s_endpgm
.endif
.endif
.Ldata:
  .long 42

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel provenance
  .amdhsa_kernarg_size 0
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 13
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: provenance
    .symbol: provenance.kd
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 13
    .vgpr_count: 1
    .wavefront_size: 32
...
.end_amdgpu_metadata
