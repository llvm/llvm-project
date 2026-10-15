; REQUIRES: comgr-has-transpiler, comgr-has-llc

; RUN: for case in SIGNED UNSIGNED LOAD DYNAMIC CLOBBER LOW_SCC HIGH_SCC \
; RUN:             ESCAPE COMPARE NO_CARRY SIGNED_LOW BAD_TARGET LOST_CARRY; do \
; RUN:   %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj --defsym=$case=1 %s -o %t.$case.o && \
; RUN:   %ld.lld -shared %t.$case.o -o %t.$case.hsaco || exit 1; \
; RUN: done
; RUN: %transpile_cli %t.SIGNED.hsaco --target-isa=gfx942 --emit-ir \
; RUN:   | %FileCheck %s
; RUN: %transpile_cli %t.UNSIGNED.hsaco --target-isa=gfx942 --emit-ir \
; RUN:   | %FileCheck %s
; RUN: %transpile_cli %t.LOAD.hsaco --target-isa=gfx942 --emit-ir \
; RUN:   | %FileCheck %s --check-prefix=LOAD
; RUN: for case in SIGNED UNSIGNED LOAD; do \
; RUN:   %transpile_cli %t.$case.hsaco --target-isa=gfx942 --emit-ir \
; RUN:     | %opt -passes='default<O2>' -verify-each -o %t.$case.bc && \
; RUN:   %llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=obj %t.$case.bc -o %t.$case.target.o || exit 1; \
; RUN: done

; RUN: for case in DYNAMIC CLOBBER LOW_SCC NO_CARRY SIGNED_LOW LOST_CARRY; do \
; RUN:   not %transpile_cli %t.$case.hsaco --target-isa=gfx942 --emit-ir 2>&1 \
; RUN:     | %FileCheck %s --check-prefix=UNRESOLVED || exit 1; \
; RUN: done
; UNRESOLVED: unsupported-instruction-form: s_add_co_{{.+}} :: operand-read: 's8' may hold a source code-object address
; RUN: not %transpile_cli %t.HIGH_SCC.hsaco --target-isa=gfx942 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SCC
; SCC: unsupported-instruction-form: s_cselect_b32 {{.+}} :: operand-read: 'scc' may hold a carry or overflow derived from a source code-object address; relocation may change the result
; RUN: not %transpile_cli %t.ESCAPE.hsaco --target-isa=gfx942 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ESCAPE
; ESCAPE: unsupported-instruction-form: v_mov_b32 {{.+}} :: operand-read: 's8' may hold a source code-object address
; RUN: not %transpile_cli %t.COMPARE.hsaco --target-isa=gfx942 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=COMPARE
; COMPARE: unsupported-instruction-form: s_cmp_eq_u64 {{.+}} :: operand-read: 's8' may hold a source code-object address
; RUN: not %transpile_cli %t.BAD_TARGET.hsaco --target-isa=gfx942 --emit-ir 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=BAD-TARGET
; BAD-TARGET: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reaches source offset {{.+}}, which no decoded instruction starts at

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6
.text
.globl register_displacement
.p2align 8
.type register_displacement,@function
; CHECK-LABEL: define amdgpu_kernel void @register_displacement(
register_displacement:
; CHECK: br label %[[START:.+]]
; CHECK: [[START]]:
; CHECK: br label %[[CAPTURE:.+]]
  s_branch .Lcapture
; CHECK: [[HELPER:.+]]:
.Lhelper:
; CHECK: add i32 42,
  v_add_nc_u32 v0, 42, v0
  s_endpgm
; CHECK: [[CAPTURE]]:
.Lcapture:
  s_get_pc_i64 s[8:9]
.Lpc:
.ifdef LOST_CARRY
  s_add_co_i32 s10, .Lforward-.Lpc-3, 4
  ; The next low add overwrites this carry before it reaches the high half.
  s_add_co_u32 s8, s8, -1
.else
.ifdef UNSIGNED
  s_add_co_u32 s10, .Lforward-.Lpc-4, 4
.else
  s_add_co_i32 s10, .Lforward-.Lpc-4, 4
.endif
.endif
.ifdef DYNAMIC
  s_mov_b32 s10, s0
.endif
.ifdef CLOBBER
  s_mov_b32 s10, 0
.endif
.ifdef SIGNED_LOW
  s_add_co_i32 s8, s8, s10
.else
  s_add_co_u32 s8, s8, s10
.endif
.ifdef LOW_SCC
  s_cselect_b32 s11, 0, 1
.endif
.ifndef NO_CARRY
.ifdef LOST_CARRY
  s_add_co_ci_u32 s9, s9, -1
.else
.ifdef BAD_TARGET
  s_add_co_ci_u32 s9, s9, 1
.else
  s_add_co_ci_u32 s9, s9, 0
.endif
.endif
.endif
.ifdef HIGH_SCC
  s_cselect_b32 s11, 0, 1
.endif
.ifdef ESCAPE
  v_mov_b32 v1, s8
.endif
.ifdef COMPARE
  s_cmp_eq_u64 s[8:9], 16
.endif
; CHECK: br label %[[FORWARD:.+]]
  s_set_pc_i64 s[8:9]
  s_endpgm
; CHECK: [[FORWARD]]:
.Lforward:
  s_get_pc_i64 s[8:9]
.Lbackpc:
.ifdef LOAD
  s_add_co_i32 s10, .Ldata-.Lbackpc-4, 4
  s_add_co_u32 s8, s8, s10
  s_add_co_ci_u32 s9, s9, 0
  s_load_b32 s12, s[8:9], 0
  s_wait_kmcnt 0
; LOAD: add i32 305419896,
  v_add_nc_u32 v0, s12, v0
.else
  s_add_co_i32 s10, .Lhelper-.Lbackpc+4, -4
  s_add_co_u32 s8, s8, s10
  s_add_co_ci_u32 s9, s9, -1
; CHECK: br label %[[HELPER]]
  s_set_pc_i64 s[8:9]
.endif
  s_endpgm
.Ldata:
  .long 0x12345678

.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel register_displacement
  .amdhsa_kernarg_size 0
  .amdhsa_next_free_vgpr 2
  .amdhsa_next_free_sgpr 13
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: register_displacement
    .symbol: register_displacement.kd
    .kernarg_segment_size: 0
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 13
    .vgpr_count: 2
    .wavefront_size: 32
...
.end_amdgpu_metadata
