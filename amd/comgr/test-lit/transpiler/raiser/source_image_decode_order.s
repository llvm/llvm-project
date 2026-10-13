; REQUIRES: comgr-has-transpiler

; The raiser fixtures assemble for gfx1250, which is also the ISA that spells
; the 64-bit scalar arithmetic these PC-relative chains use.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=defined_before_capture_kernel \
; RUN:   | %FileCheck %s

; A kernel that reads a source address is refused one kernel at a time,
; because the first refusal ends the run.

; RUN: not %transpile_cli %t.hsaco --emit-ir=refuse_back_branch_kernel \
; RUN:   2>&1 | %FileCheck %s --check-prefix=BACKBRANCH
; RUN: %transpile_cli %t.hsaco --emit-ir=unrelated_capture_kernel \
; RUN:   | %FileCheck %s --check-prefix=LATERCAPTURE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; Instructions are raised in decode order, which says nothing about the order
; the kernel runs them in, so a read can be raised before the s_get_pc_i64
; whose address it observes. Such a read is deferred and refused once every
; source address the kernel records is known.

	.globl	refuse_back_branch_kernel
	.p2align	8
	.type	refuse_back_branch_kernel,@function
refuse_back_branch_kernel:
; The branch back makes the load run after the s_get_pc_i64, so the pair holds
; a source address by the time the load reads it.
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lcapture
.Lback_branch_load:
; BACKBRANCH: unsupported-instruction-form: s_load_b32 {{.+}} :: operand-read: 's0' may hold a source code-object address
	s_load_b32 s3, s[0:1], 0x0
	s_endpgm
.Lcapture:
	s_get_pc_i64 s[0:1]
	s_branch .Lback_branch_load
	s_endpgm

	.globl	unrelated_capture_kernel
	.p2align	8
	.type	unrelated_capture_kernel,@function
unrelated_capture_kernel:
; A capture on a path that cannot reach the read does not affect it.
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Llater_capture
; LATERCAPTURE: add i32
	v_add_nc_u32 v0, s0, v0
	s_endpgm
.Llater_capture:
	s_get_pc_i64 s[0:1]
	s_endpgm

	.globl	defined_before_capture_kernel
	.p2align	8
	.type	defined_before_capture_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @defined_before_capture_kernel(
defined_before_capture_kernel:
; A block that writes the register itself reads what it wrote there, whatever
; another block records into the pair.
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Ldefined_capture
	s_mov_b32 s0, 7
; CHECK: add i32 7,
	v_add_nc_u32 v0, s0, v0
	s_endpgm
.Ldefined_capture:
	s_get_pc_i64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6
	.amdhsa_kernel refuse_back_branch_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel unrelated_capture_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel defined_before_capture_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_back_branch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         refuse_back_branch_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unrelated_capture_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         unrelated_capture_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           defined_before_capture_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         defined_before_capture_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
