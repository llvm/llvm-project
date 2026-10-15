; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: not %transpile_cli %t.hsaco \
; RUN:   --emit-ir=refuse_hwreg_read,refuse_hwreg_write,refuse_mode_read 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj --defsym=READ_REG=3 %s -o %t.trap.o
; RUN: %ld.lld -shared %t.trap.o -o %t.trap.hsaco
; RUN: not %transpile_cli %t.trap.hsaco --emit-ir=refuse_hwreg_read 2>&1 | %FileCheck %s --check-prefix=TRAPSTS
; TRAPSTS: cannot reproduce hardware-register read for id 3
; RUN: %llvm-mc -triple=amdgpu9.42-amd-amdhsa -filetype=obj --defsym=READ_REG=17 %s -o %t.except.o
; RUN: %ld.lld -shared %t.except.o -o %t.except.hsaco
; RUN: not %transpile_cli %t.except.hsaco --emit-ir=refuse_hwreg_read 2>&1 | %FileCheck %s --check-prefix=PRIVATE
; PRIVATE: cannot reproduce hardware-register read for id 17
.ifndef READ_REG
.set READ_REG, 20
.endif
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	refuse_hwreg_read
	.p2align	8
	.type	refuse_hwreg_read,@function
refuse_hwreg_read:
	; REFUSE:      in kernel 'refuse_hwreg_read'
	; REFUSE-SAME: cannot reproduce hardware-register read for id 20
	; READ-SAME: s_getreg_b32
	; READ-SAME: cannot reproduce hardware-register read for id 20
	s_getreg_b32 s0, hwreg(READ_REG)
	s_endpgm

	.globl	refuse_hwreg_write
	.p2align	8
	.type	refuse_hwreg_write,@function
refuse_hwreg_write:
	; REFUSE:      in kernel 'refuse_hwreg_write'
	; REFUSE-SAME: cannot reproduce hardware-register write for id 16
	; WRITE-SAME: s_setreg_imm32_b32
	; WRITE-SAME: cannot reproduce hardware-register write for id 16
	s_setreg_imm32_b32 hwreg(16), 1
	s_endpgm

	.globl	refuse_mode_read
	.p2align	8
	.type	refuse_mode_read,@function
refuse_mode_read:
	; REFUSE:      in kernel 'refuse_mode_read'
	; REFUSE-SAME: cannot reproduce hardware-register read for id 1
	; MODE-READ-SAME: s_getreg_b32
	; MODE-READ-SAME: cannot reproduce hardware-register read for id 1
	s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 4), 5
	s_getreg_b32 s0, hwreg(HW_REG_MODE, 0, 4)
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel refuse_hwreg_read
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel refuse_hwreg_write
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel refuse_mode_read
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
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
    .name:           refuse_hwreg_read
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_hwreg_read.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_hwreg_write
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_hwreg_write.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           refuse_mode_read
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         refuse_mode_read.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
