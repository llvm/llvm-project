; REQUIRES: comgr-has-transpiler

; The SOP1 opcodes that ask the hardware about the wave itself rather than
; computing on its registers. All five are gfx12 additions, so the fixture is a
; gfx1250 kernel.
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=alloc_vgpr_kernel \
; RUN:   | %FileCheck %s --check-prefix=ALLOC
; RUN: %transpile_cli %t.hsaco --emit-ir=sleep_var_kernel \
; RUN:   | %FileCheck %s --check-prefix=SLEEP

; RUN: not %transpile_cli %t.hsaco --emit-ir=shader_cycles_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CYCLES
; RUN: not %transpile_cli %t.hsaco --emit-ir=rtn32_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=RTN32
; RUN: not %transpile_cli %t.hsaco --emit-ir=rtn64_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=RTN64

; A SOP1 opcode the handler does not lift is refused, not mislowered.
; RUN: not %transpile_cli %t.hsaco --emit-ir=rfe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=UNHANDLED
; UNHANDLED: unsupported-instruction-form: s_rfe_i64

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	alloc_vgpr_kernel
	.p2align	8
	.type	alloc_vgpr_kernel,@function
alloc_vgpr_kernel:
; ALLOC-LABEL: define amdgpu_kernel void @alloc_vgpr_kernel(
; The allocation always succeeds, so the SCC it writes is a constant true and
; the conditional move that reads it keeps the value it was asked to take. The
; move and the reversal after it are there to hold that value in the raised IR.
; Allocating waits for the memory in flight first.
	s_mov_b32 s2, 3
	s_mov_b32 s6, 7
	s_alloc_vgpr 0x100
	s_cmov_b32 s6, s2
; ALLOC: fence syncscope("agent") seq_cst
; ALLOC: call i32 @llvm.bitreverse.i32(i32 3)
	s_brev_b32 s7, s6
; The register-operand form asks for a count nothing wrote, and still succeeds:
; the answer does not depend on how many registers were asked for.
	s_mov_b32 s8, 5
	s_mov_b32 s9, 9
	s_alloc_vgpr s4
	s_cmov_b32 s9, s8
; ALLOC: fence syncscope("agent") seq_cst
; ALLOC: call i32 @llvm.bitreverse.i32(i32 5)
	s_brev_b32 s10, s9
; ALLOC: ret void
	s_endpgm

	.globl	sleep_var_kernel
	.p2align	8
	.type	sleep_var_kernel,@function
sleep_var_kernel:
; A variable sleep spends issue slots and leaves nothing behind to read, in
; either operand form, so the body is the terminator.
; SLEEP-LABEL: define amdgpu_kernel void @sleep_var_kernel(
; SLEEP: bb_0x{{.+}}:
; SLEEP-NEXT: ret void
; SLEEP-NEXT: }
	s_sleep_var 5
	s_sleep_var s0
	s_endpgm

	.globl	shader_cycles_kernel
	.p2align	8
	.type	shader_cycles_kernel,@function
shader_cycles_kernel:
; CYCLES: unsupported-instruction-form: s_get_shader_cycles_u64 [SOP1]
; CYCLES-SAME: reads the source shader clock, whose rate and epoch the raised
; CYCLES-SAME: kernel does not share
	s_get_shader_cycles_u64 s[0:1]
	s_endpgm

	.globl	rtn32_kernel
	.p2align	8
	.type	rtn32_kernel,@function
rtn32_kernel:
; The refusal names the message that was asked for: the doorbell is 128.
; RTN32: unsupported-instruction-form: s_sendmsg_rtn_b32 [SOP1]
; RTN32-SAME: reads back message 128, which reports on the source wave's queue
; RTN32-SAME: and hardware placement
	s_sendmsg_rtn_b32 s0, sendmsg(MSG_RTN_GET_DOORBELL)
	s_endpgm

	.globl	rtn64_kernel
	.p2align	8
	.type	rtn64_kernel,@function
rtn64_kernel:
; And the realtime counter is 131.
; RTN64: unsupported-instruction-form: s_sendmsg_rtn_b64 [SOP1]
; RTN64-SAME: reads back message 131, which reports on the source wave's queue
; RTN64-SAME: and hardware placement
	s_sendmsg_rtn_b64 s[0:1], sendmsg(MSG_RTN_GET_REALTIME)
	s_endpgm

	.globl	rfe_kernel
	.p2align	8
	.type	rfe_kernel,@function
rfe_kernel:
	s_rfe_i64 s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel alloc_vgpr_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 11
	.end_amdhsa_kernel
	.amdhsa_kernel sleep_var_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel shader_cycles_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel rtn32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel rtn64_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel rfe_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
    .name:           alloc_vgpr_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .symbol:         alloc_vgpr_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sleep_var_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sleep_var_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           shader_cycles_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         shader_cycles_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rtn32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         rtn32_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rtn64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         rtn64_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           rfe_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         rfe_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
