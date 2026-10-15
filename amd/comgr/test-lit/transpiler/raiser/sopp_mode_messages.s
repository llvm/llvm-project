; REQUIRES: comgr-has-transpiler

; The SOPP opcodes that set a mode or send a message. The mode ones are gfx10
; additions, so the fixture is a gfx1250 kernel; sopp_messages_wave64.s covers
; what a pre-gfx11 source makes of the same messages.
; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco --emit-ir=sethalt_kernel \
; RUN:   | %FileCheck %s --check-prefix=HALT
; RUN: %transpile_cli %t.hsaco --emit-ir=mode_kernel \
; RUN:   | %FileCheck %s --check-prefix=MODE
; RUN: %transpile_cli %t.hsaco --emit-ir=messages_kernel \
; RUN:   | %FileCheck %s --check-prefix=MESSAGES

; RUN: not %transpile_cli %t.hsaco --emit-ir=round_up_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ROUND-UP
; RUN: not %transpile_cli %t.hsaco --emit-ir=denorm_flush_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DENORM-FLUSH
; RUN: not %transpile_cli %t.hsaco --emit-ir=dealloc_halt_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=DEALLOC-HALT
; RUN: not %transpile_cli %t.hsaco --emit-ir=sysmsg_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SYSMSG
; RUN: not %transpile_cli %t.hsaco --emit-ir=sysmsg_halt_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SYSMSG-HALT
; RUN: not %transpile_cli %t.hsaco --emit-ir=endpgm_saved_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ENDPGM-SAVED

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	sethalt_kernel
	.p2align	8
	.type	sethalt_kernel,@function
sethalt_kernel:
; The immediate reaches the intrinsic unchanged, so two different halts stay
; two different halts.
; HALT-LABEL: define amdgpu_kernel void @sethalt_kernel(
; HALT: call void @llvm.amdgcn.s.sethalt(i32 1)
	s_sethalt 1
; HALT-NEXT: call void @llvm.amdgcn.s.sethalt(i32 3)
	s_sethalt 3
; HALT-NEXT: ret void
	s_endpgm

	.globl	mode_kernel
	.p2align	8
	.type	mode_kernel,@function
mode_kernel:
; Naming the mode the raised kernel already computes in asks for nothing, so
; both of these lift to nothing.
; MODE-LABEL: define amdgpu_kernel void @mode_kernel(
; MODE: bb_0x{{.+}}:
; MODE-NEXT: ret void
; MODE-NEXT: }
	s_round_mode 0x0
	s_denorm_mode 15
	s_endpgm

	.globl	messages_kernel
	.p2align	8
	.type	messages_kernel,@function
messages_kernel:
; Both interrupts go out carrying M0, which is what the message reads its
; payload from.
; MESSAGES-LABEL: define amdgpu_kernel void @messages_kernel(
	s_mov_b32 m0, 42
; MESSAGES: call void @llvm.amdgcn.s.sendmsg(i32 1, i32 42)
	s_sendmsg sendmsg(MSG_INTERRUPT)
; MESSAGES-NEXT: call void @llvm.amdgcn.s.sendmsghalt(i32 1, i32 42)
	s_sendmsghalt sendmsg(MSG_INTERRUPT)
; The deallocation hint is dropped rather than passed on: the target backend
; decides where the raised kernel is done with its VGPRs.
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; MESSAGES-NEXT: ret void
	s_endpgm

	.globl	round_up_kernel
	.p2align	8
	.type	round_up_kernel,@function
round_up_kernel:
; ROUND-UP: unsupported-instruction-form: s_round_mode [SOPP]
; ROUND-UP-SAME: selects rounding mode 1 rather than round-to-nearest-even
	s_round_mode 0x1
	s_endpgm

	.globl	denorm_flush_kernel
	.p2align	8
	.type	denorm_flush_kernel,@function
denorm_flush_kernel:
; DENORM-FLUSH: unsupported-instruction-form: s_denorm_mode [SOPP]
; DENORM-FLUSH-SAME: selects denormal mode 0 rather than kernel descriptor mode 15
	s_denorm_mode 0
	s_endpgm

	.globl	dealloc_halt_kernel
	.p2align	8
	.type	dealloc_halt_kernel,@function
dealloc_halt_kernel:
; Dropping the deallocation hint is only free when nothing else rides on it,
; and the halting spelling carries a halt as well.
; DEALLOC-HALT: unsupported-instruction-form: s_sendmsghalt [SOPP]
; DEALLOC-HALT-SAME: halts the wave alongside a VGPR deallocation
	s_sendmsghalt sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm

	.globl	sysmsg_kernel
	.p2align	8
	.type	sysmsg_kernel,@function
sysmsg_kernel:
; SYSMSG: unsupported-instruction-form: s_sendmsg [SOPP]
; SYSMSG-SAME: sends message 0x1f, and the interrupt is the only message that
; SYSMSG-SAME: means the same thing on every target
	s_sendmsg 0x1f
	s_endpgm

	.globl	sysmsg_halt_kernel
	.p2align	8
	.type	sysmsg_halt_kernel,@function
sysmsg_halt_kernel:
; SYSMSG-HALT: unsupported-instruction-form: s_sendmsghalt [SOPP]
; SYSMSG-HALT-SAME: sends message 0x1f
	s_sendmsghalt 0x1f
	s_endpgm

	.globl	endpgm_saved_kernel
	.p2align	8
	.type	endpgm_saved_kernel,@function
endpgm_saved_kernel:
; ENDPGM-SAVED: unsupported-instruction-form: s_endpgm_saved [SOPP]
; ENDPGM-SAVED-SAME: ends the wave for a context save nothing here resumes
	s_endpgm_saved

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel sethalt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel mode_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
	.end_amdhsa_kernel
	.amdhsa_kernel messages_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel round_up_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel denorm_flush_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
	.end_amdhsa_kernel
	.amdhsa_kernel dealloc_halt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sysmsg_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel sysmsg_halt_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
	.end_amdhsa_kernel
	.amdhsa_kernel endpgm_saved_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .name:           sethalt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sethalt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           mode_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         mode_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           messages_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         messages_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           round_up_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         round_up_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           denorm_flush_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         denorm_flush_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dealloc_halt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         dealloc_halt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sysmsg_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sysmsg_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           sysmsg_halt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         sysmsg_halt_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           endpgm_saved_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         endpgm_saved_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
