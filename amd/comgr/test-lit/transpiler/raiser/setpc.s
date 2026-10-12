; REQUIRES: comgr-has-transpiler

; gfx1250 is the ISA the rest of the raiser fixtures assemble for, and the one
; that spells both the program-counter capture and the 64-bit scalar arithmetic
; the chains below are built from.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; The raised IR is fed back to the assembly parser, which verifies it. A jump
; into the entry block, or a block left without a terminator, is caught there
; rather than by a pattern below.
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_forward_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_backward_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=swappc_call_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_bare_capture_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_crossblock_kernel \
; RUN:   | %llvm-as -o /dev/null

; RUN: %transpile_cli %t.hsaco --emit-ir=nosetpc_kernel \
; RUN:   | %FileCheck %s --check-prefix=NOSETPC
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_forward_kernel \
; RUN:   | %FileCheck %s --check-prefix=FORWARD
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_backward_kernel \
; RUN:   | %FileCheck %s --check-prefix=BACKWARD
; RUN: %transpile_cli %t.hsaco --emit-ir=swappc_call_kernel \
; RUN:   | %FileCheck %s --check-prefix=CALL
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_bare_capture_kernel \
; RUN:   | %FileCheck %s --check-prefix=BARE

; RUN: not %transpile_cli %t.hsaco --emit-ir=setpc_nonpair_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=NONPAIR
; RUN: not %transpile_cli %t.hsaco --emit-ir=setpc_clobbered_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CLOBBERED
; RUN: not %transpile_cli %t.hsaco --emit-ir=setpc_folded_low_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=FOLDEDLOW
; RUN: not %transpile_cli %t.hsaco --emit-ir=setpc_carry_clobbered_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=CARRY
; RUN: %transpile_cli %t.hsaco --emit-ir=setpc_crossblock_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSSBLOCK
; RUN: not %transpile_cli %t.hsaco --emit-ir=setpc_midinst_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MIDINST
; RUN: not %transpile_cli %t.hsaco --emit-ir=setpc_outside_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=OUTSIDE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

; s_set_pc_i64 names no offset of its own: where it goes is whatever the pair
; it reads holds. A program-counter capture names the offset of the instruction
; after it, and adding a constant to that capture names another offset in the
; same kernel, so a capture carried by constants and then jumped through is a
; branch between recovered blocks.

	.globl	nosetpc_kernel
	.p2align	8
	.type	nosetpc_kernel,@function
; A kernel whose chain is never jumped through gains no block from the
; analysis: the only block start is the one the decode already found.
; NOSETPC-LABEL: define amdgpu_kernel void @nosetpc_kernel(
nosetpc_kernel:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 12
; NOSETPC: bb_0x0:
; NOSETPC-NOT: bb_
	s_endpgm

	.globl	setpc_forward_kernel
	.p2align	8
	.type	setpc_forward_kernel,@function
; FORWARD-LABEL: define amdgpu_kernel void @setpc_forward_kernel(
setpc_forward_kernel:
	s_mov_b32 s2, 11
; The capture sits four bytes into the kernel and is four bytes wide, so it
; names eight. A displacement of twelve carries it to twenty, which steps over
; the seed below.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 12
; FORWARD: br label %[[FW_TARGET:.+]]
	s_set_pc_i64 s[10:11]
	s_mov_b32 s2, 22
; FORWARD: [[FW_TARGET]]:
; FORWARD: uitofp i32 11 to float
	s_cvt_f32_u32 s3, s2
; FORWARD: ret void
	s_endpgm

	.globl	setpc_backward_kernel
	.p2align	8
	.type	setpc_backward_kernel,@function
; BACKWARD-LABEL: define amdgpu_kernel void @setpc_backward_kernel(
setpc_backward_kernel:
; BACKWARD: entry:
; BACKWARD: br label %[[SEED:.+]]
; BACKWARD: [[SEED]]:
	s_mov_b32 s2, 0
; BACKWARD: br label %[[HEAD:.+]]
; BACKWARD: [[HEAD]]:
backward_head:
; BACKWARD: [[NOTTED:%.+]] = xor i32 {{.+}}, -1
	s_not_b32 s2, s2
; BACKWARD: br i1 {{.+}}, label %[[EXIT:.+]], label %[[LATCH:.+]]
	s_cbranch_scc0 backward_exit
; BACKWARD: [[LATCH]]:
; A displacement reaching back is written across both halves of the pair: the
; low add carries the capture past the end of the address space and the high
; add brings it back, which is what the source hardware does as well. The
; capture names sixteen, so minus twelve reaches four, the head of the loop.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, -12
	s_addc_u32 s11, s11, -1
; BACKWARD: br label %[[HEAD]]
	s_set_pc_i64 s[10:11]
; BACKWARD: [[EXIT]]:
backward_exit:
; BACKWARD: uitofp i32 [[NOTTED]] to float
	s_cvt_f32_u32 s3, s2
; BACKWARD: ret void
	s_endpgm

	.globl	swappc_call_kernel
	.p2align	8
	.type	swappc_call_kernel,@function
; CALL-LABEL: define amdgpu_kernel void @swappc_call_kernel(
swappc_call_kernel:
; CALL: entry:
; CALL: br label %[[CALLER:bb_.+]]
; CALL: [[CALLER]]:
	s_mov_b32 s2, 11
; The capture names eight and a displacement of twelve carries it to twenty,
; which is the callee below.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 12
; CALL: br label %[[CALLEE:bb_.+]]
	s_swap_pc_i64 s[12:13], s[10:11]
; The call names where it returns to as well, so this leads a block of its own
; even though nothing reaches it until a return can be resolved.
; CALL: [[RETURN:bb_.+]]: {{.*}}No predecessors!
	s_mov_b32 s2, 22
; CALL: [[CALLEE]]:
; The callee is reached from the call rather than from the block the call
; returns to, so the seed it reads is the one the caller left.
; CALL: uitofp i32 11 to float
	s_cvt_f32_u32 s3, s2
	s_endpgm

	.globl	setpc_bare_capture_kernel
	.p2align	8
	.type	setpc_bare_capture_kernel,@function
; BARE-LABEL: define amdgpu_kernel void @setpc_bare_capture_kernel(
setpc_bare_capture_kernel:
	s_mov_b32 s2, 11
; BARE: uitofp i32 11 to float
	s_cvt_f32_u32 s3, s2
; A capture with nothing added to it names the offset of the instruction that
; follows it, and that is a target like any other. Reaching it from below is
; the only shape a bare capture can take, since a jump reading one can never
; get past itself, so the instructions between the two repeat.
	s_get_pc_i64 s[10:11]
; BARE: br label %[[BARE_TARGET:.+]]
; BARE: [[BARE_TARGET]]:
	s_mov_b32 s2, 22
; BARE: br label %[[BARE_TARGET]]
	s_set_pc_i64 s[10:11]
	s_endpgm

; Every shape whose target the analysis cannot work out is refused, and each
; carries the reason it could not.

	.globl	setpc_nonpair_kernel
	.p2align	8
	.type	setpc_nonpair_kernel,@function
setpc_nonpair_kernel:
; NONPAIR: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads its target from something other than a scalar register pair
	s_set_pc_i64 vcc
	s_endpgm

	.globl	setpc_clobbered_kernel
	.p2align	8
	.type	setpc_clobbered_kernel,@function
setpc_clobbered_kernel:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 12
; Writing either half of the pair ends the chain, and the write says so rather
; than the jump being read as still carrying the displaced capture.
	s_mov_b32 s11, 0
; CLOBBERED: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[10:11], which some path writes without computing a source offset in it
	s_set_pc_i64 s[10:11]
	s_mov_b32 s2, 22
	s_endpgm

	.globl	setpc_folded_low_kernel
	.p2align	8
	.type	setpc_folded_low_kernel,@function
setpc_folded_low_kernel:
	s_get_pc_i64 s[10:11]
; Folding a constant into the low half overwrites the capture, so the add that
; follows displaces nothing and the offset the capture named does not reach the
; jump even though the displacement would have landed on an instruction.
	s_add_u32 s10, 1, 2
	s_add_u32 s10, s10, 12
; FOLDEDLOW: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[10:11], which some path writes without computing a source offset in it
	s_set_pc_i64 s[10:11]
	s_endpgm

	.globl	setpc_carry_clobbered_kernel
	.p2align	8
	.type	setpc_carry_clobbered_kernel,@function
setpc_carry_clobbered_kernel:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 16
; The high half of a split displacement adds the carry the low half left in the
; condition code, so an instruction in between separates the two adds and the
; high one no longer continues the low one. What it reads is then half of a
; source address that nothing carries forward, which is refused where it is
; read rather than at the jump below.
	s_add_u32 s20, 1, 2
; CARRY: unsupported-instruction-form: s_add_co_ci_u32 {{.+}} :: operand-read: 's11' may hold a source code-object address
	s_addc_u32 s11, s11, 0
	s_set_pc_i64 s[10:11]
	s_endpgm

	.globl	setpc_crossblock_kernel
	.p2align	8
	.type	setpc_crossblock_kernel,@function
; CROSSBLOCK-LABEL: define amdgpu_kernel void @setpc_crossblock_kernel(
setpc_crossblock_kernel:
; The chain is built in one block and jumped through in another, so the jump
; reads what the paths into its block left in the pair rather than anything its
; own block computed. One offset over every path is a plain branch.
;
; The capture is four bytes wide and sits at the start of the kernel, so it
; names four. A displacement of twenty carries it to twenty-four, where the
; target below sits.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 20
; The branch forks the control flow and nothing more. It reads EXEC rather
; than SCC, which the add above leaves holding a carry out of the address the
; source image records rather than the one the loader produces.
; CROSSBLOCK: br i1 {{.+}}, label %[[CROSS_JOIN:bb_.+]], label %
	s_cbranch_execz cross_join
; CROSSBLOCK: [[CROSS_JOIN]]:
cross_join:
; CROSSBLOCK-NEXT: br label %[[CROSS_TARGET:bb_.+]]
	s_set_pc_i64 s[10:11]
	s_mov_b32 s2, 22
	s_branch cross_done
cross_target:
; CROSSBLOCK: [[CROSS_TARGET]]: {{.*}}preds = %[[CROSS_JOIN]]
	s_cvt_f32_u32 s3, s2
cross_done:
	s_endpgm

	.globl	setpc_midinst_kernel
	.p2align	8
	.type	setpc_midinst_kernel,@function
setpc_midinst_kernel:
; The move below spans three dwords, so a displacement of twelve from a capture
; naming four lands in the middle of it.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 12
; MIDINST: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reaches source offset 0x{{.+}}, which no decoded instruction starts at
	s_set_pc_i64 s[10:11]
	s_mov_b64 s[0:1], 0x123456789abcdef
	s_endpgm

	.globl	setpc_outside_kernel
	.p2align	8
	.type	setpc_outside_kernel,@function
setpc_outside_kernel:
; A displacement reaching past everything the decode followed.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 64
; OUTSIDE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reaches source offset 0x{{.+}}, which no decoded instruction starts at
	s_set_pc_i64 s[10:11]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel nosetpc_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_forward_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_backward_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel swappc_call_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_nonpair_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_clobbered_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_folded_low_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_bare_capture_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_carry_clobbered_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_crossblock_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_midinst_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel setpc_outside_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
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
    .name:           nosetpc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         nosetpc_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_forward_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_forward_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_backward_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_backward_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           swappc_call_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         swappc_call_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_nonpair_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_nonpair_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_clobbered_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_clobbered_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_bare_capture_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_bare_capture_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_carry_clobbered_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_carry_clobbered_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_crossblock_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_crossblock_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_folded_low_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_folded_low_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_midinst_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_midinst_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           setpc_outside_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         setpc_outside_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
