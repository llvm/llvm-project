; REQUIRES: comgr-has-transpiler

; gfx1250 is the ISA the rest of the raiser fixtures assemble for, and the one
; that spells both the program-counter capture and the 64-bit scalar arithmetic
; the chains below are built from.

; RUN: %llvm-mc -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco

; RUN: %transpile_cli %t.hsaco \
; RUN:   --emit-ir=dispatch_kernel,call_return_kernel,two_call_sites_kernel \
; RUN:   --emit-ir=swap_dispatch_kernel,dispatch_atcap_kernel \
; RUN:   | %FileCheck %s

; The raised IR is fed back to the assembly parser, which verifies it. A jump
; into the entry block, or a block left without a terminator, is caught there
; rather than by a pattern above.
; RUN: %transpile_cli %t.hsaco --emit-ir=dispatch_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=call_return_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=two_call_sites_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=swap_dispatch_kernel \
; RUN:   | %llvm-as -o /dev/null
; RUN: %transpile_cli %t.hsaco --emit-ir=dispatch_atcap_kernel \
; RUN:   | %llvm-as -o /dev/null

; RUN: not %transpile_cli %t.hsaco \
; RUN:   --emit-ir=dispatch_partial_kernel,dispatch_overcap_kernel \
; RUN:   --emit-ir=capture_clobber_kernel,undecoded_kernel \
; RUN:   --emit-ir=unreachable_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text

	.globl	dispatch_kernel
	.p2align	8
	.type	dispatch_kernel,@function
; Three paths each leave a different source offset in the register pair, so the
; jump goes to one of three blocks. None of them can be named by a branch, so
; the raise dispatches on the offset itself, which is what a program-counter
; capture and a call both leave behind. Falling out of the table cannot happen
; and traps rather than picking a block.
; CHECK-LABEL: define amdgpu_kernel void @dispatch_kernel(
dispatch_kernel:
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc0 disp_second
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, disp_first-.
	s_branch disp_join
disp_second:
	s_cmp_eq_u32 s1, 0
	s_cbranch_scc0 disp_third
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, disp_second_target-.
	s_branch disp_join
disp_third:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, disp_third_target-.
disp_join:
; CHECK: switch i64 %{{.+}}, label %[[TRAP:.+]] [
; CHECK-NEXT: i64 {{.+}}, label %[[FIRST:bb_.+]]
; CHECK-NEXT: i64 {{.+}}, label %[[SECOND:bb_.+]]
; CHECK-NEXT: i64 {{.+}}, label %[[THIRD:bb_.+]]
; CHECK-NEXT: ]
	s_set_pc_i64 s[10:11]
; The first target is where the jump falls through to, which leads a block
; because the jump ends one. The other two sit in the middle of what the decode
; read as straight-line code, and lead blocks only because the jump reaches
; them.
disp_first:
; CHECK: [[FIRST]]:
	s_mov_b32 s2, 11
disp_second_target:
; CHECK: [[SECOND]]:
	s_mov_b32 s2, 22
disp_third_target:
; CHECK: [[THIRD]]:
	s_cvt_f32_u32 s3, s2
	s_endpgm
; CHECK: [[TRAP]]:
; CHECK-NEXT: call void @llvm.trap()
; CHECK-NEXT: unreachable

	.globl	call_return_kernel
	.p2align	8
	.type	call_return_kernel,@function
; The call leaves the offset it returns to in its destination register pair,
; which is a source offset like any a capture computes. The callee jumps through
; that pair without writing it, so what reaches the callee is what the call
; left, and the return is a branch back.
; CHECK-LABEL: define amdgpu_kernel void @call_return_kernel(
call_return_kernel:
; CHECK: entry:
; CHECK: br label %[[CALLER:bb_.+]]
; CHECK: [[CALLER]]:
	s_mov_b32 s2, 11
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, ret_callee-.
; CHECK: br label %[[CALLEE:bb_.+]]
	s_swap_pc_i64 s[12:13], s[10:11]
ret_after_call:
; CHECK: [[AFTER:bb_.+]]: {{.*}}preds = %[[CALLEE]]
; CHECK: uitofp i32 33 to float
	s_cvt_f32_u32 s3, s2
	s_branch ret_done
ret_callee:
; CHECK: [[CALLEE]]:
	s_mov_b32 s2, 33
; CHECK: br label %[[AFTER]]
	s_set_pc_i64 s[12:13]
ret_done:
	s_endpgm

	.globl	two_call_sites_kernel
	.p2align	8
	.type	two_call_sites_kernel,@function
; Two calls to one callee leave two different return offsets in the same
; register pair, so the return jumps to one of two blocks and dispatches like
; any other jump the analysis cannot narrow to a single offset. Each call site
; names the block the callee returns to, and the two cases of the dispatch are
; those two blocks.
; CHECK-LABEL: define amdgpu_kernel void @two_call_sites_kernel(
two_call_sites_kernel:
; CHECK: [[CALLSITES:bb_.+]]:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, two_callee-.
	s_swap_pc_i64 s[12:13], s[10:11]
two_after_first:
; CHECK: [[RETURN1:bb_.+]]:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, two_callee-.
	s_swap_pc_i64 s[12:13], s[10:11]
two_after_second:
; CHECK: [[RETURN2:bb_.+]]:
	s_cvt_f32_u32 s3, s2
	s_branch two_done
two_callee:
	s_mov_b32 s2, 44
; CHECK: switch i64 %{{.+}}, label %[[TWOTRAP:.+]] [
; CHECK-NEXT: i64 {{.+}}, label %[[RETURN1]]
; CHECK-NEXT: i64 {{.+}}, label %[[RETURN2]]
; CHECK-NEXT: ]
	s_set_pc_i64 s[12:13]
two_done:
	s_endpgm
; CHECK: [[TWOTRAP]]:
; CHECK-NEXT: call void @llvm.trap()
; CHECK-NEXT: unreachable

	.globl	swap_dispatch_kernel
	.p2align	8
	.type	swap_dispatch_kernel,@function
; The call reads and writes one register pair, and two paths leave two offsets
; in it. The raise has to read the target before it writes the return address
; over it, or the dispatch reads back the address it just stored.
; CHECK-LABEL: define amdgpu_kernel void @swap_dispatch_kernel(
swap_dispatch_kernel:
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc0 swap_second
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, swap_first-.
	s_branch swap_join
swap_second:
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, swap_second_target-.
swap_join:
; CHECK: switch i64 %{{.+}}, label %{{.+}} [
; CHECK-NEXT: i64 {{.+}}, label %[[SWAPFIRST:bb_.+]]
; CHECK-NEXT: i64 {{.+}}, label %[[SWAPSECOND:bb_.+]]
; CHECK-NEXT: ]
	s_swap_pc_i64 s[10:11], s[10:11]
swap_first:
; CHECK: [[SWAPFIRST]]:
	s_mov_b32 s2, 55
swap_second_target:
; CHECK: [[SWAPSECOND]]:
	s_cvt_f32_u32 s3, s2
	s_endpgm

	.globl	dispatch_atcap_kernel
	.p2align	8
	.type	dispatch_atcap_kernel,@function
; Exactly as many source offsets reach the jump as it may enumerate, so the cap
; still lets it dispatch. One more refuses it, which dispatch_overcap_kernel
; below covers.
; CHECK-LABEL: define amdgpu_kernel void @dispatch_atcap_kernel(
dispatch_atcap_kernel:
	.set atcap_index, 0
	.rept 16
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, atcap_join + 4 * atcap_index - .
; EXEC rather than SCC, which the add above leaves holding a carry the loader
; can change.
	s_cbranch_execz atcap_join
	.set atcap_index, atcap_index + 1
	.endr
atcap_join:
; CHECK: switch i64 %{{.+}}, label %{{.+}} [
; CHECK-COUNT-16: i64 {{.+}}, label %bb_
; CHECK-NEXT: ]
	s_set_pc_i64 s[10:11]
	.rept 16
	s_nop 0
	.endr
	s_endpgm

	.globl	dispatch_partial_kernel
	.p2align	8
	.type	dispatch_partial_kernel,@function
dispatch_partial_kernel:
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc0 partial_else
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, partial_target-.
	s_branch partial_join
partial_else:
; One path leaves an offset in the register pair and the other leaves something
; the analysis cannot name. A jump reading that is refused rather than narrowed
; to the path that did name an offset.
	s_mov_b64 s[10:11], 0
partial_join:
; REFUSE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[10:11], which some path writes without computing a source offset in it
	s_set_pc_i64 s[10:11]
partial_target:
	s_cvt_f32_u32 s3, s2
	s_endpgm

	.globl	dispatch_overcap_kernel
	.p2align	8
	.type	dispatch_overcap_kernel,@function
dispatch_overcap_kernel:
; Every one of these blocks leaves a different source offset in the register
; pair, which is one more offset than the jump below may enumerate. Dispatching
; on the ones that fit would send the paths that did not to a block they never
; reach, so the jump is refused instead.
	.set overcap_index, 0
	.rept 17
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, overcap_join + 4 * overcap_index - .
	s_cbranch_execz overcap_join
	.set overcap_index, overcap_index + 1
	.endr
overcap_join:
; REFUSE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[10:11], which more source offsets reach than the analysis enumerates
	s_set_pc_i64 s[10:11]
	.rept 17
	s_nop 0
	.endr
	s_endpgm

	.globl	capture_clobber_kernel
	.p2align	8
	.type	capture_clobber_kernel,@function
capture_clobber_kernel:
; The paths into the join block leave an offset in the register pair, and then
; the block writes over it with a value that names no offset at all. What the
; block writes is what the jump reads, so the jump is refused rather than
; branching to what the paths left.
	s_get_pc_i64 s[12:13]
	s_add_u32 s12, s12, clobber_target-.
	s_cbranch_execz clobber_join
clobber_join:
	s_mov_b32 s12, 0
; REFUSE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[12:13], which some path writes without computing a source offset in it
	s_set_pc_i64 s[12:13]
clobber_target:
	s_cvt_f32_u32 s3, s2
	s_endpgm

	.globl	undecoded_kernel
	.p2align	8
	.type	undecoded_kernel,@function
undecoded_kernel:
; The displacement lands two bytes into the instruction that follows the
; capture, so the offset the paths leave in the pair is not one any decoded
; instruction starts at.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, 2
	s_cbranch_execz undecoded_join
undecoded_join:
; REFUSE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reaches source offset 0x{{.+}}, which no decoded instruction starts at
	s_set_pc_i64 s[10:11]
	s_endpgm

	.globl	unreachable_kernel
	.p2align	8
	.type	unreachable_kernel,@function
unreachable_kernel:
; A jump ends its block, so the instructions after it lead a block of their own
; that nothing falls into. The second jump sits in that block, and no path the
; decode recovered says what its register pair holds there.
	s_get_pc_i64 s[10:11]
	s_add_u32 s10, s10, unreachable_target-.
	s_set_pc_i64 s[10:11]
; REFUSE: unsupported-instruction-form: s_set_pc_i64 {{.+}} :: reads s[12:13], and no path the decode recovered reaches its block
	s_set_pc_i64 s[12:13]
unreachable_target:
	s_cvt_f32_u32 s3, s2
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel dispatch_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel call_return_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel two_call_sites_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel swap_dispatch_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel dispatch_atcap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel dispatch_partial_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel dispatch_overcap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel capture_clobber_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel undecoded_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 24
	.end_amdhsa_kernel
	.amdhsa_kernel unreachable_kernel
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
    .name:           dispatch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           call_return_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         call_return_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           two_call_sites_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         two_call_sites_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           swap_dispatch_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         swap_dispatch_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dispatch_atcap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_atcap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dispatch_partial_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_partial_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           dispatch_overcap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         dispatch_overcap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           capture_clobber_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         capture_clobber_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           undecoded_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         undecoded_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           unreachable_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .symbol:         unreachable_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
