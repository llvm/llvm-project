# REQUIRES: amdgpu

## Test LDS discovery through an indirect call, with prototype-based filtering.
## Only address-taken functions matching the indirect call's prototype should be
## considered as potential callees.
##
## target_match: void(i32) -> encoding "vi" -- matches the indirect call
## target_nomatch: i32(i32, i32) -> encoding "iii" -- does NOT match
##
## Only lds_match should be reachable from the kernel through the indirect edge.
## lds_nomatch should NOT be reachable (its function has a different prototype).

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: ld.lld %t/tu1.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## The matching target's LDS is discovered through the indirect call edge.
# SYM: 0000000000000000 {{.*}} lds_match

## Kernel's LDS size should only include lds_match (128 bytes), not lds_nomatch.
# META: .group_segment_fixed_size: 128

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	target_match                    ; -- Begin function target_match
	.p2align	6
	.type	target_match,@function
target_match:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	target_match, .Lfunc_end0-target_match
	.set .Ltarget_match.num_vgpr, 2
	.set .Ltarget_match.num_agpr, 0
	.set .Ltarget_match.numbered_sgpr, 32
	.set .Ltarget_match.num_named_barrier, 0
	.set .Ltarget_match.private_seg_size, 0
	.set .Ltarget_match.uses_vcc, 0
	.set .Ltarget_match.uses_flat_scratch, 0
	.set .Ltarget_match.has_dyn_sized_stack, 0
	.set .Ltarget_match.has_recursion, 0
	.set .Ltarget_match.has_indirect_call, 0
	.text
	.globl	target_nomatch                  ; -- Begin function target_nomatch
	.p2align	6
	.type	target_nomatch,@function
target_nomatch:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	target_nomatch, .Lfunc_end1-target_nomatch
	.set .Ltarget_nomatch.num_vgpr, 3
	.set .Ltarget_nomatch.num_agpr, 0
	.set .Ltarget_nomatch.numbered_sgpr, 32
	.set .Ltarget_nomatch.num_named_barrier, 0
	.set .Ltarget_nomatch.private_seg_size, 0
	.set .Ltarget_nomatch.uses_vcc, 0
	.set .Ltarget_nomatch.uses_flat_scratch, 0
	.set .Ltarget_nomatch.has_dyn_sized_stack, 0
	.set .Ltarget_nomatch.has_recursion, 0
	.set .Ltarget_nomatch.has_indirect_call, 0
	.text
	.globl	caller                          ; -- Begin function caller
	.p2align	6
	.type	caller,@function
caller:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	caller, .Lfunc_end2-caller
	.set .Lcaller.num_vgpr, 41
	.set .Lcaller.num_agpr, 0
	.set .Lcaller.numbered_sgpr, 66
	.set .Lcaller.num_named_barrier, 0
	.set .Lcaller.private_seg_size, 16
	.set .Lcaller.uses_vcc, 1
	.set .Lcaller.uses_flat_scratch, 1
	.set .Lcaller.has_dyn_sized_stack, 1
	.set .Lcaller.has_recursion, 1
	.set .Lcaller.has_indirect_call, 1
	.text
	.globl	addr_taker                      ; -- Begin function addr_taker
	.p2align	6
	.type	addr_taker,@function
addr_taker:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end3:
	.size	addr_taker, .Lfunc_end3-addr_taker
	.set .Laddr_taker.num_vgpr, 2
	.set .Laddr_taker.num_agpr, 0
	.set .Laddr_taker.numbered_sgpr, 33
	.set .Laddr_taker.num_named_barrier, 0
	.set .Laddr_taker.private_seg_size, 16
	.set .Laddr_taker.uses_vcc, 0
	.set .Laddr_taker.uses_flat_scratch, 0
	.set .Laddr_taker.has_dyn_sized_stack, 0
	.set .Laddr_taker.has_recursion, 0
	.set .Laddr_taker.has_indirect_call, 0
	.text
	.globl	my_kernel                       ; -- Begin function my_kernel
	.p2align	8
	.type	my_kernel,@function
my_kernel:
	s_endpgm
.Lfunc_end4:
	.size	my_kernel, .Lfunc_end4-my_kernel
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	my_kernel.kd
	.type	my_kernel.kd,@object
	.size	my_kernel.kd, 64
	.protected	my_kernel
my_kernel.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	my_kernel@rel64-my_kernel.kd
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.long	0
	.long	11469514
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.set .Lmy_kernel.num_vgpr, 42
	.set .Lmy_kernel.num_agpr, 0
	.set .Lmy_kernel.numbered_sgpr, 85
	.set .Lmy_kernel.num_named_barrier, 0
	.set .Lmy_kernel.private_seg_size, 0
	.set .Lmy_kernel.uses_vcc, 1
	.set .Lmy_kernel.uses_flat_scratch, 1
	.set .Lmy_kernel.has_dyn_sized_stack, 0
	.set .Lmy_kernel.has_recursion, 1
	.set .Lmy_kernel.has_indirect_call, 0
	.amdgpu_info target_match
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_match
		.amdgpu_typeid "vi"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info target_nomatch
		.amdgpu_flags 0
		.amdgpu_num_vgpr 3
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_nomatch
		.amdgpu_typeid "iii"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info caller
		.amdgpu_flags 7
		.amdgpu_num_vgpr 41
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 66
		.amdgpu_private_segment_size 16
		.amdgpu_indirect_call "vi"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info addr_taker
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 16
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info my_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 42
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 85
		.amdgpu_private_segment_size 0
		.amdgpu_call caller
		.amdgpu_call addr_taker
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 41
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 66
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_match
	.amdgpu_lds lds_match, 128, 16
	.globl	lds_nomatch
	.amdgpu_lds lds_nomatch, 256, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           my_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         my_kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
