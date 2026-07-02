# REQUIRES: amdgpu

## Test nested indirect calls (C++ virtual method pattern): an address-taken
## function itself makes an indirect call to another function. Both functions
## use LDS. The linker must discover all LDS through the indirect call chain
## and assign sequential non-overlapping offsets.
##
## Call graph:
##   kern -> caller --(indirect)--> outer_target --(indirect)--> inner_target
##   outer_target -> lds_outer (128 bytes)
##   inner_target -> lds_inner (64 bytes)
##
## Both indirect calls: void(i32) -> encoding "vi"
## Both targets are address-taken with encoding "vi"
##
## All LDS must be assigned unique offsets (no slot reuse).

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: ld.lld %t/tu1.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## Both LDS variables resolved with non-overlapping offsets.
# SYM-DAG: {{[0-9a-f]+}} {{.*}} lds_outer
# SYM-DAG: {{[0-9a-f]+}} {{.*}} lds_inner

## Kernel LDS size includes both: lds_outer (128) + lds_inner (64) = 192.
# META: .group_segment_fixed_size: 192

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	inner_target                    ; -- Begin function inner_target
	.p2align	6
	.type	inner_target,@function
inner_target:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	inner_target, .Lfunc_end0-inner_target
	.set .Linner_target.num_vgpr, 2
	.set .Linner_target.num_agpr, 0
	.set .Linner_target.numbered_sgpr, 32
	.set .Linner_target.num_named_barrier, 0
	.set .Linner_target.private_seg_size, 0
	.set .Linner_target.uses_vcc, 0
	.set .Linner_target.uses_flat_scratch, 0
	.set .Linner_target.has_dyn_sized_stack, 0
	.set .Linner_target.has_recursion, 0
	.set .Linner_target.has_indirect_call, 0
	.text
	.globl	dispatch                        ; -- Begin function dispatch
	.p2align	6
	.type	dispatch,@function
dispatch:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	dispatch, .Lfunc_end1-dispatch
	.set .Ldispatch.num_vgpr, 41
	.set .Ldispatch.num_agpr, 0
	.set .Ldispatch.numbered_sgpr, 66
	.set .Ldispatch.num_named_barrier, 0
	.set .Ldispatch.private_seg_size, 16
	.set .Ldispatch.uses_vcc, 1
	.set .Ldispatch.uses_flat_scratch, 1
	.set .Ldispatch.has_dyn_sized_stack, 1
	.set .Ldispatch.has_recursion, 1
	.set .Ldispatch.has_indirect_call, 1
	.text
	.globl	outer_target                    ; -- Begin function outer_target
	.p2align	6
	.type	outer_target,@function
outer_target:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	outer_target, .Lfunc_end2-outer_target
	.set .Louter_target.num_vgpr, 42
	.set .Louter_target.num_agpr, 0
	.set .Louter_target.numbered_sgpr, 66
	.set .Louter_target.num_named_barrier, 0
	.set .Louter_target.private_seg_size, 16
	.set .Louter_target.uses_vcc, 1
	.set .Louter_target.uses_flat_scratch, 0
	.set .Louter_target.has_dyn_sized_stack, 0
	.set .Louter_target.has_recursion, 1
	.set .Louter_target.has_indirect_call, 0
	.text
	.globl	caller                          ; -- Begin function caller
	.p2align	6
	.type	caller,@function
caller:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end3:
	.size	caller, .Lfunc_end3-caller
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
	.globl	kern                            ; -- Begin function kern
	.p2align	8
	.type	kern,@function
kern:
	s_endpgm
.Lfunc_end4:
	.size	kern, .Lfunc_end4-kern
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kern.kd
	.type	kern.kd,@object
	.size	kern.kd, 64
	.protected	kern
kern.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	kern@rel64-kern.kd
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
	.long	11469063
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.set .Lkern.num_vgpr, 32
	.set .Lkern.num_agpr, 0
	.set .Lkern.numbered_sgpr, 33
	.set .Lkern.num_named_barrier, 0
	.set .Lkern.private_seg_size, 0
	.set .Lkern.uses_vcc, 1
	.set .Lkern.uses_flat_scratch, 1
	.set .Lkern.has_dyn_sized_stack, 0
	.set .Lkern.has_recursion, 1
	.set .Lkern.has_indirect_call, 0
	.amdgpu_info inner_target
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_inner
		.amdgpu_typeid "vi"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info dispatch
		.amdgpu_flags 7
		.amdgpu_num_vgpr 41
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 66
		.amdgpu_private_segment_size 16
		.amdgpu_indirect_call "vi"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info outer_target
		.amdgpu_flags 1
		.amdgpu_num_vgpr 42
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 66
		.amdgpu_private_segment_size 16
		.amdgpu_use lds_outer
		.amdgpu_call dispatch
		.amdgpu_typeid "vi"
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

	.amdgpu_info kern
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_call caller
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 42
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 66
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_outer
	.amdgpu_lds lds_outer, 128, 16
	.globl	lds_inner
	.amdgpu_lds lds_inner, 64, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           kern
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kern.kd
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
