# REQUIRES: amdgpu

## Test cross-TU address-taken: the function is defined in TU1 but its address
## is taken in TU2. The linker should match the indirect call in TU2 with the
## function from TU1 via prototype matching.
##
## TU1: defines target_func (uses lds_var)
## TU2: defines kern which passes target_func as a pointer to caller
##      defines caller which makes an indirect call

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu2.s -o %t/tu2.o
# RUN: ld.lld %t/tu1.o %t/tu2.o -o %t/out
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## Kernel LDS size should include lds_var (128 bytes).
# META: .group_segment_fixed_size: 128

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	target_func                     ; -- Begin function target_func
	.p2align	6
	.type	target_func,@function
target_func:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	target_func, .Lfunc_end0-target_func
	.set .Ltarget_func.num_vgpr, 2
	.set .Ltarget_func.num_agpr, 0
	.set .Ltarget_func.numbered_sgpr, 32
	.set .Ltarget_func.num_named_barrier, 0
	.set .Ltarget_func.private_seg_size, 0
	.set .Ltarget_func.uses_vcc, 0
	.set .Ltarget_func.uses_flat_scratch, 0
	.set .Ltarget_func.has_dyn_sized_stack, 0
	.set .Ltarget_func.has_recursion, 0
	.set .Ltarget_func.has_indirect_call, 0
	.amdgpu_info target_func
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_var
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 2
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_var
	.amdgpu_lds lds_var, 128, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- tu2.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	caller                          ; -- Begin function caller
	.p2align	6
	.type	caller,@function
caller:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	caller, .Lfunc_end0-caller
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
.Lfunc_end1:
	.size	kern, .Lfunc_end1-kern
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

	.amdgpu_info target_func
		.amdgpu_typeid "vi"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 41
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 66
	.set amdgpu.max_num_named_barrier, 0
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
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         kern.kd
    .uses_dynamic_stack: true
    .vgpr_count:     32
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
