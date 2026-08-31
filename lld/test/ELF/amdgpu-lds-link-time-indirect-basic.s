# REQUIRES: amdgpu

## Test basic indirect call LDS resolution: kernel calls a function that makes
## an indirect call to an LDS-using function. The linker should discover the
## LDS variable through the indirect call edge (prototype matching) and assign
## it an offset. The kernel's LDS size should include the indirectly-reachable
## LDS.
##
## Call graph: my_kernel -> indirect_caller --(indirect)--> target_func -> lds_var
##
## target_func: void(i32) -> prototype encoding "vi"
## indirect_caller: makes indirect call with encoding "vi"
## target_func: address-taken (passed as ptr to indirect_caller)

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: ld.lld %t/tu1.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## LDS variable should be resolved with an offset.
# SYM: 0000000000000000 {{.*}} lds_var

## Kernel descriptor should have non-zero LDS size (128 bytes for [32 x i32]).
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
	.text
	.globl	indirect_caller                 ; -- Begin function indirect_caller
	.p2align	6
	.type	indirect_caller,@function
indirect_caller:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	indirect_caller, .Lfunc_end1-indirect_caller
	.set .Lindirect_caller.num_vgpr, 41
	.set .Lindirect_caller.num_agpr, 0
	.set .Lindirect_caller.numbered_sgpr, 66
	.set .Lindirect_caller.num_named_barrier, 0
	.set .Lindirect_caller.private_seg_size, 16
	.set .Lindirect_caller.uses_vcc, 1
	.set .Lindirect_caller.uses_flat_scratch, 1
	.set .Lindirect_caller.has_dyn_sized_stack, 1
	.set .Lindirect_caller.has_recursion, 1
	.set .Lindirect_caller.has_indirect_call, 1
	.text
	.globl	my_kernel                       ; -- Begin function my_kernel
	.p2align	8
	.type	my_kernel,@function
my_kernel:
	s_endpgm
.Lfunc_end2:
	.size	my_kernel, .Lfunc_end2-my_kernel
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
	.long	11469063
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.set .Lmy_kernel.num_vgpr, 32
	.set .Lmy_kernel.num_agpr, 0
	.set .Lmy_kernel.numbered_sgpr, 33
	.set .Lmy_kernel.num_named_barrier, 0
	.set .Lmy_kernel.private_seg_size, 0
	.set .Lmy_kernel.uses_vcc, 1
	.set .Lmy_kernel.uses_flat_scratch, 1
	.set .Lmy_kernel.has_dyn_sized_stack, 0
	.set .Lmy_kernel.has_recursion, 1
	.set .Lmy_kernel.has_indirect_call, 0
	.amdgpu_info target_func
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_var
		.amdgpu_typeid "vi"
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.amdgpu_info indirect_caller
		.amdgpu_flags 7
		.amdgpu_num_vgpr 41
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 66
		.amdgpu_private_segment_size 16
		.amdgpu_indirect_call "vi"
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.amdgpu_info my_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_call indirect_caller
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 41
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 66
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_var
	.amdgpu_lds lds_var, 128, 16
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
