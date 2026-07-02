# REQUIRES: amdgpu

## Test indirect call with multiple potential callees. Two address-taken
## functions have the same prototype. Both should be considered potential
## callees of the indirect call site, and LDS from both should be reachable.
##
## Call graph:
##   my_kernel -> caller --(indirect)--> {target_a, target_b}
##   target_a -> lds_a (64 bytes)
##   target_b -> lds_b (32 bytes)
##
## Both targets: void(i32) -> encoding "vi"

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: ld.lld %t/tu1.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## Both LDS variables should be resolved.
# SYM-DAG: {{[0-9a-f]+}} {{.*}} lds_a
# SYM-DAG: {{[0-9a-f]+}} {{.*}} lds_b

## Kernel's LDS size should include both lds_a (256 bytes) and lds_b (128 bytes).
## lds_a: align=4, size=256 -> offset 0, end 256
## lds_b: align=4, size=128 -> offset 256, end 384
# META: .group_segment_fixed_size: 384

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	target_a                        ; -- Begin function target_a
	.p2align	6
	.type	target_a,@function
target_a:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	target_a, .Lfunc_end0-target_a
	.set .Ltarget_a.num_vgpr, 2
	.set .Ltarget_a.num_agpr, 0
	.set .Ltarget_a.numbered_sgpr, 32
	.set .Ltarget_a.num_named_barrier, 0
	.set .Ltarget_a.private_seg_size, 0
	.set .Ltarget_a.uses_vcc, 0
	.set .Ltarget_a.uses_flat_scratch, 0
	.set .Ltarget_a.has_dyn_sized_stack, 0
	.set .Ltarget_a.has_recursion, 0
	.set .Ltarget_a.has_indirect_call, 0
	.text
	.globl	target_b                        ; -- Begin function target_b
	.p2align	6
	.type	target_b,@function
target_b:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	target_b, .Lfunc_end1-target_b
	.set .Ltarget_b.num_vgpr, 2
	.set .Ltarget_b.num_agpr, 0
	.set .Ltarget_b.numbered_sgpr, 32
	.set .Ltarget_b.num_named_barrier, 0
	.set .Ltarget_b.private_seg_size, 0
	.set .Ltarget_b.uses_vcc, 0
	.set .Ltarget_b.uses_flat_scratch, 0
	.set .Ltarget_b.has_dyn_sized_stack, 0
	.set .Ltarget_b.has_recursion, 0
	.set .Ltarget_b.has_indirect_call, 0
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
	.globl	my_kernel                       ; -- Begin function my_kernel
	.p2align	8
	.type	my_kernel,@function
my_kernel:
	s_endpgm
.Lfunc_end3:
	.size	my_kernel, .Lfunc_end3-my_kernel
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
	.set .Lmy_kernel.numbered_sgpr, 88
	.set .Lmy_kernel.num_named_barrier, 0
	.set .Lmy_kernel.private_seg_size, 0
	.set .Lmy_kernel.uses_vcc, 1
	.set .Lmy_kernel.uses_flat_scratch, 1
	.set .Lmy_kernel.has_dyn_sized_stack, 0
	.set .Lmy_kernel.has_recursion, 1
	.set .Lmy_kernel.has_indirect_call, 0
	.amdgpu_info target_a
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_a
		.amdgpu_typeid "vi"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info target_b
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_b
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

	.amdgpu_info my_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 42
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 88
		.amdgpu_private_segment_size 0
		.amdgpu_call caller
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 41
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 66
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_a
	.amdgpu_lds lds_a, 256, 16
	.globl	lds_b
	.amdgpu_lds lds_b, 128, 16
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
