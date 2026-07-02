# REQUIRES: amdgpu

## Test that the linker computes LDS offsets and per-kernel LDS sizes for
## disjoint kernel groups. This is the focused test for patching each kernel's
## group_segment_fixed_size from the resolved LDS layout.
##
## TU1: kernel_a uses lds_a (256 bytes, align 16)
## TU2: kernel_b uses lds_b (128 bytes, align 4)
##
## These are independent groups, so each allocates from offset 0:
##   Group 1: lds_a at offset 0 -> kernel_a size = 0x100
##   Group 2: lds_b at offset 0 -> kernel_b size = 0x080

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu2.s -o %t/tu2.o

## Link.
# RUN: ld.lld %t/tu1.o %t/tu2.o -o %t/out

## Verify independent groups both allocate from offset 0.
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM

## Verify per-kernel LDS sizes via kernel descriptor patching and HSA metadata.
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

# SYM-DAG: 0000000000000000 {{.*}} lds_a
# SYM-DAG: 0000000000000000 {{.*}} lds_b

## kernel_a: lds_a at offset 0, size 256 -> group_segment_fixed_size = 256
# META:      .group_segment_fixed_size: 256
# META:      .name:           kernel_a

## kernel_b: lds_b at offset 0, size 128 -> group_segment_fixed_size = 128
# META:      .group_segment_fixed_size: 128
# META:      .name:           kernel_b

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel_a                        ; -- Begin function kernel_a
	.p2align	8
	.type	kernel_a,@function
kernel_a:
	s_endpgm
.Lfunc_end0:
	.size	kernel_a, .Lfunc_end0-kernel_a
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel_a.kd
	.type	kernel_a.kd,@object
	.size	kernel_a.kd, 64
	.protected	kernel_a
kernel_a.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	kernel_a@rel64-kernel_a.kd
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
	.long	11468864
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.set .Lkernel_a.num_vgpr, 2
	.set .Lkernel_a.num_agpr, 0
	.set .Lkernel_a.numbered_sgpr, 10
	.set .Lkernel_a.num_named_barrier, 0
	.set .Lkernel_a.private_seg_size, 0
	.set .Lkernel_a.uses_vcc, 0
	.set .Lkernel_a.uses_flat_scratch, 0
	.set .Lkernel_a.has_dyn_sized_stack, 0
	.set .Lkernel_a.has_recursion, 0
	.set .Lkernel_a.has_indirect_call, 0
	.amdgpu_info kernel_a
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_a
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_a
	.amdgpu_lds lds_a, 256, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           kernel_a
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kernel_a.kd
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

#--- tu2.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel_b                        ; -- Begin function kernel_b
	.p2align	8
	.type	kernel_b,@function
kernel_b:
	s_endpgm
.Lfunc_end0:
	.size	kernel_b, .Lfunc_end0-kernel_b
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel_b.kd
	.type	kernel_b.kd,@object
	.size	kernel_b.kd, 64
	.protected	kernel_b
kernel_b.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	kernel_b@rel64-kernel_b.kd
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
	.long	11468864
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.set .Lkernel_b.num_vgpr, 2
	.set .Lkernel_b.num_agpr, 0
	.set .Lkernel_b.numbered_sgpr, 10
	.set .Lkernel_b.num_named_barrier, 0
	.set .Lkernel_b.private_seg_size, 0
	.set .Lkernel_b.uses_vcc, 0
	.set .Lkernel_b.uses_flat_scratch, 0
	.set .Lkernel_b.has_dyn_sized_stack, 0
	.set .Lkernel_b.has_recursion, 0
	.set .Lkernel_b.has_indirect_call, 0
	.amdgpu_info kernel_b
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_b
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
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
    .name:           kernel_b
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kernel_b.kd
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
