# REQUIRES: amdgpu

## Test that usage frequency drives shared-tier placement. The variable used
## by the most kernels should be placed at the lowest offset regardless of
## its size.
##
## Variables (all external linkage -> global-scope -> shared tier):
##   S1: 16 bytes align 4->16*, used by K1, K2     (2 users)
##   S2: 8 bytes align 4, used by K1, K2, K3       (3 users, smallest!)
##   S3: 32 bytes align 4->16*, used by K1, K2     (2 users)
##   (* superAlignLDSGlobals bumps alignment to 16 for vars >= 16 bytes)
##
## Per-kernel frontier order:
##   S2 has the most users and is placed first. S3 and S1 have equal use
##   counts, so S3 is placed before S1 by the size/alignment tie-breaker.
##
## Layout: S2(8,a4)@0x00, S3(32,a16)@0x10, S1(16,a16)@0x30
##
## Per-kernel sizes:
##   K1: reaches all -> 64 = 0x40
##   K2: same -> 64 = 0x40
##   K3: reaches S2(0..8) -> 8 = 0x08 (zero waste!)
##
## A naive size-desc sort [S3,S1,S2] would give K3 size = 64 (waste = 56).

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: ld.lld %t/tu1.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## S2 (most shared, 3 users) at lowest offset despite being smallest
# SYM-DAG: 0000000000000000 {{.*}} S2
# SYM-DAG: 0000000000000010 {{.*}} S3
# SYM-DAG: 0000000000000030 {{.*}} S1

## K3 uses only S2: minimal size
# META-DAG: .group_segment_fixed_size: 8
## K1 and K2 use all three
# META-DAG: .group_segment_fixed_size: 64
# META-DAG: .group_segment_fixed_size: 64

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	K1                              ; -- Begin function K1
	.p2align	8
	.type	K1,@function
K1:
	s_endpgm
.Lfunc_end0:
	.size	K1, .Lfunc_end0-K1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K1.kd
	.type	K1.kd,@object
	.size	K1.kd, 64
	.protected	K1
K1.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K1@rel64-K1.kd
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
	.set .LK1.num_vgpr, 4
	.set .LK1.num_agpr, 0
	.set .LK1.numbered_sgpr, 10
	.set .LK1.num_named_barrier, 0
	.set .LK1.private_seg_size, 0
	.set .LK1.uses_vcc, 0
	.set .LK1.uses_flat_scratch, 0
	.set .LK1.has_dyn_sized_stack, 0
	.set .LK1.has_recursion, 0
	.set .LK1.has_indirect_call, 0
	.text
	.globl	K2                              ; -- Begin function K2
	.p2align	8
	.type	K2,@function
K2:
	s_endpgm
.Lfunc_end1:
	.size	K2, .Lfunc_end1-K2
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K2.kd
	.type	K2.kd,@object
	.size	K2.kd, 64
	.protected	K2
K2.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K2@rel64-K2.kd
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
	.set .LK2.num_vgpr, 4
	.set .LK2.num_agpr, 0
	.set .LK2.numbered_sgpr, 10
	.set .LK2.num_named_barrier, 0
	.set .LK2.private_seg_size, 0
	.set .LK2.uses_vcc, 0
	.set .LK2.uses_flat_scratch, 0
	.set .LK2.has_dyn_sized_stack, 0
	.set .LK2.has_recursion, 0
	.set .LK2.has_indirect_call, 0
	.text
	.globl	K3                              ; -- Begin function K3
	.p2align	8
	.type	K3,@function
K3:
	s_endpgm
.Lfunc_end2:
	.size	K3, .Lfunc_end2-K3
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K3.kd
	.type	K3.kd,@object
	.size	K3.kd, 64
	.protected	K3
K3.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K3@rel64-K3.kd
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
	.set .LK3.num_vgpr, 2
	.set .LK3.num_agpr, 0
	.set .LK3.numbered_sgpr, 10
	.set .LK3.num_named_barrier, 0
	.set .LK3.private_seg_size, 0
	.set .LK3.uses_vcc, 0
	.set .LK3.uses_flat_scratch, 0
	.set .LK3.has_dyn_sized_stack, 0
	.set .LK3.has_recursion, 0
	.set .LK3.has_indirect_call, 0
	.amdgpu_info K1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use S3
		.amdgpu_use S2
		.amdgpu_use S1
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.amdgpu_info K2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use S3
		.amdgpu_use S2
		.amdgpu_use S1
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.amdgpu_info K3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use S2
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	S1
	.amdgpu_lds S1, 16, 16
	.globl	S2
	.amdgpu_lds S2, 8, 8
	.globl	S3
	.amdgpu_lds S3, 32, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K1
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K1.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K2
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K2.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K3
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K3.kd
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
