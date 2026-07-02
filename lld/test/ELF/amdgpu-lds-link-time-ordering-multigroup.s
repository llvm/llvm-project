# REQUIRES: amdgpu

## Test that multiple independent groups each have their shared-tier ordering
## optimized independently, and each group allocates from offset 0.
##
## Group 1 (K1, K2 share A and B):
##   A: 8 bytes align 4, used by K1, K2, K3     (3 users)
##   B: 16 bytes align 4->16*, used by K1, K2   (2 users)
##   C: 32 bytes align 4->16*, used by K1, K2   (2 users)
##
## Group 2 (K4, K5 share D):
##   D: 4 bytes align 4, used by K4, K5  (2 users)
##   E: 8 bytes align 4, used by K4, K5  (2 users)
##
## Groups are independent (no shared LDS between them).
##
## Group 1 frontier order: [A, C, B] (A@0x00, C@0x10, B@0x30)
##   K3 uses only A -> size = 8
##
## Group 2 frontier order: both D and E have use_count=2.
##   Tie-break by size: E(8)>D(4), so E is placed first.
##   Result: [E, D] (E@0x00, D@0x08)
##   (* superAlignLDSGlobals bumps alignment to 16 for vars >= 16 bytes,
##      and E to align 8 since it is 8 bytes)

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu2.s -o %t/tu2.o
# RUN: ld.lld %t/tu1.o %t/tu2.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## Group 1: A most shared, at offset 0
# SYM-DAG: 0000000000000000 {{.*}} A
# SYM-DAG: 0000000000000010 {{.*}} C
# SYM-DAG: 0000000000000030 {{.*}} B

## Group 2 is independent and starts from offset 0
# SYM-DAG: 0000000000000000 {{.*}} E
# SYM-DAG: 0000000000000008 {{.*}} D

## K3 uses only A -> 8 bytes
# META-DAG: .group_segment_fixed_size: 8
## K1 and K2 use all of group 1 -> 64 bytes
# META-DAG: .group_segment_fixed_size: 64
# META-DAG: .group_segment_fixed_size: 64
## K4 and K5 use all of group 2 -> 12 bytes (E@0..8, D@8..12)
# META-DAG: .group_segment_fixed_size: 12
# META-DAG: .group_segment_fixed_size: 12

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
		.amdgpu_use B
		.amdgpu_use C
		.amdgpu_use A
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use B
		.amdgpu_use C
		.amdgpu_use A
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use A
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	A
	.amdgpu_lds A, 8, 8
	.globl	B
	.amdgpu_lds B, 16, 16
	.globl	C
	.amdgpu_lds C, 32, 16
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

#--- tu2.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	K4                              ; -- Begin function K4
	.p2align	8
	.type	K4,@function
K4:
	s_endpgm
.Lfunc_end0:
	.size	K4, .Lfunc_end0-K4
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K4.kd
	.type	K4.kd,@object
	.size	K4.kd, 64
	.protected	K4
K4.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K4@rel64-K4.kd
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
	.set .LK4.num_vgpr, 2
	.set .LK4.num_agpr, 0
	.set .LK4.numbered_sgpr, 10
	.set .LK4.num_named_barrier, 0
	.set .LK4.private_seg_size, 0
	.set .LK4.uses_vcc, 0
	.set .LK4.uses_flat_scratch, 0
	.set .LK4.has_dyn_sized_stack, 0
	.set .LK4.has_recursion, 0
	.set .LK4.has_indirect_call, 0
	.text
	.globl	K5                              ; -- Begin function K5
	.p2align	8
	.type	K5,@function
K5:
	s_endpgm
.Lfunc_end1:
	.size	K5, .Lfunc_end1-K5
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K5.kd
	.type	K5.kd,@object
	.size	K5.kd, 64
	.protected	K5
K5.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K5@rel64-K5.kd
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
	.set .LK5.num_vgpr, 2
	.set .LK5.num_agpr, 0
	.set .LK5.numbered_sgpr, 10
	.set .LK5.num_named_barrier, 0
	.set .LK5.private_seg_size, 0
	.set .LK5.uses_vcc, 0
	.set .LK5.uses_flat_scratch, 0
	.set .LK5.has_dyn_sized_stack, 0
	.set .LK5.has_recursion, 0
	.set .LK5.has_indirect_call, 0
	.amdgpu_info K4
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use E
		.amdgpu_use D
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K5
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use E
		.amdgpu_use D
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	D
	.amdgpu_lds D, 4, 4
	.globl	E
	.amdgpu_lds E, 8, 8
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K4
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K4.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K5
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K5.kd
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
