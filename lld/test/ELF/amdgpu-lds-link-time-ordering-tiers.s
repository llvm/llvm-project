# REQUIRES: amdgpu

## Test that shared-tier ordering is optimized while kernel-tier variables
## are correctly appended after the shared tier.
##
## Variables:
##   shared_a: 8 bytes align 4, external, used by K1, K2, K3  (shared, 3 users)
##   shared_b: 32 bytes align 4->16*, external, used by K1, K2 (shared, 2 users)
##   k1_priv:  16 bytes align 4->16*, internal, used by K1     (kernel tier)
##   (* superAlignLDSGlobals bumps alignment to 16 for vars >= 16 bytes)
##
## Greedy shared tier from tail:
##   Position 2: use_count shared_a=3, shared_b=2. Min=2 -> shared_b.
##     K1,K2 fixed. shared_a->1.
##   Position 1: shared_a left.
##   Result: [shared_a, shared_b]
##
## Layout:
##   Shared: shared_a(8,a4)@0x00, shared_b(32,a16)@0x10
##   Kernel: __amdgpu_lds.K1(16,a16)@0x30
##
## Per-kernel sizes:
##   K1: shared_a(0..8), shared_b(16..48), K1's struct(48..64) -> 64 = 0x40
##   K2: shared_a(0..8), shared_b(16..48) -> 48 = 0x30
##   K3: shared_a(0..8) -> 8 = 0x08 (zero waste!)

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: ld.lld %t/tu1.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## Shared tier: shared_a first (most shared), shared_b after
# SYM-DAG: 0000000000000000 {{.*}} shared_a
# SYM-DAG: 0000000000000010 {{.*}} shared_b
## Kernel tier appended after shared tier
# SYM-DAG: 0000000000000030 {{.*}} __amdgpu_lds.K1

## K3 (uses only shared_a) has minimal size
# META-DAG: .group_segment_fixed_size: 8
## K2 (uses shared_a + shared_b)
# META-DAG: .group_segment_fixed_size: 48
## K1 (uses shared_a + shared_b + k1_priv)
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
		.amdgpu_use __amdgpu_lds.K1
		.amdgpu_use shared_b
		.amdgpu_use shared_a
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use shared_b
		.amdgpu_use shared_a
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use shared_a
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	shared_a
	.amdgpu_lds shared_a, 8, 8
	.globl	shared_b
	.amdgpu_lds shared_b, 32, 16
	.globl	__amdgpu_lds.K1
	.amdgpu_lds __amdgpu_lds.K1, 16, 16
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
