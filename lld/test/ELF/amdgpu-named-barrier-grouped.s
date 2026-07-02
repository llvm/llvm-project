# REQUIRES: amdgpu

## Test per-kernel named-barrier frontiers. A and B are allocated first because
## they have the most users. C can still reuse B's IDs because no kernel reaches
## both B and C, even though A and B connect all three kernels into one old-style
## barrier group.
##
##   K1: A, C
##   K2: A, B
##   K3: B, D
##
## With connected-component allocation, C would be assigned after A and B, so K1
## would require 6 named barriers. The frontier allocator places C at ID 3, so
## K1 only requires 4 named barriers and NAMED_BAR_CNT stays 1.

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-nm --format=posix %t | FileCheck %s --check-prefix=SYM
# RUN: llvm-objdump -s -j .rodata %t | FileCheck %s --check-prefix=KD

# SYM-DAG: __amdgpu_named_barrier.a A {{0*}}802010
# SYM-DAG: __amdgpu_named_barrier.b A {{0*}}802030
# SYM-DAG: __amdgpu_named_barrier.c A {{0*}}802030
# SYM-DAG: __amdgpu_named_barrier.d A {{0*}}802050

## K1 and K2 use IDs 1..4, so NAMED_BAR_CNT=1. K3 uses IDs 3..6, so
## NAMED_BAR_CNT=2.
# KD: {{[0-9a-f]+}} 00000000 00000000 00000000 00400000
# KD: {{[0-9a-f]+}} 00000000 00000000 00000000 00400000
# KD: {{[0-9a-f]+}} 00000000 00000000 00000000 00800000

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6

	.macro kernel name
	.text
	.globl	\name
	.p2align	8
	.type	\name,@function
\name:
	s_endpgm
.L\name\()_end:
	.size	\name, .L\name\()_end-\name
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	\name\().kd
	.type	\name\().kd,@object
	.size	\name\().kd, 64
	.protected	\name
\name\().kd:
	.long	0
	.long	0
	.long	0
	.long	0
	.quad	\name@rel64-\name\().kd
	.zero	20
	.long	0
	.long	0
	.long	0
	.short	1024
	.short	0
	.long	0
	.endm

	kernel K1
	kernel K2
	kernel K3

	.amdgpu_info K1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.a
		.amdgpu_use __amdgpu_named_barrier.c
		.amdgpu_occupancy 4
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.amdgpu_info K2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.a
		.amdgpu_use __amdgpu_named_barrier.b
		.amdgpu_occupancy 4
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.amdgpu_info K3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.b
		.amdgpu_use __amdgpu_named_barrier.d
		.amdgpu_occupancy 4
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.globl	__amdgpu_named_barrier.a
	.amdgpu_lds __amdgpu_named_barrier.a, 32, 4
	.globl	__amdgpu_named_barrier.b
	.amdgpu_lds __amdgpu_named_barrier.b, 32, 4
	.globl	__amdgpu_named_barrier.c
	.amdgpu_lds __amdgpu_named_barrier.c, 32, 4
	.globl	__amdgpu_named_barrier.d
	.amdgpu_lds __amdgpu_named_barrier.d, 32, 4
