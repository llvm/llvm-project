# REQUIRES: amdgpu

## Test per-kernel LDS frontiers. P is allocated first because it has the most
## users. L can still start at offset 0 because none of P's users use L. M then
## lands at the max frontier of K1 and K2.
##
##   K1: P, M
##   K2: M, L
##   K3: L
##   K4: P
##   K5: P

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

# SYM-DAG: 0000000000000000 {{.*}} P
# SYM-DAG: 0000000000000000 {{.*}} L
# SYM-DAG: 0000000000000010 {{.*}} M

	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
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
	.short	0
	.short	0
	.long	0
	.endm

	kernel K1
	kernel K2
	kernel K3
	kernel K4
	kernel K5

	.amdgpu_info K1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use P
		.amdgpu_use M
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use M
		.amdgpu_use L
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use L
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K4
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use P
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K5
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use P
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	P
	.amdgpu_lds P, 8, 8
	.globl	L
	.amdgpu_lds L, 16, 16
	.globl	M
	.amdgpu_lds M, 4, 4
