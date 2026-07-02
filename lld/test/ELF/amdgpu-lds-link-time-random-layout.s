# REQUIRES: amdgpu

## Generated deterministic random LDS/kernel layouts for the per-kernel
## frontier allocator. The generator was run ahead of time; lit does not
## execute any script.

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYM

## Case 0: 2 kernels, 4 LDS symbols, allocation order c0_lds1, c0_lds3, c0_lds2, c0_lds0
##   c0_lds0: size=20, align=4, users={c0_k1}, offset=0x1c
# SYM-DAG: 000000000000001c {{.*}} c0_lds0
##   c0_lds1: size=24, align=4, users={c0_k0, c0_k1}, offset=0x0
# SYM-DAG: 0000000000000000 {{.*}} c0_lds1
##   c0_lds2: size=4, align=8, users={c0_k0}, offset=0x20
# SYM-DAG: 0000000000000020 {{.*}} c0_lds2
##   c0_lds3: size=4, align=4, users={c0_k0, c0_k1}, offset=0x18
# SYM-DAG: 0000000000000018 {{.*}} c0_lds3
## Case 1: 6 kernels, 6 LDS symbols, allocation order c1_lds2, c1_lds5, c1_lds1, c1_lds4, c1_lds3, c1_lds0
##   c1_lds0: size=4, align=8, users={c1_k1, c1_k4}, offset=0x70
# SYM-DAG: 0000000000000070 {{.*}} c1_lds0
##   c1_lds1: size=32, align=4, users={c1_k0, c1_k2, c1_k3, c1_k4}, offset=0x38
# SYM-DAG: 0000000000000038 {{.*}} c1_lds1
##   c1_lds2: size=32, align=16, users={c1_k0, c1_k1, c1_k2, c1_k3, c1_k4, c1_k5}, offset=0x0
# SYM-DAG: 0000000000000000 {{.*}} c1_lds2
##   c1_lds3: size=8, align=4, users={c1_k2, c1_k4, c1_k5}, offset=0x68
# SYM-DAG: 0000000000000068 {{.*}} c1_lds3
##   c1_lds4: size=16, align=8, users={c1_k1, c1_k2, c1_k3}, offset=0x58
# SYM-DAG: 0000000000000058 {{.*}} c1_lds4
##   c1_lds5: size=24, align=4, users={c1_k0, c1_k1, c1_k2, c1_k3, c1_k4, c1_k5}, offset=0x20
# SYM-DAG: 0000000000000020 {{.*}} c1_lds5
## Case 2: 2 kernels, 4 LDS symbols, allocation order c2_lds1, c2_lds2, c2_lds0, c2_lds3
##   c2_lds0: size=16, align=4, users={c2_k0, c2_k1}, offset=0x1c
# SYM-DAG: 000000000000001c {{.*}} c2_lds0
##   c2_lds1: size=12, align=8, users={c2_k0, c2_k1}, offset=0x0
# SYM-DAG: 0000000000000000 {{.*}} c2_lds1
##   c2_lds2: size=12, align=8, users={c2_k0, c2_k1}, offset=0x10
# SYM-DAG: 0000000000000010 {{.*}} c2_lds2
##   c2_lds3: size=4, align=4, users={c2_k0, c2_k1}, offset=0x2c
# SYM-DAG: 000000000000002c {{.*}} c2_lds3
## Case 3: 2 kernels, 5 LDS symbols, allocation order c3_lds1, c3_lds3, c3_lds2, c3_lds4, c3_lds0
##   c3_lds0: size=12, align=4, users={c3_k0}, offset=0x90
# SYM-DAG: 0000000000000090 {{.*}} c3_lds0
##   c3_lds1: size=32, align=16, users={c3_k0, c3_k1}, offset=0x0
# SYM-DAG: 0000000000000000 {{.*}} c3_lds1
##   c3_lds2: size=32, align=4, users={c3_k0, c3_k1}, offset=0x50
# SYM-DAG: 0000000000000050 {{.*}} c3_lds2
##   c3_lds3: size=48, align=8, users={c3_k0, c3_k1}, offset=0x20
# SYM-DAG: 0000000000000020 {{.*}} c3_lds3
##   c3_lds4: size=32, align=4, users={c3_k0, c3_k1}, offset=0x70
# SYM-DAG: 0000000000000070 {{.*}} c3_lds4
## Case 4: 2 kernels, 3 LDS symbols, allocation order c4_lds2, c4_lds0, c4_lds1
##   c4_lds0: size=48, align=16, users={c4_k0}, offset=0x20
# SYM-DAG: 0000000000000020 {{.*}} c4_lds0
##   c4_lds1: size=24, align=8, users={c4_k1}, offset=0x18
# SYM-DAG: 0000000000000018 {{.*}} c4_lds1
##   c4_lds2: size=20, align=8, users={c4_k0, c4_k1}, offset=0x0
# SYM-DAG: 0000000000000000 {{.*}} c4_lds2
## Case 5: 6 kernels, 6 LDS symbols, allocation order c5_lds5, c5_lds3, c5_lds4, c5_lds0, c5_lds1, c5_lds2
##   c5_lds0: size=12, align=8, users={c5_k0, c5_k1, c5_k2, c5_k4}, offset=0x50
# SYM-DAG: 0000000000000050 {{.*}} c5_lds0
##   c5_lds1: size=8, align=8, users={c5_k0, c5_k3, c5_k4}, offset=0x60
# SYM-DAG: 0000000000000060 {{.*}} c5_lds1
##   c5_lds2: size=4, align=8, users={c5_k5}, offset=0x30
# SYM-DAG: 0000000000000030 {{.*}} c5_lds2
##   c5_lds3: size=32, align=16, users={c5_k0, c5_k3, c5_k4, c5_k5}, offset=0x10
# SYM-DAG: 0000000000000010 {{.*}} c5_lds3
##   c5_lds4: size=32, align=8, users={c5_k1, c5_k2, c5_k3, c5_k4}, offset=0x30
# SYM-DAG: 0000000000000030 {{.*}} c5_lds4
##   c5_lds5: size=12, align=4, users={c5_k0, c5_k1, c5_k2, c5_k3, c5_k4, c5_k5}, offset=0x0
# SYM-DAG: 0000000000000000 {{.*}} c5_lds5

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

## Case 0
	kernel c0_k0
	kernel c0_k1

	.amdgpu_info c0_k0
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c0_lds1
		.amdgpu_use c0_lds2
		.amdgpu_use c0_lds3
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c0_k1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c0_lds0
		.amdgpu_use c0_lds1
		.amdgpu_use c0_lds3
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	c0_lds0
	.amdgpu_lds c0_lds0, 20, 4
	.globl	c0_lds1
	.amdgpu_lds c0_lds1, 24, 4
	.globl	c0_lds2
	.amdgpu_lds c0_lds2, 4, 8
	.globl	c0_lds3
	.amdgpu_lds c0_lds3, 4, 4

## Case 1
	kernel c1_k0
	kernel c1_k1
	kernel c1_k2
	kernel c1_k3
	kernel c1_k4
	kernel c1_k5

	.amdgpu_info c1_k0
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c1_lds1
		.amdgpu_use c1_lds2
		.amdgpu_use c1_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c1_k1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c1_lds0
		.amdgpu_use c1_lds2
		.amdgpu_use c1_lds4
		.amdgpu_use c1_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c1_k2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c1_lds1
		.amdgpu_use c1_lds2
		.amdgpu_use c1_lds3
		.amdgpu_use c1_lds4
		.amdgpu_use c1_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c1_k3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c1_lds1
		.amdgpu_use c1_lds2
		.amdgpu_use c1_lds4
		.amdgpu_use c1_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c1_k4
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c1_lds0
		.amdgpu_use c1_lds1
		.amdgpu_use c1_lds2
		.amdgpu_use c1_lds3
		.amdgpu_use c1_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c1_k5
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c1_lds2
		.amdgpu_use c1_lds3
		.amdgpu_use c1_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	c1_lds0
	.amdgpu_lds c1_lds0, 4, 8
	.globl	c1_lds1
	.amdgpu_lds c1_lds1, 32, 4
	.globl	c1_lds2
	.amdgpu_lds c1_lds2, 32, 16
	.globl	c1_lds3
	.amdgpu_lds c1_lds3, 8, 4
	.globl	c1_lds4
	.amdgpu_lds c1_lds4, 16, 8
	.globl	c1_lds5
	.amdgpu_lds c1_lds5, 24, 4

## Case 2
	kernel c2_k0
	kernel c2_k1

	.amdgpu_info c2_k0
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c2_lds0
		.amdgpu_use c2_lds1
		.amdgpu_use c2_lds2
		.amdgpu_use c2_lds3
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c2_k1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c2_lds0
		.amdgpu_use c2_lds1
		.amdgpu_use c2_lds2
		.amdgpu_use c2_lds3
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	c2_lds0
	.amdgpu_lds c2_lds0, 16, 4
	.globl	c2_lds1
	.amdgpu_lds c2_lds1, 12, 8
	.globl	c2_lds2
	.amdgpu_lds c2_lds2, 12, 8
	.globl	c2_lds3
	.amdgpu_lds c2_lds3, 4, 4

## Case 3
	kernel c3_k0
	kernel c3_k1

	.amdgpu_info c3_k0
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c3_lds0
		.amdgpu_use c3_lds1
		.amdgpu_use c3_lds2
		.amdgpu_use c3_lds3
		.amdgpu_use c3_lds4
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c3_k1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c3_lds1
		.amdgpu_use c3_lds2
		.amdgpu_use c3_lds3
		.amdgpu_use c3_lds4
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	c3_lds0
	.amdgpu_lds c3_lds0, 12, 4
	.globl	c3_lds1
	.amdgpu_lds c3_lds1, 32, 16
	.globl	c3_lds2
	.amdgpu_lds c3_lds2, 32, 4
	.globl	c3_lds3
	.amdgpu_lds c3_lds3, 48, 8
	.globl	c3_lds4
	.amdgpu_lds c3_lds4, 32, 4

## Case 4
	kernel c4_k0
	kernel c4_k1

	.amdgpu_info c4_k0
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c4_lds0
		.amdgpu_use c4_lds2
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c4_k1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c4_lds1
		.amdgpu_use c4_lds2
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	c4_lds0
	.amdgpu_lds c4_lds0, 48, 16
	.globl	c4_lds1
	.amdgpu_lds c4_lds1, 24, 8
	.globl	c4_lds2
	.amdgpu_lds c4_lds2, 20, 8

## Case 5
	kernel c5_k0
	kernel c5_k1
	kernel c5_k2
	kernel c5_k3
	kernel c5_k4
	kernel c5_k5

	.amdgpu_info c5_k0
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c5_lds0
		.amdgpu_use c5_lds1
		.amdgpu_use c5_lds3
		.amdgpu_use c5_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c5_k1
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c5_lds0
		.amdgpu_use c5_lds4
		.amdgpu_use c5_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c5_k2
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c5_lds0
		.amdgpu_use c5_lds4
		.amdgpu_use c5_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c5_k3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c5_lds1
		.amdgpu_use c5_lds3
		.amdgpu_use c5_lds4
		.amdgpu_use c5_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c5_k4
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c5_lds0
		.amdgpu_use c5_lds1
		.amdgpu_use c5_lds3
		.amdgpu_use c5_lds4
		.amdgpu_use c5_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info c5_k5
		.amdgpu_flags 0
		.amdgpu_num_vgpr 1
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use c5_lds2
		.amdgpu_use c5_lds3
		.amdgpu_use c5_lds5
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.globl	c5_lds0
	.amdgpu_lds c5_lds0, 12, 8
	.globl	c5_lds1
	.amdgpu_lds c5_lds1, 8, 8
	.globl	c5_lds2
	.amdgpu_lds c5_lds2, 4, 8
	.globl	c5_lds3
	.amdgpu_lds c5_lds3, 32, 16
	.globl	c5_lds4
	.amdgpu_lds c5_lds4, 32, 8
	.globl	c5_lds5
	.amdgpu_lds c5_lds5, 12, 4
