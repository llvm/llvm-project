# REQUIRES: amdgpu

## Test LDS reachability through a recursive SCC. The kernel reaches lds_data
## only through the H/A/M cycle, where A is the direct LDS user.

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --notes %t | FileCheck %s

# CHECK: .group_segment_fixed_size: 128
# CHECK: .name:           kernel
# CHECK: .uses_dynamic_stack: true

	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6

	.text
	.globl	kernel
	.p2align	8
	.type	kernel,@function
kernel:
	s_endpgm
.Lkernel_end:
	.size	kernel, .Lkernel_end-kernel

	.globl	H
	.p2align	6
	.type	H,@function
H:
	s_setpc_b64 s[30:31]
.LH_end:
	.size	H, .LH_end-H

	.globl	A
	.p2align	6
	.type	A,@function
A:
	s_setpc_b64 s[30:31]
.LA_end:
	.size	A, .LA_end-A

	.globl	M
	.p2align	6
	.type	M,@function
M:
	s_setpc_b64 s[30:31]
.LM_end:
	.size	M, .LM_end-M

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel.kd
	.type	kernel.kd,@object
	.size	kernel.kd, 64
	.protected	kernel
kernel.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	kernel@rel64-kernel.kd
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
	.amdgpu_info kernel
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_call H
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info H
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_call A
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info A
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_data
		.amdgpu_call M
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info M
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_call H
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 10
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_data
	.amdgpu_lds lds_data, 128, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
