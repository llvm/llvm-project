# REQUIRES: amdgpu

## Resource usage propagation through recursive SCCs must reach a fixed point.
## In this graph, H/A/M form a cycle and A calls a high-resource leaf C:
##
##   kernel_h -> H -> A -> M -> H
##                    `-> C
##   kernel_m -> M
##
## Both kernels must receive C's resource usage through the SCC, regardless of
## which SCC member they enter through.

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --notes %t | FileCheck %s

# CHECK:      .name: kernel_h
# CHECK:      .private_segment_fixed_size: 80
# CHECK:      .uses_dynamic_stack: true
# CHECK:      .vgpr_count: 100
# CHECK:      .name: kernel_m
# CHECK:      .private_segment_fixed_size: 80
# CHECK:      .uses_dynamic_stack: true
# CHECK:      .vgpr_count: 100

	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6

	.text
	.globl	kernel_h
	.p2align	8
	.type	kernel_h,@function
kernel_h:
	s_endpgm
.Lkernel_h_end:
	.size	kernel_h, .Lkernel_h_end-kernel_h

	.globl	kernel_m
	.p2align	8
	.type	kernel_m,@function
kernel_m:
	s_endpgm
.Lkernel_m_end:
	.size	kernel_m, .Lkernel_m_end-kernel_m

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

	.globl	C
	.p2align	6
	.type	C,@function
C:
	s_setpc_b64 s[30:31]
.LC_end:
	.size	C, .LC_end-C

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel_h.kd
	.type	kernel_h.kd,@object
	.size	kernel_h.kd, 64
	.protected	kernel_h
kernel_h.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	kernel_h@rel64-kernel_h.kd
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

	.p2align	6, 0x0
	.globl	kernel_m.kd
	.type	kernel_m.kd,@object
	.size	kernel_m.kd, 64
	.protected	kernel_m
kernel_m.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	kernel_m@rel64-kernel_m.kd
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
	.amdgpu_info kernel_h
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_call H
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info kernel_m
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_call M
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info H
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 16
		.amdgpu_call A
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info A
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 16
		.amdgpu_call M
		.amdgpu_call C
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info M
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 16
		.amdgpu_call H
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info C
		.amdgpu_flags 0
		.amdgpu_num_vgpr 100
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 64
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 100
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           kernel_h
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         kernel_h.kd
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           kernel_m
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         kernel_m.kd
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
