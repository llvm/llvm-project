# REQUIRES: amdgpu

## Stale .amdgpu.info records from losing weak definitions or discarded COMDAT
## groups must not overwrite the resource usage of the prevailing definition.

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/kernel.s -o %t/kernel.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/weak-winner.s -o %t/weak-winner.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/weak-stale.s -o %t/weak-stale.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/comdat-winner.s -o %t/comdat-winner.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/comdat-stale.s -o %t/comdat-stale.o
# RUN: ld.lld %t/kernel.o %t/weak-winner.o %t/weak-stale.o %t/comdat-winner.o %t/comdat-stale.o -o %t/out
# RUN: llvm-readobj --notes %t/out | FileCheck %s

# CHECK:      .name: kernel
# CHECK:      .private_segment_fixed_size: 32

#--- kernel.s
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

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel.kd
	.type	kernel.kd,@object
	.size	kernel.kd, 64
	.protected	kernel
kernel.kd:
	.long	0
	.long	0
	.long	0
	.long	0
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
	.long	0
	.long	0
	.short	0
	.short	0
	.long	0

	.text
	.amdgpu_info kernel
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
		.amdgpu_call weak_helper
		.amdgpu_call comdat_helper
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- weak-winner.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.weak	weak_helper
	.p2align	6
	.type	weak_helper,@function
weak_helper:
	s_setpc_b64 s[30:31]
.Lweak_helper_end:
	.size	weak_helper, .Lweak_helper_end-weak_helper

	.amdgpu_info weak_helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 8
		.amdgpu_num_sgpr 8
		.amdgpu_private_segment_size 16
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- weak-stale.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.weak	weak_helper
	.p2align	6
	.type	weak_helper,@function
weak_helper:
	s_setpc_b64 s[30:31]
.Lweak_helper_end:
	.size	weak_helper, .Lweak_helper_end-weak_helper

	.amdgpu_info weak_helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 64
		.amdgpu_num_sgpr 64
		.amdgpu_private_segment_size 128
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- comdat-winner.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.section	.text.comdat_helper,"axG",@progbits,comdat_helper,comdat
	.globl	comdat_helper
	.p2align	6
	.type	comdat_helper,@function
comdat_helper:
	s_setpc_b64 s[30:31]
.Lcomdat_helper_end:
	.size	comdat_helper, .Lcomdat_helper_end-comdat_helper

	.amdgpu_info comdat_helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 12
		.amdgpu_num_sgpr 12
		.amdgpu_private_segment_size 32
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- comdat-stale.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.section	.text.comdat_helper,"axG",@progbits,comdat_helper,comdat
	.globl	comdat_helper
	.p2align	6
	.type	comdat_helper,@function
comdat_helper:
	s_setpc_b64 s[30:31]
.Lcomdat_helper_end:
	.size	comdat_helper, .Lcomdat_helper_end-comdat_helper

	.amdgpu_info comdat_helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 96
		.amdgpu_num_sgpr 96
		.amdgpu_private_segment_size 256
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
