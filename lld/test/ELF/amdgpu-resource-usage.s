# REQUIRES: amdgpu

## Test link-time resource usage propagation for AMDGPU.
## The linker parses AMDGPU object linking metadata in the `.amdgpu.info` section,
## propagates resource usage
## across the call graph, and patches the kernel descriptor and HSA metadata
## with the propagated values.

# RUN: split-file %s %t

## --- GFX900 test ---
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out

## Verify the linker patches HSA metadata after resource propagation.
# RUN: llvm-readobj --notes %t/out | FileCheck %s

## kernel calls helper (defined in b.ll). After propagation, the kernel's
## metadata should reflect the max of both functions' register usage.
# CHECK:      .name: kernel
# CHECK:      .sgpr_count:
# CHECK:      .vgpr_count:

## --- GFX90A test ---
## Test that on GFX90A the linker correctly accounts for extra SGPRs
## and uses the proper VGPR total computation (GFX90A alignment rule).
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj %t/a90a.s -o %t/a90a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj %t/b90a.s -o %t/b90a.o
# RUN: ld.lld %t/a90a.o %t/b90a.o -o %t/out90a
# RUN: llvm-readobj --notes %t/out90a | FileCheck %s --check-prefix=GFX90A

## After propagation, the kernel should have SGPRs with extras accounted for
## and VGPR count from the callee.
# GFX90A:      .name: kernel90a
# GFX90A:      .sgpr_count:
# GFX90A:      .vgpr_count:

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel                          ; -- Begin function kernel
	.p2align	8
	.type	kernel,@function
kernel:
	s_endpgm
.Lfunc_end0:
	.size	kernel, .Lfunc_end0-kernel
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel.kd
	.type	kernel.kd,@object
	.size	kernel.kd, 64
	.protected	kernel
kernel.kd:
	.long	0
	.long	0
	.long	264
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
	.set .Lkernel.num_vgpr, 32
	.set .Lkernel.num_agpr, 0
	.set .Lkernel.numbered_sgpr, 33
	.set .Lkernel.num_named_barrier, 0
	.set .Lkernel.private_seg_size, 0
	.set .Lkernel.uses_vcc, 1
	.set .Lkernel.uses_flat_scratch, 1
	.set .Lkernel.has_dyn_sized_stack, 0
	.set .Lkernel.has_recursion, 0
	.set .Lkernel.has_indirect_call, 0
	.amdgpu_info kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_call helper
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     32
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- b.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	helper                          ; -- Begin function helper
	.p2align	6
	.type	helper,@function
helper:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	helper, .Lfunc_end0-helper
	.set .Lhelper.num_vgpr, 0
	.set .Lhelper.num_agpr, 0
	.set .Lhelper.numbered_sgpr, 32
	.set .Lhelper.num_named_barrier, 0
	.set .Lhelper.private_seg_size, 0
	.set .Lhelper.uses_vcc, 0
	.set .Lhelper.uses_flat_scratch, 0
	.set .Lhelper.has_dyn_sized_stack, 0
	.set .Lhelper.has_recursion, 0
	.set .Lhelper.has_indirect_call, 0
	.amdgpu_info helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 0
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- a90a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel90a                       ; -- Begin function kernel90a
	.p2align	8
	.type	kernel90a,@function
kernel90a:
	s_endpgm
.Lfunc_end0:
	.size	kernel90a, .Lfunc_end0-kernel90a
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel90a.kd
	.type	kernel90a.kd,@object
	.size	kernel90a.kd, 64
	.protected	kernel90a
kernel90a.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	kernel90a@rel64-kernel90a.kd
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
	.long	7
	.long	11469059
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.set .Lkernel90a.num_vgpr, 32
	.set .Lkernel90a.num_agpr, 0
	.set .Lkernel90a.numbered_sgpr, 33
	.set .Lkernel90a.num_named_barrier, 0
	.set .Lkernel90a.private_seg_size, 0
	.set .Lkernel90a.uses_vcc, 1
	.set .Lkernel90a.uses_flat_scratch, 1
	.set .Lkernel90a.has_dyn_sized_stack, 0
	.set .Lkernel90a.has_recursion, 0
	.set .Lkernel90a.has_indirect_call, 0
	.text
	.amdgpu_info kernel90a
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_call mfma_helper
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .name:           x
        .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         88
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         96
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         104
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         112
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         120
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         208
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           kernel90a
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         kernel90a.kd
    .uses_dynamic_stack: false
    .vgpr_count:     32
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- b90a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
	.amdhsa_code_object_version 6
	.text
	.globl	mfma_helper                     ; -- Begin function mfma_helper
	.p2align	6
	.type	mfma_helper,@function
mfma_helper:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	mfma_helper, .Lfunc_end0-mfma_helper
	.set .Lmfma_helper.num_vgpr, 8
	.set .Lmfma_helper.num_agpr, 0
	.set .Lmfma_helper.numbered_sgpr, 32
	.set .Lmfma_helper.num_named_barrier, 0
	.set .Lmfma_helper.private_seg_size, 0
	.set .Lmfma_helper.uses_vcc, 0
	.set .Lmfma_helper.uses_flat_scratch, 0
	.set .Lmfma_helper.has_dyn_sized_stack, 0
	.set .Lmfma_helper.has_recursion, 0
	.set .Lmfma_helper.has_indirect_call, 0
	.text
	.amdgpu_info mfma_helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 8
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 8
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
