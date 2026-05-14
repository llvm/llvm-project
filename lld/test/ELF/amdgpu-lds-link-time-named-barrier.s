# REQUIRES: amdgpu

## Test that lld patches compute_pgm_rsrc3.NAMED_BAR_CNT with the cross-TU
## propagated named-barrier count for GFX1250.
##
## TU A: kern_a uses 1 named barrier, calls external helper.
## TU B: helper uses 5 named barriers, kern_b also uses 5.
## After linking, kern_a gets max(1, 5) = 5 barriers -> NamedBarCnt = ceil(5/4) = 2.
## NAMED_BAR_CNT occupies bits [14:16] of compute_pgm_rsrc3.
## Expected: 2 << 14 = 0x8000 for both kern_a and kern_b.

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out

## .rodata contains two 64-byte kernel descriptors. Check compute_pgm_rsrc3 at
## offset 44 from each KD start. NAMED_BAR_CNT=2 is encoded as 0x00008000,
## printed below as little-endian bytes 00800000.
# RUN: llvm-objdump -s -j .rodata %t/out | FileCheck %s --check-prefix=KD

# KD: {{[0-9a-f]+}} 00000000 00000000 00000000 00800000
# KD: {{[0-9a-f]+}} 00000000 00000000 00000000 00800000

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kern_a                          ; -- Begin function kern_a
	.p2align	8
	.type	kern_a,@function
kern_a:
	s_endpgm
.Lfunc_end0:
	.size	kern_a, .Lfunc_end0-kern_a
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kern_a.kd
	.type	kern_a.kd,@object
	.size	kern_a.kd, 64
	.protected	kern_a
kern_a.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	kern_a@rel64-kern_a.kd
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
	.long	3222208513
	.long	5008
	.short	1054
	.short	0
	.long	0
	.text
	.set .Lkern_a.num_vgpr, 32
	.set .Lkern_a.num_agpr, 0
	.set .Lkern_a.numbered_sgpr, 33
	.set .Lkern_a.num_named_barrier, 0
	.set .Lkern_a.private_seg_size, 0
	.set .Lkern_a.uses_vcc, 1
	.set .Lkern_a.uses_flat_scratch, 0
	.set .Lkern_a.has_dyn_sized_stack, 0
	.set .Lkern_a.has_recursion, 0
	.set .Lkern_a.has_indirect_call, 0
	.text
	.amdgpu_info kern_a
		.amdgpu_flags 1
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.bar_a.4afea3fa75d4f11a11e1e1e3237d94b2
		.amdgpu_call helper
		.amdgpu_occupancy 8
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.globl	__amdgpu_named_barrier.bar_a.4afea3fa75d4f11a11e1e1e3237d94b2
	.amdgpu_lds __amdgpu_named_barrier.bar_a.4afea3fa75d4f11a11e1e1e3237d94b2, 16, 4
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           kern_a
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kern_a.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- b.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	helper                          ; -- Begin function helper
	.p2align	7
	.type	helper,@function
helper:
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
	.text
	.globl	kern_b                          ; -- Begin function kern_b
	.p2align	8
	.type	kern_b,@function
kern_b:
	s_endpgm
.Lfunc_end1:
	.size	kern_b, .Lfunc_end1-kern_b
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kern_b.kd
	.type	kern_b.kd,@object
	.size	kern_b.kd, 64
	.protected	kern_b
kern_b.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	kern_b@rel64-kern_b.kd
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
	.long	3222208512
	.long	5008
	.short	1054
	.short	0
	.long	0
	.text
	.set .Lkern_b.num_vgpr, 0
	.set .Lkern_b.num_agpr, 0
	.set .Lkern_b.numbered_sgpr, 1
	.set .Lkern_b.num_named_barrier, 0
	.set .Lkern_b.private_seg_size, 0
	.set .Lkern_b.uses_vcc, 0
	.set .Lkern_b.uses_flat_scratch, 0
	.set .Lkern_b.has_dyn_sized_stack, 0
	.set .Lkern_b.has_recursion, 0
	.set .Lkern_b.has_indirect_call, 0
	.text
	.amdgpu_info helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 0
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.bar_b.36a90f902d5ba5557b314d56b5259233
		.amdgpu_occupancy 8
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.amdgpu_info kern_b
		.amdgpu_flags 0
		.amdgpu_num_vgpr 0
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 1
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.bar_b.36a90f902d5ba5557b314d56b5259233
		.amdgpu_occupancy 8
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.text
	.globl	__amdgpu_named_barrier.bar_b.36a90f902d5ba5557b314d56b5259233
	.amdgpu_lds __amdgpu_named_barrier.bar_b.36a90f902d5ba5557b314d56b5259233, 80, 4
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           kern_b
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kern_b.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
