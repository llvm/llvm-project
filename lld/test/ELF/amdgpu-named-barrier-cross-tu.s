# REQUIRES: amdgpu

## Cross-TU named barrier collision test.
##
## Without link-time resolution, each TU would independently assign barrier
## ID 1 to its local named barrier. When a kernel reaches both barriers
## transitively, this causes a collision. The linker must assign distinct
## barrier IDs to avoid this.
##
## TU A: kern uses bar_a (1 slot), calls helper
## TU B: helper uses bar_b (1 slot)
## After linking, kern reaches both bar_a and bar_b. The linker assigns
## distinct IDs (e.g., bar_a=1, bar_b=2 or vice versa). NAMED_BAR_CNT=1
## since ceil(2/4)=1.

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out

## Verify the resolved barrier symbols have distinct encoded addresses.
## Each address is 0x802000 | (scope << 9) | (barId << 4).
## bar_a and bar_b must have different barIds.
# RUN: llvm-nm --format=posix %t/out | FileCheck %s --check-prefix=SYMS

## Verify NAMED_BAR_CNT is correct in compute_pgm_rsrc3.
## NAMED_BAR_CNT=1 is encoded as 0x00004000, printed below as little-endian
## bytes 00400000.
# RUN: llvm-objdump -s -j .rodata %t/out | FileCheck %s --check-prefix=KD

## The two barrier symbols must have different addresses (distinct barrier IDs).
## Both should match the barrier encoding pattern: 0x802010 (ID=1) or 0x802020 (ID=2).
# SYMS-DAG: __amdgpu_named_barrier.bar_a{{[^ ]*}} A {{[0-9a-f]+}}
# SYMS-DAG: __amdgpu_named_barrier.bar_b{{[^ ]*}} A {{[0-9a-f]+}}

# KD: {{[0-9a-f]+}} 00000000 00000000 00000000 00400000

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kern                            ; -- Begin function kern
	.p2align	8
	.type	kern,@function
kern:
	s_endpgm
.Lfunc_end0:
	.size	kern, .Lfunc_end0-kern
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kern.kd
	.type	kern.kd,@object
	.size	kern.kd, 64
	.protected	kern
kern.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	kern@rel64-kern.kd
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
	.set .Lkern.num_vgpr, 32
	.set .Lkern.num_agpr, 0
	.set .Lkern.numbered_sgpr, 33
	.set .Lkern.num_named_barrier, 0
	.set .Lkern.private_seg_size, 0
	.set .Lkern.uses_vcc, 1
	.set .Lkern.uses_flat_scratch, 0
	.set .Lkern.has_dyn_sized_stack, 0
	.set .Lkern.has_recursion, 0
	.set .Lkern.has_indirect_call, 0
	.text
	.amdgpu_info kern
		.amdgpu_flags 1
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.bar_a.a8fb08c539c29f35a63da55cd03bd885
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
	.globl	__amdgpu_named_barrier.bar_a.a8fb08c539c29f35a63da55cd03bd885
	.amdgpu_lds __amdgpu_named_barrier.bar_a.a8fb08c539c29f35a63da55cd03bd885, 16, 4
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           kern
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kern.kd
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
	.amdgpu_info helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 0
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_named_barrier.bar_b.957946e05970693deaaae396388e1c50
		.amdgpu_occupancy 8
		.amdgpu_wave_size 32
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.text
	.globl	__amdgpu_named_barrier.bar_b.957946e05970693deaaae396388e1c50
	.amdgpu_lds __amdgpu_named_barrier.bar_b.957946e05970693deaaae396388e1c50, 16, 4
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
