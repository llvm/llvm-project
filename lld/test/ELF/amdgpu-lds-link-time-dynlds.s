# REQUIRES: amdgpu

## Test that size-0 LDS symbols (dynamic LDS) are placed after all static LDS
## symbols and that the kernel descriptor's group_segment_fixed_size reflects
## the dynamic base offset.

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out

## Verify symbol offsets.
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM

## Verify group_segment_fixed_size in HSA metadata is patched.
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## LDS layout:
##   static_lds: align=16, size=256 -> offset 0
##   dyn_lds: align=4, size=0 -> offset 256
## Total fixed LDS = 256 = 0x100 (dynamic base)
##
## group_segment_fixed_size = 256 because the size-0 dyn_lds symbol
## sits at offset 256 but contributes 0 bytes.

# SYM-DAG: 0000000000000000 {{.*}} static_lds
# SYM-DAG: 0000000000000100 {{.*}} dyn_lds

# META: .group_segment_fixed_size: 256
# META: .name: my_kernel

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	my_kernel                       ; -- Begin function my_kernel
	.p2align	8
	.type	my_kernel,@function
my_kernel:
	s_endpgm
.Lfunc_end0:
	.size	my_kernel, .Lfunc_end0-my_kernel
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	my_kernel.kd
	.type	my_kernel.kd,@object
	.size	my_kernel.kd, 64
	.protected	my_kernel
my_kernel.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	my_kernel@rel64-my_kernel.kd
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
	.set .Lmy_kernel.num_vgpr, 32
	.set .Lmy_kernel.num_agpr, 0
	.set .Lmy_kernel.numbered_sgpr, 33
	.set .Lmy_kernel.num_named_barrier, 0
	.set .Lmy_kernel.private_seg_size, 0
	.set .Lmy_kernel.uses_vcc, 1
	.set .Lmy_kernel.uses_flat_scratch, 1
	.set .Lmy_kernel.has_dyn_sized_stack, 0
	.set .Lmy_kernel.has_recursion, 0
	.set .Lmy_kernel.has_indirect_call, 0
	.amdgpu_info my_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use static_lds
		.amdgpu_use dyn_lds
		.amdgpu_call helper
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	static_lds
	.amdgpu_lds static_lds, 256, 16
	.globl	dyn_lds
	.amdgpu_lds dyn_lds, 0, 4
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           my_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         my_kernel.kd
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
