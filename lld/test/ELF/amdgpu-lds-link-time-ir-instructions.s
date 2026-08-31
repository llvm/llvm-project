# REQUIRES: amdgpu

## End-to-end IR-to-linked-binary instruction verification for link-time LDS.
## Two TUs share an LDS symbol (__lds_shared) across translation units and
## TU2 also has a private LDS symbol (__lds_private). This tests the full
## pipeline from compiler output through linking, checking:
##   1) Pre-link: relocations in object files (R_AMDGPU_ABS32_LO)
##   2) Pre-link: placeholder 0 in address-computation instructions
##   3) Post-link: resolved offsets patched into instructions
##   4) LDS load/store instructions remain intact through linking
##
## LDS layout (alignment desc, size desc):
##   __lds_shared  (align=16, size=256) -> offset 0x000
##   __lds_private (align=4,  size=128) -> offset 0x100
##
## TU1: kernel_a stores 42 to __lds_shared[idx].
## TU2: kernel_b reads __lds_shared[idx], increments it, writes it back,
##      and stores 99 to __lds_private[idx].

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu2.s -o %t/tu2.o

## Pre-link: verify relocations target the LDS symbols.
# RUN: llvm-objdump -d -r %t/tu1.o | FileCheck %s --check-prefix=PRE1
# RUN: llvm-objdump -d -r %t/tu2.o | FileCheck %s --check-prefix=PRE2

## Link.
# RUN: ld.lld %t/tu1.o %t/tu2.o -o %t/out

## Post-link: verify resolved instructions.
# RUN: llvm-objdump -d %t/out | FileCheck %s --check-prefix=POST

## === Pre-link TU1: kernel_a ===
## The compiler emits s_lshl2_add_u32 to compute byte offset (idx << 2) + base.
## The LDS base is a placeholder 0 with a relocation.

# PRE1-LABEL: <kernel_a>:
# PRE1:      s_lshl2_add_u32 s0, s0, lit(0x0)
# PRE1-NEXT:   {{.*}} R_AMDGPU_ABS32_LO __lds_shared
# PRE1:      ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}

## === Pre-link TU2: kernel_b ===
## Two LDS accesses: __lds_shared and __lds_private, both with relocations.

# PRE2-LABEL: <kernel_b>:
## First access: __lds_shared (s_add_i32 with relocation).
# PRE2:      s_add_i32 s{{[0-9]+}}, s0, lit(0x0)
# PRE2-NEXT:   {{.*}} R_AMDGPU_ABS32_LO __lds_shared
# PRE2:      ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}
## Second access: __lds_private (s_add_i32 with relocation).
# PRE2:      s_add_i32 s0, s0, lit(0x0)
# PRE2-NEXT:   {{.*}} R_AMDGPU_ABS32_LO __lds_private
# PRE2:      ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}
## Increment and write-back to __lds_shared.
# PRE2:      v_add_u32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
# PRE2:      ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}

## === Post-link: resolved offsets ===

## kernel_a: __lds_shared resolved to offset 0.
# POST-LABEL: <kernel_a>:
# POST:      s_lshl2_add_u32 s0, s0, lit(0x0)
# POST:      ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}

## kernel_b: __lds_shared at offset 0, __lds_private at offset 0x100.
# POST-LABEL: <kernel_b>:
## __lds_shared base = 0 (unchanged from placeholder).
# POST:      s_add_i32 s{{[0-9]+}}, s0, lit(0x0)
# POST:      ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}
## __lds_private base = 0x100 (resolved from placeholder 0).
# POST:      s_add_i32 s0, s0, 0x100
# POST:      ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}
## Increment + write-back.
# POST:      v_add_u32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
# POST:      ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel_a                        ; -- Begin function kernel_a
	.p2align	8
	.type	kernel_a,@function
kernel_a:                               ; @kernel_a
	s_load_dword s0, s[8:9], 0x0
	v_mov_b32_e32 v0, 42
	s_waitcnt lgkmcnt(0)
	s_lshl2_add_u32 s0, s0, __lds_shared@abs32@lo
	v_mov_b32_e32 v1, s0
	ds_write_b32 v1, v0
	s_endpgm
.Lfunc_end0:
	.size	kernel_a, .Lfunc_end0-kernel_a
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel_a.kd
	.type	kernel_a.kd,@object
	.size	kernel_a.kd, 64
	.protected	kernel_a
kernel_a.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	kernel_a@rel64-kernel_a.kd
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
	.long	11468864
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.amdgpu_info kernel_a
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.globl	__lds_shared
	.amdgpu_lds __lds_shared, 256, 16
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           kernel_a
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kernel_a.kd
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
#--- tu2.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel_b                        ; -- Begin function kernel_b
	.p2align	8
	.type	kernel_b,@function
kernel_b:                               ; @kernel_b
	s_load_dword s0, s[8:9], 0x0
	v_mov_b32_e32 v2, 0x63
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s0, s0, 2
	s_add_i32 s1, s0, __lds_shared@abs32@lo
	v_mov_b32_e32 v0, s1
	ds_read_b32 v1, v0
	s_add_i32 s0, s0, __lds_private@abs32@lo
	v_mov_b32_e32 v3, s0
	ds_write_b32 v3, v2
	s_waitcnt lgkmcnt(1)
	v_add_u32_e32 v1, 1, v1
	ds_write_b32 v0, v1
	s_endpgm
.Lfunc_end0:
	.size	kernel_b, .Lfunc_end0-kernel_b
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel_b.kd
	.type	kernel_b.kd,@object
	.size	kernel_b.kd, 64
	.protected	kernel_b
kernel_b.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	kernel_b@rel64-kernel_b.kd
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
	.long	11468864
	.long	5020
	.short	63
	.short	0
	.long	0
	.text
	.amdgpu_info kernel_b
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.globl	__lds_private
	.amdgpu_lds __lds_private, 128, 4
	.globl	__lds_shared
	.amdgpu_lds __lds_shared, 256, 16
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           kernel_b
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         kernel_b.kd
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
