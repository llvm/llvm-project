# REQUIRES: amdgpu

## Test that link-time resource propagation handles functions referenced via
## STT_SECTION symbols in relocations against the `.amdgpu.info` section.
##
## When a local (internal-linkage) function is placed in its own text section
## (e.g. .text.unlikely. due to the cold attribute), ELF assemblers
## canonically convert relocations targeting that function from the named
## local symbol to section_symbol + addend.  The linker must resolve these
## section symbol references back to the named function symbol so that it
## can build the call graph and propagate resource usage from `.amdgpu.info`
## correctly.
##
## Without this handling, the callee's resource info (scratch size, VGPR
## count, etc.) would be silently dropped and the kernel descriptor would
## be patched with incorrect values.

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o
# RUN: llvm-readobj -S %t.o | FileCheck %s --check-prefix=SHDR
# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readobj --notes %t.out | FileCheck %s

# SHDR: Name: .amdgpu.info

## The kernel itself has private_segment_fixed_size 0 but calls callee which
## has 132 bytes of scratch.  After propagation, the kernel should reflect the
## callee's scratch requirement.
# CHECK:       .name:           kernel
# CHECK:       .private_segment_fixed_size: 132

	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6

## callee lives in .text.unlikely. — a separate text section.
## Because callee is local (not .globl), llvm-mc will emit its relocations
## in `.amdgpu.info` using the STT_SECTION symbol for
## .text.unlikely. rather than the named symbol "callee".
	.section	.text.unlikely.,"ax",@progbits
	.p2align	6
	.type	callee,@function
callee:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_lshr_b32 s4, s32, 6
	v_lshl_add_u32 v1, v0, 2, s4
	buffer_store_dword v0, v1, s[0:3], 0 offen
	s_waitcnt vmcnt(0)
	buffer_load_dword v0, v1, s[0:3], 0 offen glc
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	callee, .Lfunc_end0-callee

	.text
	.globl	kernel
	.p2align	8
	.type	kernel,@function
kernel:
	s_add_u32 flat_scratch_lo, s12, s17
	s_addc_u32 flat_scratch_hi, s13, 0
	s_mov_b32 s13, s15
	s_load_dwordx2 s[18:19], s[8:9], 0x0
	s_load_dword s15, s[8:9], 0x8
	s_add_u32 s0, s0, s17
	s_addc_u32 s1, s1, 0
	s_add_u32 s8, s8, 16
	s_addc_u32 s9, s9, 0
	v_lshlrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v1, 10, v1
	s_mov_b32 s12, s14
	s_getpc_b64 s[20:21]
	s_add_u32 s20, s20, callee@rel32@lo+4
	s_addc_u32 s21, s21, callee@rel32@hi+12
	v_or3_b32 v31, v0, v1, v2
	s_mov_b32 s14, s16
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s15
	s_mov_b32 s32, 0
	v_mov_b32_e32 v3, 0
	s_swappc_b64 s[30:31], s[20:21]
	global_store_dword v3, v0, s[18:19]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel.kd
	.type	kernel.kd,@object
	.size	kernel.kd, 64
	.protected	kernel
kernel.kd:
	.long	0
	.long	0
	.long	272
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
.Lfunc_end1:
	.size	kernel, .Lfunc_end1-kernel

## Object linking metadata: callee has 132 bytes of scratch, kernel has 0.
## Flags: 0x1=is_kernel, 0x4=uses_flat_scratch (kernel only).
	.amdgpu_info callee
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 132
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info kernel
		.amdgpu_flags 2
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_call callee
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           x
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         96
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         104
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         112
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         120
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         128
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         216
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         kernel.kd
    .uses_dynamic_stack: true
    .vgpr_count:     32
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
