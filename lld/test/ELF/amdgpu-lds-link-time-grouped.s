# REQUIRES: amdgpu

## Test grouped LDS allocation with independent kernel groups and tier ordering.
##
## TU1 has:
##   - func: uses @lds_callee (callee-scope, internal, 64 bytes)
##   - K1: uses @lds_global (global-scope, external, 32 bytes),
##         uses @lds_k1 (kernel-scope, internal, 16 bytes), calls func
##   - K2: uses @lds_global, calls func (no kernel-scope LDS)
##
## TU2 has:
##   - K3: uses @lds_k3 (kernel-scope, internal, 128 bytes)
##
## Grouping:
##   K1, K2 share lds_callee and lds_global -> Group 1
##   K3 is independent -> Group 2
##
## Group 1 tier classification:
##   Shared tier: __amdgpu_lds.func (callee-scope, user=func not a kernel)
##                lds_global (global-scope, users=K1,K2)
##   Kernel tier: __amdgpu_lds.K1 (kernel-scope, user=K1 a kernel)
##
## Group 1 layout (per-kernel frontier order; both shared vars have use_count=2,
## tie-broken by size -- larger first):
##   Shared: __amdgpu_lds.func (align 16, size 64) @ 0x00
##           lds_global (align 16, size 32) @ 0x40
##   Kernel: __amdgpu_lds.K1 (align 16, size 16) @ 0x60
##
## Group 2 layout (starts from offset 0):
##   __amdgpu_lds.K3 (align 16, size 128) @ 0x00
##
## Per-kernel sizes:
##   K1: reaches func's struct(0..64), lds_global(64..96), K1's struct(96..112) -> 0x70
##   K2: reaches func's struct(0..64), lds_global(64..96) -> 0x60
##   K3: reaches K3's struct(0..128) -> 0x80

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu1.s -o %t/tu1.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu2.s -o %t/tu2.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/tu3.s -o %t/tu3.o
# RUN: ld.lld %t/tu1.o %t/tu2.o %t/tu3.o -o %t/out
# RUN: llvm-readelf -s %t/out | FileCheck %s --check-prefix=SYM
# RUN: llvm-readobj --notes %t/out | FileCheck %s --check-prefix=META

## LDS symbol offsets (Group 1 shared tier first, kernel tier last)
# SYM-DAG: 0000000000000000 {{.*}} __amdgpu_lds.func
# SYM-DAG: 0000000000000040 {{.*}} lds_global
# SYM-DAG: 0000000000000060 {{.*}} __amdgpu_lds.K1

## Group 2 allocates from offset 0 independently
# SYM-DAG: 0000000000000000 {{.*}} __amdgpu_lds.K3

## Per-kernel LDS sizes patched into the kernel descriptor and HSA metadata
# META-DAG: .group_segment_fixed_size: 112
# META-DAG: .group_segment_fixed_size: 96
# META-DAG: .group_segment_fixed_size: 128

#--- tu1.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	func                            ; -- Begin function func
	.p2align	6
	.type	func,@function
func:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	func, .Lfunc_end0-func
	.set .Lfunc.num_vgpr, 41
	.set .Lfunc.num_agpr, 0
	.set .Lfunc.numbered_sgpr, 34
	.set .Lfunc.num_named_barrier, 0
	.set .Lfunc.private_seg_size, 16
	.set .Lfunc.uses_vcc, 1
	.set .Lfunc.uses_flat_scratch, 0
	.set .Lfunc.has_dyn_sized_stack, 0
	.set .Lfunc.has_recursion, 0
	.set .Lfunc.has_indirect_call, 0
	.text
	.globl	K1                              ; -- Begin function K1
	.p2align	8
	.type	K1,@function
K1:
	s_endpgm
.Lfunc_end1:
	.size	K1, .Lfunc_end1-K1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K1.kd
	.type	K1.kd,@object
	.size	K1.kd, 64
	.protected	K1
K1.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K1@rel64-K1.kd
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
	.set .LK1.num_vgpr, 32
	.set .LK1.num_agpr, 0
	.set .LK1.numbered_sgpr, 33
	.set .LK1.num_named_barrier, 0
	.set .LK1.private_seg_size, 0
	.set .LK1.uses_vcc, 1
	.set .LK1.uses_flat_scratch, 1
	.set .LK1.has_dyn_sized_stack, 0
	.set .LK1.has_recursion, 1
	.set .LK1.has_indirect_call, 0
	.text
	.globl	K2                              ; -- Begin function K2
	.p2align	8
	.type	K2,@function
K2:
	s_endpgm
.Lfunc_end2:
	.size	K2, .Lfunc_end2-K2
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K2.kd
	.type	K2.kd,@object
	.size	K2.kd, 64
	.protected	K2
K2.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K2@rel64-K2.kd
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
	.set .LK2.num_vgpr, 32
	.set .LK2.num_agpr, 0
	.set .LK2.numbered_sgpr, 33
	.set .LK2.num_named_barrier, 0
	.set .LK2.private_seg_size, 0
	.set .LK2.uses_vcc, 1
	.set .LK2.uses_flat_scratch, 1
	.set .LK2.has_dyn_sized_stack, 0
	.set .LK2.has_recursion, 1
	.set .LK2.has_indirect_call, 0
	.amdgpu_info func
		.amdgpu_flags 1
		.amdgpu_num_vgpr 41
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 34
		.amdgpu_private_segment_size 16
		.amdgpu_use __amdgpu_lds.func
		.amdgpu_call extern_func
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K1
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_lds.K1
		.amdgpu_use lds_global
		.amdgpu_call func
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.amdgpu_info K2
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_global
		.amdgpu_call func
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 41
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 34
	.set amdgpu.max_num_named_barrier, 0
	.globl	lds_global
	.amdgpu_lds lds_global, 32, 16
	.globl	__amdgpu_lds.func
	.amdgpu_lds __amdgpu_lds.func, 64, 16
	.globl	__amdgpu_lds.K1
	.amdgpu_lds __amdgpu_lds.K1, 16, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K1
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K1.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K2
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K2.kd
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
	.globl	K3                              ; -- Begin function K3
	.p2align	8
	.type	K3,@function
K3:
	s_endpgm
.Lfunc_end0:
	.size	K3, .Lfunc_end0-K3
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	K3.kd
	.type	K3.kd,@object
	.size	K3.kd, 64
	.protected	K3
K3.kd:
	.long	0
	.long	0
	.long	264
	.long	0
	.quad	K3@rel64-K3.kd
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
	.set .LK3.num_vgpr, 2
	.set .LK3.num_agpr, 0
	.set .LK3.numbered_sgpr, 10
	.set .LK3.num_named_barrier, 0
	.set .LK3.private_seg_size, 0
	.set .LK3.uses_vcc, 0
	.set .LK3.uses_flat_scratch, 0
	.set .LK3.has_dyn_sized_stack, 0
	.set .LK3.has_recursion, 0
	.set .LK3.has_indirect_call, 0
	.amdgpu_info K3
		.amdgpu_flags 0
		.amdgpu_num_vgpr 2
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 10
		.amdgpu_private_segment_size 0
		.amdgpu_use __amdgpu_lds.K3
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.globl	__amdgpu_lds.K3
	.amdgpu_lds __amdgpu_lds.K3, 128, 16
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 1024
    .name:           K3
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .sgpr_spill_count: 0
    .symbol:         K3.kd
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

#--- tu3.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	extern_func                     ; -- Begin function extern_func
	.p2align	6
	.type	extern_func,@function
extern_func:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	extern_func, .Lfunc_end0-extern_func
	.set .Lextern_func.num_vgpr, 0
	.set .Lextern_func.num_agpr, 0
	.set .Lextern_func.numbered_sgpr, 32
	.set .Lextern_func.num_named_barrier, 0
	.set .Lextern_func.private_seg_size, 0
	.set .Lextern_func.uses_vcc, 0
	.set .Lextern_func.uses_flat_scratch, 0
	.set .Lextern_func.has_dyn_sized_stack, 0
	.set .Lextern_func.has_recursion, 0
	.set .Lextern_func.has_indirect_call, 0
	.amdgpu_info extern_func
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
