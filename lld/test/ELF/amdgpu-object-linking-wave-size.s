# REQUIRES: amdgpu

## Test that the linker rejects cross-TU calls between functions compiled with
## different wavefront sizes.

# RUN: split-file %s %t

# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %t/b.s -o %t/b.o
# RUN: not ld.lld %t/a.o %t/b.o -o /dev/null 2>&1 | FileCheck %s
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %t/wgp-direct.s -o %t/wgp-direct.o
# RUN: not ld.lld %t/wgp-direct.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE-DIRECT
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %t/wgp-indirect.s -o %t/wgp-indirect.o
# RUN: not ld.lld %t/wgp-indirect.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE-INDIRECT

# CHECK: error: AMDGPU object linking: wave size mismatch in call from 'kernel' (wave32) to 'helper' (wave64)
# MODE-DIRECT: error: AMDGPU object linking: CU/WGP mode mismatch in call from 'wgp_caller' (WGP) to 'cu_callee' (CU)
# MODE-INDIRECT: error: AMDGPU object linking: CU/WGP mode mismatch in call from 'wgp_indirect_caller' (WGP) to 'cu_addr_taker' (CU)

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text
	.globl	kernel
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
	.short	1063
	.short	0
	.long	0
	.text
	.amdgpu_info kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 32
		.amdgpu_call helper
	.end_amdgpu_info

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
    .symbol:         kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     32
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- wgp-direct.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text
	.globl	wgp_caller
	.p2align	2
	.type	wgp_caller,@function
wgp_caller:
	s_setpc_b64 s[30:31]
.Lwgp_caller_end:
	.size	wgp_caller, .Lwgp_caller_end-wgp_caller

	.globl	cu_callee
	.p2align	2
	.type	cu_callee,@function
cu_callee:
	s_setpc_b64 s[30:31]
.Lcu_callee_end:
	.size	cu_callee, .Lcu_callee_end-cu_callee

	.amdgpu_info wgp_caller
		.amdgpu_flags 8
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
		.amdgpu_call cu_callee
	.end_amdgpu_info

	.amdgpu_info cu_callee
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- wgp-indirect.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text
	.globl	wgp_indirect_caller
	.p2align	2
	.type	wgp_indirect_caller,@function
wgp_indirect_caller:
	s_setpc_b64 s[30:31]
.Lwgp_indirect_caller_end:
	.size	wgp_indirect_caller, .Lwgp_indirect_caller_end-wgp_indirect_caller

	.globl	cu_addr_taker
	.p2align	2
	.type	cu_addr_taker,@function
cu_addr_taker:
	s_setpc_b64 s[30:31]
.Lcu_addr_taker_end:
	.size	cu_addr_taker, .Lcu_addr_taker_end-cu_addr_taker

	.amdgpu_info wgp_indirect_caller
		.amdgpu_flags 8
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
		.amdgpu_indirect_call "v"
	.end_amdgpu_info

	.amdgpu_info cu_addr_taker
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
		.amdgpu_typeid "v"
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- b.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text
	.globl	helper
	.p2align	2
	.type	helper,@function
helper:
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	helper, .Lfunc_end1-helper

	.amdgpu_info helper
		.amdgpu_flags 0
		.amdgpu_num_vgpr 10
		.amdgpu_num_sgpr 8
		.amdgpu_private_segment_size 16
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
