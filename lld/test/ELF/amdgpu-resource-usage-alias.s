# REQUIRES: amdgpu

## Test that the linker resolves function aliases during resource propagation.
## In the C++ Itanium ABI, C1 (complete object) constructors are often aliases
## to C2 (base object) constructors. The compiler emits a resource entry
## only for C2; C1 appears as a stub without has_local_res.
## Without alias resolution, the linker would report "incomplete resource usage".
##
## Call graph: kernel -> _ZN3FooC1Ev (alias) -> _ZN3FooC2Ev (real definition)

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out

## Verify the linker succeeds and patches HSA metadata.
# RUN: llvm-readobj --notes %t/out | FileCheck %s

# CHECK:      .name: test_kernel
# CHECK:      .sgpr_count:
# CHECK:      .vgpr_count:

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	test_kernel                     ; -- Begin function test_kernel
	.p2align	8
	.type	test_kernel,@function
test_kernel:
	s_endpgm
.Lfunc_end0:
	.size	test_kernel, .Lfunc_end0-test_kernel
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	test_kernel.kd
	.type	test_kernel.kd,@object
	.size	test_kernel.kd, 64
	.protected	test_kernel
test_kernel.kd:
	.long	0
	.long	0
	.long	256
	.long	0
	.quad	test_kernel@rel64-test_kernel.kd
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
	.set .Ltest_kernel.num_vgpr, 32
	.set .Ltest_kernel.num_agpr, 0
	.set .Ltest_kernel.numbered_sgpr, 33
	.set .Ltest_kernel.num_named_barrier, 0
	.set .Ltest_kernel.private_seg_size, 0
	.set .Ltest_kernel.uses_vcc, 1
	.set .Ltest_kernel.uses_flat_scratch, 1
	.set .Ltest_kernel.has_dyn_sized_stack, 0
	.set .Ltest_kernel.has_recursion, 0
	.set .Ltest_kernel.has_indirect_call, 0
	.amdgpu_info test_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_call _ZN3FooC1Ev
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
    .kernarg_segment_size: 256
    .max_flat_workgroup_size: 1024
    .name:           test_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         test_kernel.kd
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
	.globl	_ZN3FooC2Ev                     ; -- Begin function _ZN3FooC2Ev
	.p2align	6
	.type	_ZN3FooC2Ev,@function
_ZN3FooC2Ev:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	_ZN3FooC2Ev, .Lfunc_end0-_ZN3FooC2Ev
	.set .L_ZN3FooC2Ev.num_vgpr, 8
	.set .L_ZN3FooC2Ev.num_agpr, 0
	.set .L_ZN3FooC2Ev.numbered_sgpr, 32
	.set .L_ZN3FooC2Ev.num_named_barrier, 0
	.set .L_ZN3FooC2Ev.private_seg_size, 0
	.set .L_ZN3FooC2Ev.uses_vcc, 0
	.set .L_ZN3FooC2Ev.uses_flat_scratch, 0
	.set .L_ZN3FooC2Ev.has_dyn_sized_stack, 0
	.set .L_ZN3FooC2Ev.has_recursion, 0
	.set .L_ZN3FooC2Ev.has_indirect_call, 0
	.amdgpu_info _ZN3FooC2Ev
		.amdgpu_flags 0
		.amdgpu_num_vgpr 8
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 32
		.amdgpu_private_segment_size 0
		.amdgpu_typeid "v"
		.amdgpu_occupancy 4
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 8
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.globl	_ZN3FooC1Ev
	.type	_ZN3FooC1Ev,@function
.set _ZN3FooC1Ev, _ZN3FooC2Ev
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
