; REQUIRES: comgr-has-transpiler

; RUN: %llvm-mc -defsym=GFX942=1 -triple=amdgpu9.42-amd-amdhsa -filetype=obj %s -o %t.gfx942.o
; RUN: %ld.lld -shared %t.gfx942.o -o %t.gfx942.hsaco
; RUN: %transpile_cli %t.gfx942.hsaco --emit-ir=trap_kernel,debugtrap_kernel \
; RUN:   --target-isa=gfx942 | %FileCheck %s --check-prefixes=GFX942-TRAP,GFX942-DEBUGTRAP
; RUN: %llvm-mc -defsym=GFX942=0 -triple=amdgpu12.50-amd-amdhsa -filetype=obj %s -o %t.gfx1250.o
; RUN: %ld.lld -shared %t.gfx1250.o -o %t.gfx1250.hsaco
; RUN: not %transpile_cli %t.gfx1250.hsaco --emit-ir=trap_kernel \
; RUN:   --target-isa=gfx942 2>&1 | %FileCheck %s --check-prefix=GFX1250-TO-GFX942

	.if GFX942
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.else
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.endif
	.amdhsa_code_object_version 6
	.text
	.globl	trap_kernel
	.p2align	8
	.type	trap_kernel,@function
trap_kernel:
; GFX942-TRAP-LABEL: define amdgpu_kernel void @trap_kernel(
; GFX942-TRAP: call void @llvm.trap()
; GFX942-TRAP-NEXT: unreachable
; GFX1250-TO-GFX942: requires-per-source-wave-execution: s_trap [SOPP]
; GFX1250-TO-GFX942-SAME: instruction requires a separate target wave for each source wave
	s_trap 0x102
	s_endpgm

	.globl	debugtrap_kernel
	.p2align	8
	.type	debugtrap_kernel,@function
debugtrap_kernel:
; GFX942-DEBUGTRAP-LABEL: define amdgpu_kernel void @debugtrap_kernel(
; GFX942-DEBUGTRAP: call void @llvm.debugtrap()
; GFX942-DEBUGTRAP-NEXT: ret void
	.if GFX942
	s_trap 3
	.else
	s_trap 0x13
	.endif
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.if GFX942
	.amdhsa_kernel trap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.amdhsa_kernel debugtrap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.else
	.amdhsa_kernel trap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel debugtrap_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.endif
	.text
	.if GFX942
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           trap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         trap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           debugtrap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         debugtrap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
	.else
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           trap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         trap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           debugtrap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         debugtrap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
	.endif
