# REQUIRES: amdgpu

## The type-id string in .amdgpu.strtab is intentionally not NUL-terminated.
## The linker should bound the read to the section contents and still match the
## indirect call to the address-taken function.

# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/a.s -o %t/a.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out
# RUN: llvm-readobj --notes %t/out | FileCheck %s

# CHECK:      .group_segment_fixed_size: 128
# CHECK:      .name: kernel

#--- a.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	target_func
	.p2align	6
	.type	target_func,@function
target_func:
	s_setpc_b64 s[30:31]
.Ltarget_func_end:
	.size	target_func, .Ltarget_func_end-target_func

	.globl	lds_var
	.amdgpu_lds lds_var, 128, 16

	.section	.amdgpu.info,"e",@progbits
	.byte	1
	.byte	8
	.quad	target_func
	.byte	2
	.byte	4
	.long	0
	.byte	4
	.byte	4
	.long	4
	.byte	3
	.byte	4
	.long	4
	.byte	6
	.byte	4
	.long	0
	.byte	11
	.byte	4
	.long	8
	.byte	12
	.byte	4
	.long	64
	.byte	7
	.byte	8
	.quad	lds_var
	.byte	10
	.byte	4
	.long	0

	.section	.amdgpu.strtab,"e",@progbits
	.ascii	"vi"

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
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
	.globl	caller
	.p2align	6
	.type	caller,@function
caller:
	s_setpc_b64 s[30:31]
.Lcaller_end:
	.size	caller, .Lcaller_end-caller

	.globl	kernel
	.p2align	8
	.type	kernel,@function
kernel:
	s_endpgm
.Lkernel_end:
	.size	kernel, .Lkernel_end-kernel

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.globl	kernel.kd
	.type	kernel.kd,@object
	.size	kernel.kd, 64
	.protected	kernel
kernel.kd:
	.long	0
	.long	0
	.long	0
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
	.long	0
	.long	0
	.short	0
	.short	0
	.long	0

	.section	.amdgpu.info,"e",@progbits
	.byte	1
	.byte	8
	.quad	caller
	.byte	2
	.byte	4
	.long	0
	.byte	4
	.byte	4
	.long	4
	.byte	3
	.byte	4
	.long	4
	.byte	6
	.byte	4
	.long	0
	.byte	11
	.byte	4
	.long	8
	.byte	12
	.byte	4
	.long	64
	.byte	9
	.byte	4
	.long	0
	.byte	1
	.byte	8
	.quad	kernel
	.byte	2
	.byte	4
	.long	0
	.byte	4
	.byte	4
	.long	4
	.byte	3
	.byte	4
	.long	4
	.byte	6
	.byte	4
	.long	0
	.byte	11
	.byte	4
	.long	8
	.byte	12
	.byte	4
	.long	64
	.byte	8
	.byte	8
	.quad	caller

	.section	.amdgpu.strtab,"e",@progbits
	.ascii	"vi"

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
