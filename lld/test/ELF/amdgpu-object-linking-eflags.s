# REQUIRES: amdgpu

## Test that the linker rejects objects with incompatible AMDGPU e_flags before
## AMDGPU object linking.

# RUN: split-file %s %t

## --- Incompatible GPU architecture ---
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %t/func-gfx900.s -o %t/func-gfx900.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %t/func-gfx1030.s -o %t/func-gfx1030.o
# RUN: not ld.lld %t/func-gfx900.o %t/func-gfx1030.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARCH

# ARCH: error: incompatible mach:

## --- Incompatible xnack ---
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+xnack -filetype=obj %t/func-xnack-on.s -o %t/func-xnack-on.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-xnack -filetype=obj %t/func-xnack-off.s -o %t/func-xnack-off.o
# RUN: not ld.lld %t/func-xnack-on.o %t/func-xnack-off.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=XNACK

# XNACK: error: incompatible xnack:

## --- Incompatible sramecc ---
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=+sramecc -filetype=obj %t/func-sramecc-on.s -o %t/func-sramecc-on.o
# RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx908 -mattr=-sramecc -filetype=obj %t/func-sramecc-off.s -o %t/func-sramecc-off.o
# RUN: not ld.lld %t/func-sramecc-on.o %t/func-sramecc-off.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=SRAMECC
# RUN: ld.lld %t/func-sramecc-on.o -o %t/sramecc-on

# SRAMECC: error: incompatible sramecc:

#--- func-gfx900.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	func_a
	.p2align	2
	.type	func_a,@function
func_a:
	s_setpc_b64 s[30:31]
.Lfunc_end_a:
	.size	func_a, .Lfunc_end_a-func_a

	.amdgpu_info func_a
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

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

#--- func-gfx1030.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text
	.globl	func_b
	.p2align	2
	.type	func_b,@function
func_b:
	s_setpc_b64 s[30:31]
.Lfunc_end_b:
	.size	func_b, .Lfunc_end_b-func_b

	.amdgpu_info func_b
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 32
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

#--- func-xnack-on.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack+"
	.amdhsa_code_object_version 6
	.text
	.globl	func_c
	.p2align	2
	.type	func_c,@function
func_c:
	s_setpc_b64 s[30:31]
.Lfunc_end_c:
	.size	func_c, .Lfunc_end_c-func_c

	.amdgpu_info func_c
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900:xnack+
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- func-xnack-off.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack-"
	.amdhsa_code_object_version 6
	.text
	.globl	func_d
	.p2align	2
	.type	func_d,@function
func_d:
	s_setpc_b64 s[30:31]
.Lfunc_end_d:
	.size	func_d, .Lfunc_end_d-func_d

	.amdgpu_info func_d
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx900:xnack-
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- func-sramecc-on.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc+"
	.amdhsa_code_object_version 6
	.text
	.globl	func_e
	.p2align	2
	.type	func_e,@function
func_e:
	s_setpc_b64 s[30:31]
.Lfunc_end_e:
	.size	func_e, .Lfunc_end_e-func_e

	.amdgpu_info func_e
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx908:sramecc+
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

#--- func-sramecc-off.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc-"
	.amdhsa_code_object_version 6
	.text
	.globl	func_f
	.p2align	2
	.type	func_f,@function
func_f:
	s_setpc_b64 s[30:31]
.Lfunc_end_f:
	.size	func_f, .Lfunc_end_f-func_f

	.amdgpu_info func_f
		.amdgpu_flags 0
		.amdgpu_num_vgpr 4
		.amdgpu_num_sgpr 4
		.amdgpu_private_segment_size 0
		.amdgpu_occupancy 8
		.amdgpu_wave_size 64
	.end_amdgpu_info

	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels: []
amdhsa.target:   amdgcn-amd-amdhsa--gfx908:sramecc-
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
