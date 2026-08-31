# REQUIRES: amdgpu

## Verify that raw SHN_AMDGPU_LDS rescanning preserves normal common-symbol
## resolution semantics.

# RUN: split-file %s %t

## Multiple LDS declarations merge their maximum size and alignment. A common
## LDS declaration also overrides a weak regular definition.
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/dup-a.s -o %t/dup-a.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/dup-b.s -o %t/dup-b.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/weak.s -o %t/weak.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/dup-kernel.s -o %t/dup-kernel.o
# RUN: ld.lld %t/weak.o %t/dup-a.o %t/dup-b.o %t/dup-kernel.o -o %t/dup
# RUN: llvm-readobj --symbols %t/dup | FileCheck %s --check-prefix=MERGED

# MERGED:      Name: weak_lds
# MERGED-NEXT: Value: 0x30
# MERGED-NEXT: Size: 16
# MERGED:      Section: Absolute
# MERGED:      Name: merged
# MERGED-NEXT: Value: 0x0
# MERGED-NEXT: Size: 32
# MERGED:      Section: Absolute
# MERGED:      Name: prefix
# MERGED-NEXT: Value: 0x20
# MERGED-NEXT: Size: 16
# MERGED:      Section: Absolute

## A strong regular definition overrides an LDS common declaration.
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/strong-lds.s -o %t/strong-lds.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/strong-def.s -o %t/strong-def.o
# RUN: ld.lld %t/strong-lds.o %t/strong-def.o -o %t/strong
# RUN: llvm-readobj --symbols %t/strong | FileCheck %s --check-prefix=STRONG

# STRONG:      Name: strong_lds
# STRONG:      Size: 4
# STRONG:      Section: .data

## A regular common and an LDS common cannot describe the same symbol.
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/mixed-lds.s -o %t/mixed-lds.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/mixed-common.s -o %t/mixed-common.o
# RUN: not ld.lld %t/mixed-lds.o %t/mixed-common.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=MIXED

# MIXED: error: symbol 'mixed' is defined as both an AMDGPU LDS symbol and a regular common symbol

## --fortran-common must not extract an archive member whose definition is
## another common-like SHN_AMDGPU_LDS symbol.
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/archive-main.s -o %t/archive-main.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/archive-member.s -o %t/archive-member.o
# RUN: llvm-ar crs %t/liblds.a %t/archive-member.o
# RUN: ld.lld --fortran-common %t/archive-main.o %t/liblds.a -o %t/archive
# RUN: llvm-readobj --symbols %t/archive | FileCheck %s --check-prefix=ARCHIVE --implicit-check-not=member_marker

# ARCHIVE:      Name: archive_lds
# ARCHIVE-NEXT: Value: 0x0
# ARCHIVE-NEXT: Size: 8
# ARCHIVE:      Section: Absolute

#--- dup-a.s
	.globl	merged
	.amdgpu_lds merged, 32, 4

	.globl	prefix
	.amdgpu_lds prefix, 16, 16

	.globl	weak_lds
	.amdgpu_lds weak_lds, 16, 8

#--- dup-b.s
	.globl	merged
	.amdgpu_lds merged, 8, 64

#--- weak.s
	.data
	.weak	weak_lds
	.type	weak_lds,@object
weak_lds:
	.long	0
	.size	weak_lds, .-weak_lds

#--- dup-kernel.s
	.text
	.globl	dup_kernel
	.p2align	8
	.type	dup_kernel,@function
dup_kernel:
	s_endpgm
	.size	dup_kernel, .-dup_kernel

	.amdgpu_info dup_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use merged
		.amdgpu_use prefix
		.amdgpu_use weak_lds
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.section	.rodata,"a",@progbits
	.p2align	6
	.globl	dup_kernel.kd
	.type	dup_kernel.kd,@object
	.size	dup_kernel.kd, 64
dup_kernel.kd:
	.zero	64

#--- strong-lds.s
	.globl	strong_lds
	.amdgpu_lds strong_lds, 64, 16

#--- strong-def.s
	.data
	.globl	strong_lds
	.type	strong_lds,@object
strong_lds:
	.long	0
	.size	strong_lds, .-strong_lds

#--- mixed-lds.s
	.globl	mixed
	.amdgpu_lds mixed, 16, 16

#--- mixed-common.s
	.comm	mixed, 8, 4

#--- archive-main.s
	.globl	archive_lds
	.amdgpu_lds archive_lds, 8, 4

	.text
	.globl	archive_kernel
	.p2align	8
	.type	archive_kernel,@function
archive_kernel:
	s_endpgm
	.size	archive_kernel, .-archive_kernel

	.amdgpu_info archive_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use archive_lds
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.section	.rodata,"a",@progbits
	.p2align	6
	.globl	archive_kernel.kd
	.type	archive_kernel.kd,@object
	.size	archive_kernel.kd, 64
archive_kernel.kd:
	.zero	64

#--- archive-member.s
	.globl	archive_lds
	.amdgpu_lds archive_lds, 64, 64

	.text
	.globl	member_marker
	.type	member_marker,@function
member_marker:
	s_endpgm
	.size	member_marker, .-member_marker
