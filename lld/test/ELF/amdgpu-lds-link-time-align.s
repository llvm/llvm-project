# REQUIRES: amdgpu

## Test that lld sorts LDS symbols by alignment (descending), then size
## (descending), and correctly inserts padding between symbols.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/c.s -o %t/c.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/kernel.s -o %t/kernel.o
# RUN: ld.lld %t/a.o %t/b.o %t/c.o %t/kernel.o -o %t/out
# RUN: llvm-readobj --syms %t/out | FileCheck %s

## Three LDS symbols with varied alignments and sizes:
##   lds_small:  align=4,  size=12  (from a.o)
##   lds_big:    align=16, size=64  (from b.o)
##   lds_medium: align=8,  size=32  (from c.o)
##
## After sorting by alignment (desc), then size (desc):
##   lds_big    (align=16, size=64) -> offset 0
##   lds_medium (align=8,  size=32) -> offset 64
##   lds_small  (align=4,  size=12) -> offset 96

## Symbols appear in input file order (a.o first). Check each with its offset.
# CHECK:      Name: lds_small
# CHECK-NEXT: Value: 0x60
# CHECK-NEXT: Size: 12
# CHECK:      Section: Absolute

# CHECK:      Name: lds_big
# CHECK-NEXT: Value: 0x0
# CHECK-NEXT: Size: 64
# CHECK:      Section: Absolute

# CHECK:      Name: lds_medium
# CHECK-NEXT: Value: 0x40
# CHECK-NEXT: Size: 32
# CHECK:      Section: Absolute

#--- a.s
	.text
	.globl use_lds_small
	.p2align 8
	.type use_lds_small,@function
use_lds_small:
	s_mov_b32 s0, lds_small@abs32@lo
	s_endpgm
.Lfunc_end_a:
	.size use_lds_small, .Lfunc_end_a-use_lds_small

	.globl lds_small
	.amdgpu_lds lds_small, 12, 4

#--- b.s
	.text
	.globl use_lds_big
	.p2align 8
	.type use_lds_big,@function
use_lds_big:
	s_mov_b32 s0, lds_big@abs32@lo
	s_endpgm
.Lfunc_end_b:
	.size use_lds_big, .Lfunc_end_b-use_lds_big

	.globl lds_big
	.amdgpu_lds lds_big, 64, 16

#--- c.s
	.text
	.globl use_lds_medium
	.p2align 8
	.type use_lds_medium,@function
use_lds_medium:
	s_mov_b32 s0, lds_medium@abs32@lo
	s_endpgm
.Lfunc_end_c:
	.size use_lds_medium, .Lfunc_end_c-use_lds_medium

	.globl lds_medium
	.amdgpu_lds lds_medium, 32, 8

#--- kernel.s
	.text
	.globl test_kernel
	.p2align 8
	.type test_kernel,@function
test_kernel:
	s_endpgm
.Lfunc_end_kernel:
	.size test_kernel, .Lfunc_end_kernel-test_kernel

	.amdgpu_info test_kernel
		.amdgpu_flags 3
		.amdgpu_num_vgpr 32
		.amdgpu_num_agpr 0
		.amdgpu_num_sgpr 33
		.amdgpu_private_segment_size 0
		.amdgpu_use lds_small
		.amdgpu_use lds_big
		.amdgpu_use lds_medium
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.section .rodata,"a",@progbits
	.p2align 6
	.globl test_kernel.kd
	.type test_kernel.kd,@object
	.size test_kernel.kd, 64
test_kernel.kd:
	.zero 64
