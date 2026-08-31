# REQUIRES: amdgpu

## Test that lld resolves SHN_AMDGPU_LDS symbols to absolute symbols with
## assigned offsets, and patches R_AMDGPU_ABS32_LO relocations in code.

## Build two object files with LDS symbols and one kernel object that uses both
## symbols.
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/kernel.s -o %t/kernel.o

## Verify that the object files have SHN_AMDGPU_LDS symbols.
# RUN: llvm-readobj --syms %t/a.o | FileCheck %s --check-prefix=OBJ-A
# RUN: llvm-readobj --syms %t/b.o | FileCheck %s --check-prefix=OBJ-B

## A final link cannot assign LDS without a kernel descriptor.
# RUN: not ld.lld %t/a.o %t/b.o -o %t/no-kernel 2>&1 | FileCheck %s --check-prefix=NO-KERNEL

## Link and verify the symbols are resolved to absolute with correct offsets.
# RUN: ld.lld %t/a.o %t/b.o %t/kernel.o -o %t/out
# RUN: llvm-readobj --syms %t/out | FileCheck %s --check-prefix=LINKED
# RUN: llvm-objdump -d %t/out | FileCheck %s --check-prefix=DISASM

# NO-KERNEL: error: cannot resolve AMDGPU LDS symbols without a kernel descriptor and .amdgpu.info metadata
# NO-KERNEL-NOT: common symbol reached writer
# NO-KERNEL-NOT: UNREACHABLE

# OBJ-A:      Symbol {
# OBJ-A:        Name: lds_a
# OBJ-A-NEXT:   Value: 0x10
# OBJ-A-NEXT:   Size: 256
# OBJ-A-NEXT:   Binding: Global
# OBJ-A-NEXT:   Type: Object
# OBJ-A-NEXT:   Other: 0
# OBJ-A-NEXT:   Section: Processor Specific (0xFF00)
# OBJ-A-NEXT: }

# OBJ-B:      Symbol {
# OBJ-B:        Name: lds_b
# OBJ-B-NEXT:   Value: 0x4
# OBJ-B-NEXT:   Size: 128
# OBJ-B-NEXT:   Binding: Global
# OBJ-B-NEXT:   Type: Object
# OBJ-B-NEXT:   Other: 0
# OBJ-B-NEXT:   Section: Processor Specific (0xFF00)
# OBJ-B-NEXT: }

## After linking, LDS symbols become absolute with assigned offsets.
## lds_a: align=16, size=256 -> offset 0
## lds_b: align=4,  size=128 -> offset 256

# LINKED:      Symbol {
# LINKED:        Name: lds_a
# LINKED-NEXT:   Value: 0x0
# LINKED-NEXT:   Size: 256
# LINKED-NEXT:   Binding: Global
# LINKED-NEXT:   Type: None
# LINKED:        Section: Absolute
# LINKED-NEXT: }

# LINKED:      Symbol {
# LINKED:        Name: lds_b
# LINKED-NEXT:   Value: 0x100
# LINKED-NEXT:   Size: 128
# LINKED-NEXT:   Binding: Global
# LINKED-NEXT:   Type: None
# LINKED:        Section: Absolute
# LINKED-NEXT: }

## The s_mov_b32 instructions should be patched with the resolved offsets.
# DISASM: s_mov_b32 s0, lit(0x0)
# DISASM: s_mov_b32 s0, 0x100

#--- a.s
	.text
	.globl use_lds_a
	.p2align 8
	.type use_lds_a,@function
use_lds_a:
	s_mov_b32 s0, lds_a@abs32@lo
	v_lshl_add_u32 v1, v0, 2, s0
	ds_read_b32 v2, v1
	s_endpgm
.Lfunc_end_a:
	.size use_lds_a, .Lfunc_end_a-use_lds_a

	.globl lds_a
	.amdgpu_lds lds_a, 256, 16

#--- b.s
	.text
	.globl use_lds_b
	.p2align 8
	.type use_lds_b,@function
use_lds_b:
	s_mov_b32 s0, lds_b@abs32@lo
	v_lshl_add_u32 v1, v0, 2, s0
	ds_write_b32 v1, v2
	s_endpgm
.Lfunc_end_b:
	.size use_lds_b, .Lfunc_end_b-use_lds_b

	.globl lds_b
	.amdgpu_lds lds_b, 128, 4

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
		.amdgpu_use lds_a
		.amdgpu_use lds_b
		.amdgpu_occupancy 4
	.end_amdgpu_info

	.section .rodata,"a",@progbits
	.p2align 6
	.globl test_kernel.kd
	.type test_kernel.kd,@object
	.size test_kernel.kd, 64
test_kernel.kd:
	.zero 64
