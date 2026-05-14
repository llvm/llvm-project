# REQUIRES: amdgpu

## Test that the linker inserts padding between LDS symbols when a symbol's
## size is not a multiple of the next symbol's alignment requirement.
##
## Symbols:
##   lds_big   align=16 size=20  — leaves offset at 20, not 8-aligned
##   lds_med   align=8  size=7   — needs padding to offset 24, leaves at 31
##   lds_small align=4  size=5   — needs padding to offset 32, leaves at 37
##   lds_tiny  align=1  size=1   — no padding needed, offset 37
##
## Expected layout (alignment desc, size desc):
##   lds_big   -> offset 0x00  (0)
##   lds_med   -> offset 0x18  (24 = alignTo(20, 8), 4 bytes padding)
##   lds_small -> offset 0x20  (32 = alignTo(31, 4), 1 byte padding)
##   lds_tiny  -> offset 0x25  (37 = alignTo(37, 1), no padding)
##
## Total: 38 bytes. Padding: 4 bytes at [20,24) + 1 byte at [31,32).

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/b.s -o %t/b.o
# RUN: ld.lld %t/a.o %t/b.o -o %t/out
# RUN: llvm-readobj --syms %t/out | FileCheck %s

## Symbols appear in file-definition order (a.o then b.o).

## From a.o: lds_tiny (align=1, size=1) -> offset 0x25 (no padding from lds_small)
# CHECK:      Name: lds_tiny
# CHECK-NEXT: Value: 0x25
# CHECK-NEXT: Size: 1

## From a.o: lds_big (align=16, size=20) -> offset 0x0
# CHECK:      Name: lds_big
# CHECK-NEXT: Value: 0x0
# CHECK-NEXT: Size: 20

## From b.o: lds_small (align=4, size=5) -> offset 0x20 (1 byte padding from 31)
# CHECK:      Name: lds_small
# CHECK-NEXT: Value: 0x20
# CHECK-NEXT: Size: 5

## From b.o: lds_med (align=8, size=7) -> offset 0x18 (4 bytes padding from 20)
# CHECK:      Name: lds_med
# CHECK-NEXT: Value: 0x18
# CHECK-NEXT: Size: 7

#--- a.s
	.text
	.globl f1
	.p2align 8
	.type f1,@function
f1:
	s_mov_b32 s0, lds_tiny@abs32@lo
	s_mov_b32 s1, lds_big@abs32@lo
	s_endpgm
.Lf1_end:
	.size f1, .Lf1_end-f1

	.globl lds_tiny
	.amdgpu_lds lds_tiny, 1, 1

	.globl lds_big
	.amdgpu_lds lds_big, 20, 16

#--- b.s
	.text
	.globl f2
	.p2align 8
	.type f2,@function
f2:
	s_mov_b32 s0, lds_small@abs32@lo
	s_mov_b32 s1, lds_med@abs32@lo
	s_endpgm
.Lf2_end:
	.size f2, .Lf2_end-f2

	.globl lds_small
	.amdgpu_lds lds_small, 5, 4

	.globl lds_med
	.amdgpu_lds lds_med, 7, 8
