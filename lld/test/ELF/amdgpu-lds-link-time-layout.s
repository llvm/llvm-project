# REQUIRES: amdgpu

## Comprehensive test for the LDS layout algorithm. The linker sorts
## SHN_AMDGPU_LDS symbols by alignment (desc), size (desc), then name (asc)
## for deterministic output, and inserts padding for alignment.
##
## Symbols (deliberately supplied in a scrambled order across TUs):
##   lds_a16_s64   align=16 size=64   — tier 1 (highest alignment)
##   lds_a16_s32   align=16 size=32   — tier 1, smaller (size tiebreaker)
##   lds_a8_s20    align=8  size=20   — tier 2
##   lds_a4_s12_x  align=4  size=12   — tier 3 (name tiebreaker with y)
##   lds_a4_s12_y  align=4  size=12   — tier 3, same align+size, name > x
##   lds_a1_s3     align=1  size=3    — tier 4 (lowest alignment)
##
## Expected sorted order (alignment desc, size desc, name asc):
##   lds_a16_s64   align=16 size=64  -> offset 0x00  (0)
##   lds_a16_s32   align=16 size=32  -> offset 0x40  (64)
##   lds_a8_s20    align=8  size=20  -> offset 0x60  (96, 64+32=96 already 8-aligned)
##   lds_a4_s12_x  align=4  size=12  -> offset 0x74  (116, 96+20=116 already 4-aligned)
##   lds_a4_s12_y  align=4  size=12  -> offset 0x80  (128, 116+12=128 already 4-aligned)
##   lds_a1_s3     align=1  size=3   -> offset 0x8C  (140, 128+12=140 1-aligned trivially)
##
## Total: 143 bytes.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/tu1.s -o %t/tu1.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/tu2.s -o %t/tu2.o
# RUN: llvm-mc -filetype=obj -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %t/tu3.s -o %t/tu3.o
# RUN: ld.lld %t/tu1.o %t/tu2.o %t/tu3.o -o %t/out
# RUN: llvm-readobj --syms %t/out | FileCheck %s

## Symbols appear in file-definition order after linking (tu1, tu2, tu3).
## Verify each symbol's assigned offset matches the layout algorithm:
##   sort by alignment desc, size desc, name asc.

## From tu1: lds_a4_s12_y (align=4, size=12) -> offset 0x80
# CHECK:      Name: lds_a4_s12_y
# CHECK-NEXT: Value: 0x80
# CHECK-NEXT: Size: 12

## From tu1: lds_a16_s64 (align=16, size=64) -> offset 0x0
# CHECK:      Name: lds_a16_s64
# CHECK-NEXT: Value: 0x0
# CHECK-NEXT: Size: 64

## From tu2: lds_a1_s3 (align=1, size=3) -> offset 0x8C
# CHECK:      Name: lds_a1_s3
# CHECK-NEXT: Value: 0x8C
# CHECK-NEXT: Size: 3

## From tu2: lds_a8_s20 (align=8, size=20) -> offset 0x60
# CHECK:      Name: lds_a8_s20
# CHECK-NEXT: Value: 0x60
# CHECK-NEXT: Size: 20

## From tu3: lds_a16_s32 (align=16, size=32) -> offset 0x40
# CHECK:      Name: lds_a16_s32
# CHECK-NEXT: Value: 0x40
# CHECK-NEXT: Size: 32

## From tu3: lds_a4_s12_x (align=4, size=12) -> offset 0x74
# CHECK:      Name: lds_a4_s12_x
# CHECK-NEXT: Value: 0x74
# CHECK-NEXT: Size: 12

#--- tu1.s
## Deliberately interleave symbols from different tiers.
	.text
	.globl f1
	.p2align 8
	.type f1,@function
f1:
	s_mov_b32 s0, lds_a4_s12_y@abs32@lo
	s_mov_b32 s1, lds_a16_s64@abs32@lo
	s_endpgm
.Lf1_end:
	.size f1, .Lf1_end-f1

	.globl lds_a4_s12_y
	.amdgpu_lds lds_a4_s12_y, 12, 4

	.globl lds_a16_s64
	.amdgpu_lds lds_a16_s64, 64, 16

#--- tu2.s
	.text
	.globl f2
	.p2align 8
	.type f2,@function
f2:
	s_mov_b32 s0, lds_a1_s3@abs32@lo
	s_mov_b32 s1, lds_a8_s20@abs32@lo
	s_endpgm
.Lf2_end:
	.size f2, .Lf2_end-f2

	.globl lds_a1_s3
	.amdgpu_lds lds_a1_s3, 3, 1

	.globl lds_a8_s20
	.amdgpu_lds lds_a8_s20, 20, 8

#--- tu3.s
	.text
	.globl f3
	.p2align 8
	.type f3,@function
f3:
	s_mov_b32 s0, lds_a16_s32@abs32@lo
	s_mov_b32 s1, lds_a4_s12_x@abs32@lo
	s_endpgm
.Lf3_end:
	.size f3, .Lf3_end-f3

	.globl lds_a16_s32
	.amdgpu_lds lds_a16_s32, 32, 16

	.globl lds_a4_s12_x
	.amdgpu_lds lds_a4_s12_x, 12, 4
