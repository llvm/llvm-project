# Check that llvm-bolt is able to parse DWARF expressions in CFI instructions,
# store them in memory and correctly write them back to the output binary.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t && llvm-dwarfdump --eh-frame %t | FileCheck %s
#
# CHECK:       DW_CFA_advance_loc: 5
# CHECK-NEXT:  DW_CFA_def_cfa: R10 +0
# CHECK-NEXT:  DW_CFA_advance_loc: 9
# CHECK-NEXT:  DW_CFA_expression: RBP DW_OP_breg6 RBP+0
# CHECK-NEXT:  DW_CFA_advance_loc: 5
# CHECK-NEXT:  DW_CFA_def_cfa_expression: DW_OP_breg6 RBP-8, DW_OP_deref
# CHECK-NEXT:  DW_CFA_advance_loc2: 3130
# CHECK-NEXT:  DW_CFA_def_cfa: R10 +0
# CHECK-NEXT:  DW_CFA_advance_loc: 5
# CHECK-NEXT:  DW_CFA_def_cfa: RSP +8

	.text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 0
	.cfi_startproc
.LBB06:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	blake2b_compress_avx2
	movl	$0x0, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size main, .-main

  .globl blake2b_compress_avx2
  .type blake2b_compress_avx2, %function
blake2b_compress_avx2:
# FDATA: 0 [unknown] 0 1 blake2b_compress_avx2 0 0 0
	.cfi_startproc
.LBB07:
	leaq	0x8(%rsp), %r10
	.cfi_def_cfa %r10, 0
	andq	$-0x20, %rsp
	pushq	-0x8(%r10)
	pushq	%rbp
	.cfi_escape 0x10, 0x06, 0x02, 0x76, 0x00 #
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0x0f, 0x03, 0x76, 0x78, 0x06 #
	subq	$0x170, %rsp
	vbroadcasti128	(%rsi), %ymm10
	vbroadcasti128	0x10(%rsi), %ymm8
	vbroadcasti128	0x30(%rsi), %ymm7
	vbroadcasti128	0x70(%rsi), %ymm4
	vpunpcklqdq	%ymm8, %ymm10, %ymm0
	vbroadcasti128	0x20(%rsi), %ymm11
	vmovdqa	%ymm7, -0x30(%rbp)
	vpunpcklqdq	-0x30(%rbp), %ymm11, %ymm1
	vmovdqa	%ymm4, %ymm15
	vpblendd	$0xf0, %ymm1, %ymm0, %ymm4
	vpaddq	(%rdi), %ymm4, %ymm13
	vpaddq	0x20(%rdi), %ymm13, %ymm13
	vmovdqa	%ymm4, -0x190(%rbp)
	vbroadcasti128	0x60(%rsi), %ymm7
	vbroadcasti128	0x50(%rsi), %ymm6
	vbroadcasti128	0x40(%rsi), %ymm9
	vpunpcklqdq	%ymm15, %ymm7, %ymm14
	vmovq	0x50(%rdi), %xmm5
	vpinsrq	$0x1, 0x58(%rdi), %xmm5, %xmm1
	vmovq	0x40(%rdi), %xmm5
	vpinsrq	$0x1, 0x48(%rdi), %xmm5, %xmm0
	vinserti128	$0x1, %xmm1, %ymm0, %ymm1
	vpunpckhqdq	%ymm8, %ymm10, %ymm0
	vpxor	"blake2b_IV/1"+32(%rip), %ymm1, %ymm1
	vpunpckhqdq	-0x30(%rbp), %ymm11, %ymm4
	vpblendd	$0xf0, %ymm4, %ymm0, %ymm0
	vmovdqa	DATAat0x401380(%rip), %ymm5
	vmovdqa	%ymm0, -0xb0(%rbp)
	vpxor	%ymm13, %ymm1, %ymm1
	vpaddq	-0xb0(%rbp), %ymm13, %ymm13
	vpshufd	$0xb1, %ymm1, %ymm1
	vmovdqa	DATAat0x4013a0(%rip), %ymm4
	vpaddq	"blake2b_IV/1"(%rip), %ymm1, %ymm3
	vpxor	0x20(%rdi), %ymm3, %ymm2
	vpshufb	%ymm5, %ymm2, %ymm2
	vpaddq	%ymm2, %ymm13, %ymm13
	vpxor	%ymm1, %ymm13, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm2, %ymm3, %ymm2
	vpsrlq	$0x3f, %ymm2, %ymm12
	vpaddq	%ymm2, %ymm2, %ymm2
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm2, %ymm2
	vpermq	$0x39, %ymm2, %ymm0
	vpunpcklqdq	%ymm6, %ymm9, %ymm2
	vpunpckhqdq	%ymm15, %ymm7, %ymm12
	vpblendd	$0xf0, %ymm14, %ymm2, %ymm2
	vmovdqa	%ymm2, -0xd0(%rbp)
	vpaddq	-0xd0(%rbp), %ymm13, %ymm13
	vpunpckhqdq	%ymm6, %ymm9, %ymm2
	vpblendd	$0xf0, %ymm12, %ymm2, %ymm2
	vmovdqa	%ymm2, -0xf0(%rbp)
	vmovdqa	%ymm15, %ymm12
	vpaddq	%ymm0, %ymm13, %ymm13
	vmovdqa	%ymm12, -0x50(%rbp)
	vpxor	%ymm13, %ymm1, %ymm1
	vpaddq	-0xf0(%rbp), %ymm13, %ymm13
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm13, %ymm13
	vpxor	%ymm1, %ymm13, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm0
	vpunpcklqdq	%ymm11, %ymm15, %ymm2
	vmovdqa	%ymm2, %ymm15
	vpunpckhqdq	%ymm7, %ymm9, %ymm2
	vpblendd	$0xf0, %ymm2, %ymm15, %ymm2
	vmovdqa	%ymm15, -0x1b0(%rbp)
	vpermq	$0x93, %ymm0, %ymm0
	vmovdqa	%ymm2, -0x110(%rbp)
	vpunpcklqdq	%ymm9, %ymm6, %ymm2
	vpaddq	-0x110(%rbp), %ymm13, %ymm13
	vmovdqa	%ymm2, %ymm15
	vmovdqa	-0x30(%rbp), %ymm2
	vpalignr	$0x8, -0x50(%rbp), %ymm2, %ymm2
	vmovdqa	%ymm15, -0x1d0(%rbp)
	vpaddq	%ymm0, %ymm13, %ymm13
	vpblendd	$0xf0, %ymm2, %ymm15, %ymm12
	vpxor	%ymm13, %ymm1, %ymm1
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vmovdqa	%ymm12, -0x130(%rbp)
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	-0x130(%rbp), %ymm13, %ymm13
	vmovdqa	-0x30(%rbp), %ymm15
	vpunpckhqdq	%ymm11, %ymm6, %ymm12
	vpaddq	%ymm0, %ymm13, %ymm13
	vpxor	%ymm1, %ymm13, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm0
	vpshufd	$0x4e, %ymm10, %ymm2
	vpblendd	$0xf0, %ymm12, %ymm2, %ymm12
	vpermq	$0x39, %ymm0, %ymm0
	vmovdqa	%ymm12, -0x150(%rbp)
	vpunpckhqdq	%ymm8, %ymm15, %ymm12
	vpunpcklqdq	%ymm8, %ymm7, %ymm2
	vpaddq	-0x150(%rbp), %ymm13, %ymm13
	vmovdqa	%ymm12, -0x70(%rbp)
	vmovdqa	-0x50(%rbp), %ymm15
	vpblendd	$0xf0, -0x70(%rbp), %ymm2, %ymm2
	vpaddq	%ymm0, %ymm13, %ymm13
	vmovdqa	%ymm2, -0x170(%rbp)
	vpxor	%ymm13, %ymm1, %ymm1
	vpunpckhqdq	%ymm15, %ymm11, %ymm12
	vpaddq	-0x170(%rbp), %ymm13, %ymm13
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vmovdqa	%ymm12, -0x90(%rbp)
	vpunpcklqdq	%ymm10, %ymm9, %ymm12
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm13, %ymm13
	vpxor	%ymm1, %ymm13, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm0
	vpalignr	$0x8, %ymm6, %ymm7, %ymm2
	vpermq	$0x93, %ymm0, %ymm0
	vpblendd	$0xf0, -0x90(%rbp), %ymm2, %ymm2
	vpaddq	%ymm2, %ymm13, %ymm2
	vpblendd	$0x33, %ymm8, %ymm7, %ymm13
	vpaddq	%ymm0, %ymm2, %ymm2
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vmovdqa	-0x30(%rbp), %ymm13
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm12
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm12, %ymm12
	vpunpckhqdq	%ymm9, %ymm13, %ymm13
	vpxor	%ymm1, %ymm12, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm0
	vpblendd	$0x33, %ymm6, %ymm8, %ymm2
	vpermq	$0x39, %ymm0, %ymm0
	vpblendd	$0xf0, %ymm13, %ymm2, %ymm13
	vpunpcklqdq	-0x30(%rbp), %ymm15, %ymm2
	vmovdqa	-0x70(%rbp), %ymm15
	vpaddq	%ymm13, %ymm12, %ymm13
	vpalignr	$0x8, %ymm10, %ymm11, %ymm12
	vpaddq	%ymm0, %ymm13, %ymm13
	vpblendd	$0xf0, %ymm12, %ymm2, %ymm2
	vpxor	%ymm13, %ymm1, %ymm1
	vpaddq	%ymm2, %ymm13, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpunpckhqdq	%ymm10, %ymm9, %ymm13
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpblendd	$0xf0, %ymm14, %ymm13, %ymm13
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x39, %ymm1, %ymm1
	vpor	%ymm12, %ymm0, %ymm0
	vpunpckhqdq	%ymm6, %ymm7, %ymm12
	vpermq	$0x93, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpblendd	$0xf0, %ymm12, %ymm15, %ymm12
	vmovdqa	-0x50(%rbp), %ymm15
	vpaddq	%ymm12, %ymm2, %ymm12
	vpaddq	%ymm0, %ymm12, %ymm12
	vpblendd	$0x33, %ymm11, %ymm15, %ymm14
	vpunpcklqdq	%ymm6, %ymm8, %ymm15
	vpxor	%ymm12, %ymm1, %ymm1
	vpaddq	%ymm13, %ymm12, %ymm13
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm13, %ymm13
	vpxor	%ymm1, %ymm13, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm0
	vpblendd	$0x33, %ymm8, %ymm11, %ymm2
	vpermq	$0x39, %ymm0, %ymm12
	vmovdqa	-0x30(%rbp), %ymm0
	vpblendd	$0xf0, %ymm14, %ymm2, %ymm2
	vpaddq	%ymm2, %ymm13, %ymm2
	vpunpcklqdq	%ymm6, %ymm0, %ymm13
	vmovdqa	%ymm13, %ymm0
	vpunpcklqdq	%ymm9, %ymm10, %ymm13
	vmovdqa	%ymm0, -0x1f0(%rbp)
	vpaddq	%ymm12, %ymm2, %ymm2
	vpblendd	$0xf0, %ymm13, %ymm0, %ymm0
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm0, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm12, %ymm3, %ymm12
	vpshufb	%ymm5, %ymm12, %ymm12
	vpaddq	%ymm12, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm12, %ymm3, %ymm12
	vpsrlq	$0x3f, %ymm12, %ymm0
	vpaddq	%ymm12, %ymm12, %ymm12
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm0, %ymm12, %ymm12
	vpermq	$0x93, %ymm12, %ymm0
	vpunpckhqdq	%ymm11, %ymm9, %ymm12
	vpblendd	$0xf0, %ymm15, %ymm12, %ymm12
	vmovdqa	-0x30(%rbp), %ymm15
	vpaddq	%ymm12, %ymm2, %ymm2
	vpaddq	%ymm0, %ymm2, %ymm2
	vpblendd	$0x33, %ymm10, %ymm15, %ymm12
	vpxor	%ymm2, %ymm1, %ymm1
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpblendd	$0xf0, %ymm14, %ymm12, %ymm14
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vmovdqa	-0x50(%rbp), %ymm15
	vpaddq	%ymm14, %ymm2, %ymm2
	vpblendd	$0x33, -0x30(%rbp), %ymm8, %ymm14
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpblendd	$0x33, %ymm15, %ymm6, %ymm12
	vpermq	$0x39, %ymm0, %ymm0
	vpblendd	$0xf0, %ymm14, %ymm12, %ymm12
	vpblendd	$0x33, %ymm9, %ymm7, %ymm14
	vpaddq	%ymm12, %ymm2, %ymm2
	vpalignr	$0x8, %ymm10, %ymm7, %ymm12
	vpaddq	%ymm0, %ymm2, %ymm2
	vpblendd	$0xf0, %ymm14, %ymm12, %ymm12
	vmovdqa	-0x30(%rbp), %ymm14
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpunpcklqdq	-0x30(%rbp), %ymm8, %ymm12
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm13
	vpermq	$0x93, %ymm0, %ymm0
	vpaddq	%ymm13, %ymm2, %ymm2
	vpunpcklqdq	%ymm6, %ymm7, %ymm12
	vpunpckhqdq	%ymm8, %ymm6, %ymm13
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vpaddq	%ymm0, %ymm2, %ymm2
	vpunpckhqdq	%ymm10, %ymm15, %ymm13
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpblendd	$0x33, %ymm11, %ymm14, %ymm12
	vmovdqa	%ymm15, %ymm14
	vpermq	$0x39, %ymm0, %ymm0
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vpblendd	$0x33, %ymm14, %ymm9, %ymm13
	vmovdqa	-0x90(%rbp), %ymm15
	vpaddq	%ymm12, %ymm2, %ymm2
	vpunpckhqdq	%ymm11, %ymm7, %ymm12
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vpshufd	$0x4e, %ymm9, %ymm13
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpblendd	$0x33, %ymm7, %ymm10, %ymm12
	vpermq	$0x93, %ymm0, %ymm0
	vpblendd	$0xf0, -0x1b0(%rbp), %ymm12, %ymm12
	vpaddq	%ymm12, %ymm2, %ymm2
	vpalignr	$0x8, %ymm7, %ymm6, %ymm12
	vpaddq	%ymm0, %ymm2, %ymm2
	vpblendd	$0xf0, %ymm12, %ymm15, %ymm12
	vmovdqa	-0x70(%rbp), %ymm15
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpunpcklqdq	-0x30(%rbp), %ymm10, %ymm12
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vpermq	$0x39, %ymm0, %ymm0
	vpaddq	%ymm12, %ymm2, %ymm2
	vpblendd	$0x33, %ymm8, %ymm6, %ymm12
	vpblendd	$0x33, %ymm7, %ymm8, %ymm13
	vpaddq	%ymm0, %ymm2, %ymm2
	vpblendd	$0xf0, %ymm12, %ymm15, %ymm12
	vmovdqa	-0x90(%rbp), %ymm15
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpunpckhqdq	-0x30(%rbp), %ymm7, %ymm12
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vmovdqa	%ymm14, %ymm13
	vpaddq	%ymm12, %ymm2, %ymm2
	vpalignr	$0x8, %ymm6, %ymm14, %ymm14
	vpunpckhqdq	%ymm9, %ymm10, %ymm12
	vpermq	$0x93, %ymm0, %ymm0
	vpblendd	$0xf0, %ymm12, %ymm14, %ymm12
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vpunpcklqdq	%ymm8, %ymm9, %ymm12
	vpblendd	$0xf0, %ymm12, %ymm15, %ymm12
	vpermq	$0x39, %ymm0, %ymm0
	vpaddq	%ymm12, %ymm2, %ymm2
	vpunpcklqdq	%ymm11, %ymm10, %ymm12
	vmovdqa	%ymm13, %ymm15
	vpblendd	$0xf0, -0x1f0(%rbp), %ymm12, %ymm12
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	%ymm12, %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm12
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm12, %ymm0, %ymm0
	vmovdqa	-0x30(%rbp), %ymm12
	vpermq	$0x93, %ymm0, %ymm0
	vpunpcklqdq	%ymm13, %ymm12, %ymm12
	vpalignr	$0x8, %ymm6, %ymm10, %ymm13
	vpblendd	$0xf0, %ymm13, %ymm12, %ymm12
	vpunpckhqdq	%ymm9, %ymm15, %ymm13
	vpalignr	$0x8, %ymm8, %ymm9, %ymm9
	vpaddq	%ymm12, %ymm2, %ymm12
	vpalignr	$0x8, %ymm10, %ymm6, %ymm6
	vpblendd	$0xf0, %ymm9, %ymm13, %ymm9
	vpaddq	%ymm0, %ymm12, %ymm12
	vpblendd	$0xf0, %ymm6, %ymm7, %ymm6
	vpxor	%ymm12, %ymm1, %ymm1
	vpaddq	%ymm9, %ymm12, %ymm12
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm0, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm12, %ymm12
	vpxor	%ymm1, %ymm12, %ymm1
	vpaddq	%ymm6, %ymm12, %ymm12
	vpshufb	%ymm4, %ymm1, %ymm1
	vmovdqa	-0x30(%rbp), %ymm6
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x93, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm0
	vpblendd	$0x33, %ymm8, %ymm6, %ymm2
	vpermq	$0x39, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm12, %ymm12
	vpblendd	$0xf0, %ymm11, %ymm2, %ymm2
	vpxor	%ymm12, %ymm1, %ymm1
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpaddq	%ymm2, %ymm12, %ymm12
	vpxor	%ymm3, %ymm0, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vmovdqa	-0x30(%rbp), %ymm6
	vpaddq	%ymm0, %ymm12, %ymm12
	vpxor	%ymm1, %ymm12, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpermq	$0x39, %ymm1, %ymm1
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x4e, %ymm3, %ymm3
	vpor	%ymm2, %ymm0, %ymm2
	vpunpckhqdq	%ymm10, %ymm6, %ymm0
	vmovdqa	-0x1d0(%rbp), %ymm6
	vpermq	$0x93, %ymm2, %ymm2
	vpblendd	$0xf0, %ymm0, %ymm6, %ymm0
	vpaddq	%ymm0, %ymm12, %ymm12
	vpunpcklqdq	%ymm11, %ymm8, %ymm0
	vpblendd	$0x33, -0x30(%rbp), %ymm11, %ymm11
	vpunpckhqdq	%ymm7, %ymm8, %ymm8
	vpaddq	%ymm2, %ymm12, %ymm12
	vpblendd	$0xf0, %ymm11, %ymm0, %ymm11
	vpblendd	$0xf0, %ymm8, %ymm13, %ymm8
	vpunpcklqdq	%ymm10, %ymm7, %ymm7
	vpxor	%ymm12, %ymm1, %ymm1
	vpaddq	%ymm11, %ymm12, %ymm12
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm2, %ymm2
	vpshufb	%ymm5, %ymm2, %ymm2
	vpaddq	%ymm2, %ymm12, %ymm12
	vpblendd	$0xf0, %ymm7, %ymm14, %ymm7
	vpxor	%ymm1, %ymm12, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpaddq	%ymm8, %ymm12, %ymm8
	vpxor	%ymm2, %ymm3, %ymm2
	vpsrlq	$0x3f, %ymm2, %ymm0
	vpaddq	%ymm2, %ymm2, %ymm2
	vpermq	$0x93, %ymm1, %ymm1
	vpor	%ymm0, %ymm2, %ymm2
	vpermq	$0x39, %ymm2, %ymm2
	vpaddq	%ymm2, %ymm8, %ymm8
	vpermq	$0x4e, %ymm3, %ymm3
	vpxor	%ymm8, %ymm1, %ymm1
	vpaddq	%ymm7, %ymm8, %ymm8
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm2, %ymm2
	vpshufb	%ymm5, %ymm2, %ymm2
	vpaddq	%ymm2, %ymm8, %ymm8
	vpxor	%ymm1, %ymm8, %ymm1
	vpaddq	-0x190(%rbp), %ymm8, %ymm8
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm2, %ymm3, %ymm2
	vpsrlq	$0x3f, %ymm2, %ymm0
	vpaddq	%ymm2, %ymm2, %ymm2
	vpermq	$0x39, %ymm1, %ymm1
	vpor	%ymm0, %ymm2, %ymm2
	vpermq	$0x93, %ymm2, %ymm0
	vpaddq	%ymm0, %ymm8, %ymm8
	vpermq	$0x4e, %ymm3, %ymm3
	vpxor	%ymm8, %ymm1, %ymm1
	vpaddq	-0xb0(%rbp), %ymm8, %ymm8
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm0, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm8, %ymm8
	vpxor	%ymm1, %ymm8, %ymm1
	vpaddq	-0xd0(%rbp), %ymm8, %ymm8
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpaddq	%ymm0, %ymm0, %ymm6
	vpermq	$0x93, %ymm1, %ymm1
	vpermq	$0x4e, %ymm3, %ymm3
	vpsrlq	$0x3f, %ymm0, %ymm2
	vpor	%ymm2, %ymm6, %ymm6
	vpermq	$0x39, %ymm6, %ymm6
	vpaddq	%ymm6, %ymm8, %ymm2
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	-0xf0(%rbp), %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm6, %ymm6
	vpshufb	%ymm5, %ymm6, %ymm6
	vpaddq	%ymm6, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpaddq	-0x110(%rbp), %ymm2, %ymm2
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm6, %ymm3, %ymm6
	vpsrlq	$0x3f, %ymm6, %ymm0
	vpaddq	%ymm6, %ymm6, %ymm6
	vpermq	$0x39, %ymm1, %ymm1
	vpor	%ymm0, %ymm6, %ymm0
	vpermq	$0x93, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpermq	$0x4e, %ymm3, %ymm3
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	-0x130(%rbp), %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm0, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpaddq	-0x150(%rbp), %ymm2, %ymm2
	vpshufb	%ymm4, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm0, %ymm3, %ymm0
	vpsrlq	$0x3f, %ymm0, %ymm6
	vpaddq	%ymm0, %ymm0, %ymm0
	vpermq	$0x93, %ymm1, %ymm1
	vpor	%ymm6, %ymm0, %ymm0
	vpermq	$0x39, %ymm0, %ymm0
	vpaddq	%ymm0, %ymm2, %ymm2
	vpermq	$0x4e, %ymm3, %ymm3
	vpxor	%ymm2, %ymm1, %ymm1
	vpaddq	-0x170(%rbp), %ymm2, %ymm2
	vpshufd	$0xb1, %ymm1, %ymm1
	vpaddq	%ymm1, %ymm3, %ymm3
	vpxor	%ymm3, %ymm0, %ymm0
	vpshufb	%ymm5, %ymm0, %ymm5
	vpaddq	%ymm5, %ymm2, %ymm2
	vpxor	%ymm1, %ymm2, %ymm1
	vpshufb	%ymm4, %ymm1, %ymm4
	vpaddq	%ymm4, %ymm3, %ymm0
	vpermq	$0x39, %ymm4, %ymm4
	vpxor	%ymm5, %ymm0, %ymm5
	vpsrlq	$0x3f, %ymm5, %ymm1
	vpaddq	%ymm5, %ymm5, %ymm5
	vpermq	$0x4e, %ymm0, %ymm0
	vpor	%ymm1, %ymm5, %ymm5
	vpxor	(%rdi), %ymm0, %ymm0
	vpermq	$0x93, %ymm5, %ymm5
	vpxor	%ymm2, %ymm0, %ymm2
	vmovdqu	%ymm2, (%rdi)
	vpxor	0x20(%rdi), %ymm5, %ymm5
	vpxor	%ymm4, %ymm5, %ymm4
	vmovdqu	%ymm4, 0x20(%rdi)
	vzeroupper
	xorl	%eax, %eax
	addq	$0x170, %rsp
	popq	%r10
	.cfi_def_cfa %r10, 0
	popq	%rbp
	leaq	-0x8(%r10), %rsp
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size blake2b_compress_avx2, .-blake2b_compress_avx2
.section .rodata
"blake2b_IV/1":
"DATAat0x4013a0":
"DATAat0x401380":
