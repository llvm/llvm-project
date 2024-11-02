# Check that llvm-bolt is able to read a file with DWARF Exception CFI
# information and fix CFI information after reordering.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clangxx %cflags %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t --reorder-blocks=cache --print-after-lowering \
# RUN:   --print-only=_Z10SolveCubicddddPiPd 2>&1 | FileCheck %s
#
# Entry BB
# CHECK:      divsd   %xmm0, %xmm1
# CHECK:      pushq   %rbx
# CHECK:      !CFI    $0      ; OpDefCfaOffset
# CHECK:      !CFI    $1      ; OpOffset
# CHECK:      movq    %rsi, %rbx
# CHECK:      subq    $0x70, %rsp
# CHECK:      !CFI    $2      ; OpDefCfaOffset
# CHECK:      divsd   %xmm0, %xmm2
# Duplicated tail
# CHECK:      addq    $0x70, %rsp
# CHECK:      !CFI    $3      ; OpDefCfaOffset
# CHECK:      popq    %rbx
# CHECK:      !CFI    $4      ; OpDefCfaOffset
# CHECK:      retq
# CHECK:      !CFI    {{.*}}  ; OpDefCfa
# Epilogue rescheduled to the middle of the function
# CHECK:      addq    $0x70, %rsp
# CHECK:      !CFI    $6      ; OpDefCfaOffset
# CHECK:      popq    %rbx
# CHECK:      !CFI    $7      ; OpDefCfaOffset
# CHECK:      retq
# CHECK:      !CFI    {{.*}}  ; OpDefCfa

	.text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 0
	.cfi_startproc
LBB00:
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset %r15, -16
	movl	$0x401520, %edi
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset %r14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset %r13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset %r12, -40
	xorl	%r12d, %r12d
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbp, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	subq	$0x98, %rsp
	.cfi_def_cfa_offset 208
	leaq	0x60(%rsp), %rbx
	leaq	0x8c(%rsp), %rbp
	callq	puts@PLT
	movsd	DATAat0x401640(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x401648(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x401650(%rip), %xmm1
	movq	%rbx, %r13
	movsd	DATAat0x401658(%rip), %xmm0
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
LBB00_br: 	jmp	Ltmp0
# FDATA: 1 main #LBB00_br# 1 main #Ltmp0# 0 0

Ltmp1:
	movsd	(%r13), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	addl	$0x1, %r12d
	addq	$0x8, %r13
Ltmp1_br: 	callq	printf@PLT
# FDATA: 1 main #Ltmp1_br# 1 main #Ltmp0# 0 0

Ltmp0:
	cmpl	0x8c(%rsp), %r12d
Ltmp0_br: 	jl	Ltmp1
# FDATA: 1 main #Ltmp0_br# 1 main #Ltmp1# 0 0
# FDATA: 1 main #Ltmp0_br# 1 main #LFT2# 0 0

LFT2:
	movl	$0xa, %edi
	movq	%rbx, %r13
	xorl	%r12d, %r12d
	callq	putchar@PLT
	movsd	DATAat0x401640(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x401660(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x401668(%rip), %xmm1
	movsd	DATAat0x401658(%rip), %xmm0
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
LFT2_br: 	jmp	Ltmp2
# FDATA: 1 main #LFT2_br# 1 main #Ltmp2# 0 0

Ltmp3:
	movsd	(%r13), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	addl	$0x1, %r12d
	addq	$0x8, %r13
Ltmp3_br: 	callq	printf@PLT
# FDATA: 1 main #Ltmp3_br# 1 main #Ltmp2# 0 0

Ltmp2:
	cmpl	0x8c(%rsp), %r12d
Ltmp2_br: 	jl	Ltmp3
# FDATA: 1 main #Ltmp2_br# 1 main #Ltmp3# 0 0
# FDATA: 1 main #Ltmp2_br# 1 main #LFT4# 0 0

LFT4:
	movl	$0xa, %edi
	callq	putchar@PLT
	movsd	DATAat0x401670(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x401678(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x401680(%rip), %xmm1
	movsd	DATAat0x401658(%rip), %xmm0
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
	movq	%rbx, %r12
	xorl	%r13d, %r13d
LFT4_br: 	jmp	Ltmp4
# FDATA: 1 main #LFT4_br# 1 main #Ltmp4# 0 0

Ltmp5:
	movsd	(%r12), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	callq	printf@PLT
	addl	$0x1, %r13d
Ltmp5_br: 	addq	$0x8, %r12
# FDATA: 1 main #Ltmp5_br# 1 main #Ltmp4# 0 0

Ltmp4:
	cmpl	0x8c(%rsp), %r13d
Ltmp4_br: 	jl	Ltmp5
# FDATA: 1 main #Ltmp4_br# 1 main #Ltmp5# 0 0
# FDATA: 1 main #Ltmp4_br# 1 main #LFT6# 0 0

LFT6:
	movl	$0xa, %edi
LFT6_br: 	callq	putchar@PLT
# FDATA: 1 main #LFT6_br# 1 main #Ltmp27# 0 0

Ltmp27:
	movsd	DATAat0x401658(%rip), %xmm2
	movq	%rbx, %rsi
	movsd	DATAat0x401688(%rip), %xmm3
	movq	%rbp, %rdi
	movsd	DATAat0x401690(%rip), %xmm1
	movq	%rbx, %r13
	movapd	%xmm2, %xmm0
	xorl	%r12d, %r12d
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
Ltmp27_br: 	jmp	Ltmp6
# FDATA: 1 main #Ltmp27_br# 1 main #Ltmp6# 0 0

Ltmp7:
	movsd	(%r13), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	addl	$0x1, %r12d
	addq	$0x8, %r13
Ltmp7_br: 	callq	printf@PLT
# FDATA: 1 main #Ltmp7_br# 1 main #Ltmp6# 0 0

Ltmp6:
	cmpl	0x8c(%rsp), %r12d
Ltmp6_br: 	jl	Ltmp7
# FDATA: 1 main #Ltmp6_br# 1 main #Ltmp7# 0 0
# FDATA: 1 main #Ltmp6_br# 1 main #LFT9# 0 0

LFT9:
	movl	$0xa, %edi
	movq	%rbx, %r13
	xorl	%r12d, %r12d
	callq	putchar@PLT
	movsd	DATAat0x401698(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x4016a0(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x4016a8(%rip), %xmm1
	movsd	DATAat0x4016b0(%rip), %xmm0
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
LFT9_br: 	jmp	Ltmp8
# FDATA: 1 main #LFT9_br# 1 main #Ltmp8# 0 0

Ltmp9:
	movsd	(%r13), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	addl	$0x1, %r12d
	addq	$0x8, %r13
Ltmp9_br: 	callq	printf@PLT
# FDATA: 1 main #Ltmp9_br# 1 main #Ltmp8# 0 0

Ltmp8:
	cmpl	0x8c(%rsp), %r12d
Ltmp8_br: 	jl	Ltmp9
# FDATA: 1 main #Ltmp8_br# 1 main #Ltmp9# 0 0
# FDATA: 1 main #Ltmp8_br# 1 main #LFT11# 0 0

LFT11:
	movl	$0xa, %edi
	movq	%rbx, %r13
	xorl	%r12d, %r12d
	callq	putchar@PLT
	movsd	DATAat0x4016b8(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x4016c0(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x4016c8(%rip), %xmm1
	movsd	DATAat0x4016d0(%rip), %xmm0
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
LFT11_br: 	jmp	Ltmp10
# FDATA: 1 main #LFT11_br# 1 main #Ltmp10# 0 0

Ltmp11:
	movsd	(%r13), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	addl	$0x1, %r12d
	addq	$0x8, %r13
Ltmp11_br: 	callq	printf@PLT
# FDATA: 1 main #Ltmp11_br# 1 main #Ltmp10# 0 0

Ltmp10:
	cmpl	0x8c(%rsp), %r12d
Ltmp10_br: 	jl	Ltmp11
# FDATA: 1 main #Ltmp10_br# 1 main #Ltmp11# 0 0
# FDATA: 1 main #Ltmp10_br# 1 main #LFT13# 0 0

LFT13:
	movl	$0xa, %edi
	callq	putchar@PLT
	movsd	DATAat0x4016d8(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x4016e0(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x4016e8(%rip), %xmm1
	movsd	DATAat0x4016f0(%rip), %xmm0
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
	movq	%rbx, %r12
	xorl	%r13d, %r13d
LFT13_br: 	jmp	Ltmp12
# FDATA: 1 main #LFT13_br# 1 main #Ltmp12# 0 0

Ltmp13:
	movsd	(%r12), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	callq	printf@PLT
	addl	$0x1, %r13d
Ltmp13_br: 	addq	$0x8, %r12
# FDATA: 1 main #Ltmp13_br# 1 main #Ltmp12# 0 0

Ltmp12:
	cmpl	0x8c(%rsp), %r13d
Ltmp12_br: 	jl	Ltmp13
# FDATA: 1 main #Ltmp12_br# 1 main #Ltmp13# 0 0
# FDATA: 1 main #Ltmp12_br# 1 main #LFT15# 0 0

LFT15:
	movl	$0xa, %edi
LFT15_br: 	callq	putchar@PLT
# FDATA: 1 main #LFT15_br# 1 main #Ltmp29# 0 0

Ltmp29:
	movsd	DATAat0x4016f8(%rip), %xmm3
	movq	%rbx, %rsi
	movsd	DATAat0x401700(%rip), %xmm2
	movq	%rbp, %rdi
	movsd	DATAat0x401708(%rip), %xmm1
	movq	%rbx, %r13
	movsd	DATAat0x401710(%rip), %xmm0
	xorl	%r12d, %r12d
	callq	_Z10SolveCubicddddPiPd
	movl	$0x4015d8, %edi
	xorl	%eax, %eax
	callq	printf@PLT
Ltmp29_br: 	jmp	Ltmp14
# FDATA: 1 main #Ltmp29_br# 1 main #Ltmp14# 0 0

Ltmp15:
	movsd	(%r13), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	addl	$0x1, %r12d
	addq	$0x8, %r13
Ltmp15_br: 	callq	printf@PLT
# FDATA: 1 main #Ltmp15_br# 1 main #Ltmp14# 0 0

Ltmp14:
	cmpl	0x8c(%rsp), %r12d
Ltmp14_br: 	jl	Ltmp15
# FDATA: 1 main #Ltmp14_br# 1 main #Ltmp15# 0 0
# FDATA: 1 main #Ltmp14_br# 1 main #LFT16# 0 0

LFT16:
	movl	$0xa, %edi
	movabsq	$-0x4010000000000000, %r14
	callq	putchar@PLT
	movabsq	$0x3ff0000000000000, %rsi
	movl	$0x0, 0x5c(%rsp)
LFT16_br: 	movq	%rsi, 0x50(%rsp)
# FDATA: 1 main #LFT16_br# 1 main #Ltmp21# 0 0

Ltmp21:
	movabsq	$0x4024000000000000, %rax
	xorl	%r15d, %r15d
Ltmp21_br: 	movq	%rax, 0x48(%rsp)
# FDATA: 1 main #Ltmp21_br# 1 main #Ltmp20# 0 0

Ltmp20:
	movabsq	$0x4014000000000000, %rdx
	xorl	%r13d, %r13d
Ltmp20_br: 	movq	%rdx, 0x40(%rsp)
# FDATA: 1 main #Ltmp20_br# 1 main #Ltmp19# 0 0

Ltmp19:
	xorl	%r12d, %r12d
Ltmp19_br: 	movq	%r14, 0x38(%rsp)
# FDATA: 1 main #Ltmp19_br# 1 main #Ltmp18# 0 0

Ltmp18:
	movsd	0x38(%rsp), %xmm3
	movq	%rbx, %rsi
	movsd	0x40(%rsp), %xmm2
	movq	%rbp, %rdi
	movsd	0x48(%rsp), %xmm1
	movsd	0x50(%rsp), %xmm0
	callq	_Z10SolveCubicddddPiPd
	xorl	%eax, %eax
	movl	$0x4015d8, %edi
	callq	printf@PLT
	movl	0x8c(%rsp), %ecx
	testl	%ecx, %ecx
Ltmp18_br: 	jle	Ltmp16
# FDATA: 1 main #Ltmp18_br# 1 main #Ltmp16# 0 0
# FDATA: 1 main #Ltmp18_br# 1 main #LFT17# 0 0

LFT17:
	movq	%rbx, %rcx
LFT17_br: 	xorl	%edx, %edx
# FDATA: 1 main #LFT17_br# 1 main #Ltmp17# 0 0

Ltmp17:
	movsd	(%rcx), %xmm0
	movl	$0x4015e3, %edi
	movl	$0x1, %eax
	movl	%edx, 0x20(%rsp)
	movq	%rcx, 0x30(%rsp)
	callq	printf@PLT
	movl	0x20(%rsp), %edx
	movq	0x30(%rsp), %rcx
	addl	$0x1, %edx
	addq	$0x8, %rcx
	cmpl	%edx, 0x8c(%rsp)
Ltmp17_br: 	jg	Ltmp17
# FDATA: 1 main #Ltmp17_br# 1 main #Ltmp17# 0 0
# FDATA: 1 main #Ltmp17_br# 1 main #Ltmp16# 0 0

Ltmp16:
	movl	$0xa, %edi
	addl	$0x1, %r12d
	callq	putchar@PLT
	movsd	0x38(%rsp), %xmm0
	cmpl	$0x9, %r12d
	subsd	DATAat0x401718(%rip), %xmm0
	movsd	%xmm0, 0x38(%rsp)
Ltmp16_br: 	jne	Ltmp18
# FDATA: 1 main #Ltmp16_br# 1 main #Ltmp18# 0 0
# FDATA: 1 main #Ltmp16_br# 1 main #LFT18# 0 0

LFT18:
	movsd	DATAat0x401720(%rip), %xmm0
	addl	$0x1, %r13d
	cmpl	$0x11, %r13d
	addsd	0x40(%rsp), %xmm0
	movsd	%xmm0, 0x40(%rsp)
LFT18_br: 	jne	Ltmp19
# FDATA: 1 main #LFT18_br# 1 main #Ltmp19# 0 0
# FDATA: 1 main #LFT18_br# 1 main #LFT19# 0 0

LFT19:
	movsd	0x48(%rsp), %xmm0
	addl	$0x1, %r15d
	cmpl	$0x28, %r15d
	subsd	DATAat0x401728(%rip), %xmm0
	movsd	%xmm0, 0x48(%rsp)
LFT19_br: 	jne	Ltmp20
# FDATA: 1 main #LFT19_br# 1 main #Ltmp20# 0 0
# FDATA: 1 main #LFT19_br# 1 main #LFT20# 0 0

LFT20:
	movsd	DATAat0x401658(%rip), %xmm0
	addl	$0x1, 0x5c(%rsp)
	cmpl	$0x9, 0x5c(%rsp)
	addsd	0x50(%rsp), %xmm0
	movsd	%xmm0, 0x50(%rsp)
LFT20_br: 	jne	Ltmp21
# FDATA: 1 main #LFT20_br# 1 main #Ltmp21# 0 0
# FDATA: 1 main #LFT20_br# 1 main #LFT21# 0 0

LFT21:
	leaq	0x80(%rsp), %r12
	movl	$0x401548, %edi
	movl	$0xc350, %ebp
	xorl	%ebx, %ebx
LFT21_br: 	callq	puts@PLT
# FDATA: 1 main #LFT21_br# 1 main #Ltmp22# 0 0

Ltmp22:
	movslq	%ebx, %rdi
	movq	%r12, %rsi
	callq	_Z5usqrtmP8int_sqrt
	movl	0x80(%rsp), %edx
	movl	%ebx, %esi
	xorl	%eax, %eax
	movl	$0x4015f2, %edi
	addl	$0x2, %ebx
	callq	printf@PLT
	subl	$0x1, %ebp
Ltmp22_br: 	jne	Ltmp22
# FDATA: 1 main #Ltmp22_br# 1 main #Ltmp22# 0 0
# FDATA: 1 main #Ltmp22_br# 1 main #LFT22# 0 0

LFT22:
	movl	$0xa, %edi
	movl	$0x3fed0169, %ebx
LFT22_br: 	callq	putchar@PLT
# FDATA: 1 main #LFT22_br# 1 main #Ltmp23# 0 0

Ltmp23:
	movq	%rbx, %rdi
	movq	%r12, %rsi
	callq	_Z5usqrtmP8int_sqrt
	movl	0x80(%rsp), %edx
	movq	%rbx, %rsi
	xorl	%eax, %eax
	movl	$0x401603, %edi
	addq	$0x1, %rbx
	callq	printf@PLT
	cmpq	$0x3fed4169, %rbx
Ltmp23_br: 	jne	Ltmp23
# FDATA: 1 main #Ltmp23_br# 1 main #Ltmp23# 0 0
# FDATA: 1 main #Ltmp23_br# 1 main #LFT23# 0 0

LFT23:
	movl	$0x401570, %edi
	xorl	%ebx, %ebx
	callq	puts@PLT
	movq	%rbx, 0x8(%rsp)
	movsd	DATAat0x401748(%rip), %xmm3
LFT23_br: 	movsd	0x8(%rsp), %xmm2
# FDATA: 1 main #LFT23_br# 1 main #Ltmp24# 0 0

Ltmp24:
	movsd	DATAat0x401730(%rip), %xmm1
	movl	$0x401598, %edi
	movapd	%xmm2, %xmm0
	movl	$0x2, %eax
	mulsd	%xmm2, %xmm1
	movsd	%xmm2, 0x20(%rsp)
	movsd	%xmm3, 0x10(%rsp)
	divsd	DATAat0x401738(%rip), %xmm1
	callq	printf@PLT
	movsd	0x20(%rsp), %xmm2
	movsd	0x10(%rsp), %xmm3
	addsd	DATAat0x401740(%rip), %xmm2
	ucomisd	%xmm2, %xmm3
Ltmp24_br: 	jae	Ltmp24
# FDATA: 1 main #Ltmp24_br# 1 main #Ltmp24# 0 0
# FDATA: 1 main #Ltmp24_br# 1 main #LFT24# 0 0

LFT24:
	movl	$0x401612, %edi
	callq	puts@PLT
	movq	%rbx, 0x8(%rsp)
	movsd	DATAat0x401758(%rip), %xmm3
LFT24_br: 	movsd	0x8(%rsp), %xmm2
# FDATA: 1 main #LFT24_br# 1 main #Ltmp25# 0 0

Ltmp25:
	movsd	DATAat0x401738(%rip), %xmm1
	movl	$0x4015b8, %edi
	movapd	%xmm2, %xmm0
	movl	$0x2, %eax
	mulsd	%xmm2, %xmm1
	movsd	%xmm2, 0x20(%rsp)
	movsd	%xmm3, 0x10(%rsp)
	divsd	DATAat0x401730(%rip), %xmm1
	callq	printf@PLT
	movsd	0x20(%rsp), %xmm2
	movsd	0x10(%rsp), %xmm3
	addsd	DATAat0x401750(%rip), %xmm2
	ucomisd	%xmm2, %xmm3
Ltmp25_br: 	jae	Ltmp25
# FDATA: 1 main #Ltmp25_br# 1 main #Ltmp25# 0 0
# FDATA: 1 main #Ltmp25_br# 1 main #LFT25# 0 0

LFT25:
	addq	$0x98, %rsp
	.cfi_def_cfa_offset 56
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
	.cfi_def_cfa %rsp, 208

LLP0:
	cmpq	$0x1, %rdx
	movq	%rax, %rdi
LLP0_br: 	je	Ltmp26
# FDATA: 1 main #LLP0_br# 1 main #Ltmp26# 0 0
# FDATA: 1 main #LLP0_br# 1 main #Ltmp28# 0 0

Ltmp28:
Ltmp28_br: 	callq	_Unwind_Resume@PLT
# FDATA: 1 main #Ltmp28_br# 1 main #Ltmp26# 0 0

Ltmp26:
	callq	__cxa_begin_catch@PLT
	movl	$0x4015e7, %edi
	callq	puts@PLT
	callq	__cxa_end_catch@PLT
Ltmp26_br: 	jmp	Ltmp27
# FDATA: 1 main #Ltmp26_br# 1 main #Ltmp27# 0 0

Ltmp30:
	movq	%rax, 0x30(%rsp)
	callq	__cxa_end_catch@PLT
	movq	0x30(%rsp), %rdi
Ltmp30_br: 	jmp	Ltmp28
# FDATA: 1 main #Ltmp30_br# 1 main #Ltmp28# 0 0

LLP1:
	cmpq	$0x1, %rdx
	movq	%rax, %rdi
LLP1_br: 	jne	Ltmp28
# FDATA: 1 main #LLP1_br# 1 main #Ltmp28# 0 0
# FDATA: 1 main #LLP1_br# 1 main #LFT26# 0 0

LFT26:
	callq	__cxa_begin_catch@PLT
	movl	$0x4015e7, %edi
	callq	puts@PLT
	callq	__cxa_end_catch@PLT
LFT26_br: 	jmp	Ltmp29
# FDATA: 1 main #LFT26_br# 1 main #Ltmp29# 0 0

LLP2:
LLP2_br: 	jmp	Ltmp30
# FDATA: 1 main #LLP2_br# 1 main #Ltmp30# 0 0

	.cfi_endproc
.size main, .-main

.section .rodata
"DATAat0x401738":
"DATAat0x401748":
"DATAat0x401728":
"DATAat0x401718":
"DATAat0x4016f8":
"DATAat0x401690":
"DATAat0x401668":
"DATAat0x401650":
"DATAat0x401700":
"DATAat0x401698":
"DATAat0x401688":
"DATAat0x401648":
"DATAat0x401680":
"DATAat0x4016c8":
"DATAat0x401750":
"DATAat0x401678":
"DATAat0x4016c0":
"DATAat0x401758":
"DATAat0x401660":
"DATAat0x4016b0":
"DATAat0x401640":
"DATAat0x401658":
"DATAat0x401720":
"DATAat0x4016f0":
"DATAat0x401710":
"DATAat0x4016a8":
"DATAat0x401730":
"DATAat0x4016b8":
"DATAat0x401708":
"DATAat0x401670":
"DATAat0x4016a0":
"DATAat0x4016d0":
"DATAat0x4016d8":
"DATAat0x4016e0":
"DATAat0x401740":
"DATAat0x4016e8":
	.text
  .globl _Z10SolveCubicddddPiPd
  .type _Z10SolveCubicddddPiPd, %function
_Z10SolveCubicddddPiPd:
# FDATA: 0 [unknown] 0 1 _Z10SolveCubicddddPiPd 0 0 57
	.cfi_startproc
LBB01:
	divsd	%xmm0, %xmm1
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movq	%rsi, %rbx
	subq	$0x70, %rsp
	.cfi_def_cfa_offset 128
	divsd	%xmm0, %xmm2
	movsd	%xmm1, 0x48(%rsp)
	fldl	0x48(%rsp)
	divsd	%xmm0, %xmm3
	movsd	%xmm2, 0x48(%rsp)
	fldl	0x48(%rsp)
	fld	%st(1)
	fmul	%st(2), %st
	movsd	%xmm3, 0x48(%rsp)
	fld	%st(1)
	fmuls	DATAat0x401760(%rip)
	faddp	%st, %st(1)
	fdivs	DATAat0x401764(%rip)
	fld	%st(2)
	fadd	%st(3), %st
	fmul	%st(3), %st
	fmul	%st(3), %st
	fld	%st(3)
	fmuls	DATAat0x401768(%rip)
	fmulp	%st, %st(3)
	faddp	%st, %st(2)
	flds	DATAat0x40176c(%rip)
	fmull	0x48(%rsp)
	faddp	%st, %st(2)
	fxch	%st(1)
	fdivs	DATAat0x401770(%rip)
	fld	%st(1)
	fmul	%st(2), %st
	fmul	%st(2), %st
	fld	%st(1)
	fmul	%st(2), %st
	fsub	%st(1), %st
	fstpl	0x68(%rsp)
	movsd	0x68(%rsp), %xmm0
	ucomisd	DATAat0x401778(%rip), %xmm0
LBB01_br: 	jbe	Ltmp31
# FDATA: 1 _Z10SolveCubicddddPiPd #LBB01_br# 1 _Z10SolveCubicddddPiPd #Ltmp31# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LBB01_br# 1 _Z10SolveCubicddddPiPd #LFT1# 0 0

LFT1:
	fstp	%st(0)
LFT1_br: 	jmp	Ltmp32
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT1_br# 1 _Z10SolveCubicddddPiPd #Ltmp32# 0 0

Ltmp37:
Ltmp37_br: 	fstp	%st(0)
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp37_br# 1 _Z10SolveCubicddddPiPd #Ltmp32# 0 0

Ltmp32:
	sqrtsd	%xmm0, %xmm1
	movl	$0x1, (%rdi)
	ucomisd	%xmm1, %xmm1
Ltmp32_br: 	jp	Ltmp33
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp32_br# 1 _Z10SolveCubicddddPiPd #Ltmp33# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp32_br# 1 _Z10SolveCubicddddPiPd #LFT3# 0 0

LFT3:
LFT3_br: 	jne	Ltmp33
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT3_br# 1 _Z10SolveCubicddddPiPd #Ltmp33# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT3_br# 1 _Z10SolveCubicddddPiPd #Ltmp36# 0 0

Ltmp36:
	fstl	0x68(%rsp)
	movsd	DATAat0x4017b0(%rip), %xmm2
	fstpt	0x30(%rsp)
	fxch	%st(1)
	movsd	0x68(%rsp), %xmm0
	fstpt	0x10(%rsp)
	andpd	%xmm2, %xmm0
	fstpt	0x20(%rsp)
	addsd	%xmm1, %xmm0
	movsd	DATAat0x401798(%rip), %xmm1
	callq	pow@PLT
	movsd	%xmm0, 0x8(%rsp)
	fldl	0x8(%rsp)
	fldz
	fldt	0x30(%rsp)
	fxch	%st(1)
	fucompi	%st(1), %st
	fstp	%st(0)
	fldt	0x10(%rsp)
	fldt	0x20(%rsp)
Ltmp36_br: 	ja	Ltmp34
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp36_br# 1 _Z10SolveCubicddddPiPd #Ltmp34# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp36_br# 1 _Z10SolveCubicddddPiPd #LFT5# 0 0

LFT5:
LFT5_br: 	movsd	DATAat0x4017a0(%rip), %xmm1
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT5_br# 1 _Z10SolveCubicddddPiPd #Ltmp35# 0 0

Ltmp35:
	fdiv	%st(2), %st
	faddp	%st, %st(2)
	fxch	%st(1)
	fstpl	0x68(%rsp)
	fdivs	DATAat0x401760(%rip)
	movsd	0x68(%rsp), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	%xmm0, 0x8(%rsp)
	fldl	0x8(%rsp)
	faddp	%st, %st(1)
	fstpl	(%rbx)
	addq	$0x70, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
	.cfi_def_cfa %rsp, 128

Ltmp34:
	movsd	DATAat0x401658(%rip), %xmm1
Ltmp34_br: 	jmp	Ltmp35
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp34_br# 1 _Z10SolveCubicddddPiPd #Ltmp35# 0 0

Ltmp33:
	fstpt	0x30(%rsp)
	fxch	%st(1)
	fstpt	0x10(%rsp)
	fstpt	0x20(%rsp)
	callq	sqrt@PLT
	movapd	%xmm0, %xmm1
	fldt	0x20(%rsp)
	fldt	0x10(%rsp)
	fldt	0x30(%rsp)
	fxch	%st(1)
	fxch	%st(2)
	fxch	%st(1)
Ltmp33_br: 	jmp	Ltmp36
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp33_br# 1 _Z10SolveCubicddddPiPd #Ltmp36# 0 0

Ltmp31:
Ltmp31_br: 	jp	Ltmp37
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp31_br# 1 _Z10SolveCubicddddPiPd #Ltmp37# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp31_br# 1 _Z10SolveCubicddddPiPd #LFT7# 0 0

LFT7:
	fstpl	0x68(%rsp)
	movl	$0x3, (%rdi)
	movsd	0x68(%rsp), %xmm1
	sqrtsd	%xmm1, %xmm0
	ucomisd	%xmm0, %xmm0
LFT7_br: 	jp	Ltmp38
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT7_br# 1 _Z10SolveCubicddddPiPd #Ltmp38# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT7_br# 1 _Z10SolveCubicddddPiPd #LFT8# 0 0

LFT8:
LFT8_br: 	jne	Ltmp38
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT8_br# 1 _Z10SolveCubicddddPiPd #Ltmp38# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT8_br# 1 _Z10SolveCubicddddPiPd #Ltmp47# 0 0

Ltmp47:
	movsd	%xmm0, 0x8(%rsp)
	fldl	0x8(%rsp)
	fdivrp	%st, %st(1)
	fstpl	0x68(%rsp)
	fxch	%st(1)
	fstpt	0x10(%rsp)
	movsd	0x68(%rsp), %xmm0
	fstpt	0x20(%rsp)
	callq	acos@PLT
	movsd	%xmm0, 0x48(%rsp)
	fldt	0x20(%rsp)
	fstpl	0x60(%rsp)
	sqrtsd	0x60(%rsp), %xmm1
	movapd	%xmm1, %xmm2
	ucomisd	%xmm1, %xmm1
	fldt	0x10(%rsp)
Ltmp47_br: 	jp	Ltmp39
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp47_br# 1 _Z10SolveCubicddddPiPd #Ltmp39# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp47_br# 1 _Z10SolveCubicddddPiPd #LFT10# 0 0

LFT10:
LFT10_br: 	jne	Ltmp40
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT10_br# 1 _Z10SolveCubicddddPiPd #Ltmp40# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT10_br# 1 _Z10SolveCubicddddPiPd #Ltmp46# 0 0

Ltmp46:
	divsd	DATAat0x4016b0(%rip), %xmm0
	movsd	%xmm1, 0x20(%rsp)
	movsd	%xmm2, 0x30(%rsp)
	fstpt	0x10(%rsp)
	callq	cos@PLT
	movsd	0x30(%rsp), %xmm2
	movsd	0x20(%rsp), %xmm1
	fldt	0x10(%rsp)
	ucomisd	%xmm1, %xmm1
	mulsd	DATAat0x401780(%rip), %xmm2
	fdivs	DATAat0x401760(%rip)
	mulsd	%xmm0, %xmm2
	movsd	%xmm2, 0x8(%rsp)
	movapd	%xmm1, %xmm2
	fld	%st(0)
	fstpt	0x50(%rsp)
	fldl	0x8(%rsp)
	faddp	%st, %st(1)
	fstpl	(%rbx)
Ltmp46_br: 	jp	Ltmp41
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp46_br# 1 _Z10SolveCubicddddPiPd #Ltmp41# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp46_br# 1 _Z10SolveCubicddddPiPd #LFT12# 0 0

LFT12:
LFT12_br: 	jne	Ltmp41
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT12_br# 1 _Z10SolveCubicddddPiPd #Ltmp41# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT12_br# 1 _Z10SolveCubicddddPiPd #Ltmp44# 0 0

Ltmp44:
	movsd	0x48(%rsp), %xmm0
	movsd	%xmm1, 0x20(%rsp)
	addsd	DATAat0x401788(%rip), %xmm0
	movsd	%xmm2, 0x30(%rsp)
	divsd	DATAat0x4016b0(%rip), %xmm0
	callq	cos@PLT
	movsd	0x30(%rsp), %xmm2
	fldt	0x50(%rsp)
	mulsd	DATAat0x401780(%rip), %xmm2
	movsd	0x20(%rsp), %xmm1
	ucomisd	%xmm1, %xmm1
	mulsd	%xmm0, %xmm2
	movsd	%xmm2, 0x8(%rsp)
	fldl	0x8(%rsp)
	faddp	%st, %st(1)
	fstpl	0x8(%rbx)
Ltmp44_br: 	jp	Ltmp42
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp44_br# 1 _Z10SolveCubicddddPiPd #Ltmp42# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp44_br# 1 _Z10SolveCubicddddPiPd #LFT14# 0 0

LFT14:
LFT14_br: 	jne	Ltmp42
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT14_br# 1 _Z10SolveCubicddddPiPd #Ltmp42# 0 0
# FDATA: 1 _Z10SolveCubicddddPiPd #LFT14_br# 1 _Z10SolveCubicddddPiPd #Ltmp43# 0 0

Ltmp43:
	movsd	0x48(%rsp), %xmm0
	movsd	%xmm1, 0x20(%rsp)
	addsd	DATAat0x401790(%rip), %xmm0
	divsd	DATAat0x4016b0(%rip), %xmm0
	callq	cos@PLT
	fldt	0x50(%rsp)
	movsd	DATAat0x401780(%rip), %xmm2
	movsd	0x20(%rsp), %xmm1
	mulsd	%xmm1, %xmm2
	mulsd	%xmm0, %xmm2
	movsd	%xmm2, 0x8(%rsp)
	fldl	0x8(%rsp)
	faddp	%st, %st(1)
	fstpl	0x10(%rbx)
	addq	$0x70, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
	.cfi_def_cfa %rsp, 128

Ltmp42:
	movsd	0x60(%rsp), %xmm0
	callq	sqrt@PLT
	movapd	%xmm0, %xmm1
Ltmp42_br: 	jmp	Ltmp43
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp42_br# 1 _Z10SolveCubicddddPiPd #Ltmp43# 0 0

Ltmp41:
	movsd	0x60(%rsp), %xmm0
	callq	sqrt@PLT
	movsd	0x20(%rsp), %xmm1
	movapd	%xmm0, %xmm2
Ltmp41_br: 	jmp	Ltmp44
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp41_br# 1 _Z10SolveCubicddddPiPd #Ltmp44# 0 0

Ltmp39:
	fstp	%st(0)
Ltmp39_br: 	jmp	Ltmp45
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp39_br# 1 _Z10SolveCubicddddPiPd #Ltmp45# 0 0

Ltmp40:
Ltmp40_br: 	fstp	%st(0)
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp40_br# 1 _Z10SolveCubicddddPiPd #Ltmp45# 0 0

Ltmp45:
	movsd	0x60(%rsp), %xmm0
	movsd	%xmm1, 0x20(%rsp)
	callq	sqrt@PLT
	movsd	0x20(%rsp), %xmm1
	movapd	%xmm0, %xmm2
	movsd	0x48(%rsp), %xmm0
	fldt	0x10(%rsp)
Ltmp45_br: 	jmp	Ltmp46
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp45_br# 1 _Z10SolveCubicddddPiPd #Ltmp46# 0 0

Ltmp38:
	fstpt	0x30(%rsp)
	fxch	%st(1)
	movapd	%xmm1, %xmm0
	fstpt	0x10(%rsp)
	fstpt	0x20(%rsp)
	callq	sqrt@PLT
	fldt	0x20(%rsp)
	fldt	0x10(%rsp)
	fldt	0x30(%rsp)
	fxch	%st(1)
	fxch	%st(2)
	fxch	%st(1)
Ltmp38_br: 	jmp	Ltmp47
# FDATA: 1 _Z10SolveCubicddddPiPd #Ltmp38_br# 1 _Z10SolveCubicddddPiPd #Ltmp47# 0 0

	.cfi_endproc
.size _Z10SolveCubicddddPiPd, .-_Z10SolveCubicddddPiPd
.section .rodata
"DATAat0x401788":
"DATAat0x401790":
"DATAat0x401780":
"DATAat0x401770":
"DATAat0x40176c":
"DATAat0x401760":
"DATAat0x401768":
"DATAat0x401778":
"DATAat0x401764":
"DATAat0x401798":
"DATAat0x4017b0":
"DATAat0x4017a0":

	.text
  .globl _Z5usqrtmP8int_sqrt
  .type _Z5usqrtmP8int_sqrt, %function
_Z5usqrtmP8int_sqrt:
# FDATA: 0 [unknown] 0 1 _Z5usqrtmP8int_sqrt 0 0 6
	.cfi_startproc
LBB02:
	xorl	%r9d, %r9d
	xorl	%eax, %eax
LBB02_br: 	xorl	%ecx, %ecx
# FDATA: 1 _Z5usqrtmP8int_sqrt #LBB02_br# 1 _Z5usqrtmP8int_sqrt #Ltmp48# 0 0

Ltmp48:
	movq	%rdi, %rdx
	leaq	(%rax,%rax), %r10
	leaq	0x1(,%rax,4), %r8
	andl	$0xc0000000, %edx
	shlq	$0x2, %rdi
	shrq	$0x1e, %rdx
	leaq	0x1(%r10), %rax
	leaq	(%rdx,%rcx,4), %rdx
	movq	%rdx, %rcx
	subq	%r8, %rcx
	cmpq	%r8, %rdx
	cmovbq	%rdx, %rcx
	cmovbq	%r10, %rax
	addl	$0x1, %r9d
	cmpl	$0x20, %r9d
Ltmp48_br: 	jne	Ltmp48
# FDATA: 1 _Z5usqrtmP8int_sqrt #Ltmp48_br# 1 _Z5usqrtmP8int_sqrt #Ltmp48# 0 0
# FDATA: 1 _Z5usqrtmP8int_sqrt #Ltmp48_br# 1 _Z5usqrtmP8int_sqrt #LFT0# 0 0

LFT0:
	movq	%rax, (%rsi)
	retq

	.cfi_endproc
.size _Z5usqrtmP8int_sqrt, .-_Z5usqrtmP8int_sqrt
