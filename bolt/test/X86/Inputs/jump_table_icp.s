	.text
  .globl main
  .type main, %function
main:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%rbx
	subq	$0x18, %rsp
	.cfi_offset %rbx, -24
	movl	$0x0, -0x14(%rbp)
	movl	$0x0, -0x18(%rbp)
	jmp	Ltmp16

Ltmp17:
	callq	rand@PLT
	movl	%eax, %ecx
	movl	$0x92492493, %edx
	movl	%ecx, %eax
	imull	%edx
	leal	(%rdx,%rcx), %eax
	sarl	$0x2, %eax
	movl	%eax, %edx
	movl	%ecx, %eax
	sarl	$0x1f, %eax
	subl	%eax, %edx
	movl	%edx, %eax
	movl	%eax, -0x1c(%rbp)
	movl	-0x1c(%rbp), %edx
	movl	%edx, %eax
	shll	$0x3, %eax
	subl	%edx, %eax
	subl	%eax, %ecx
	movl	%ecx, %eax
	movl	%eax, -0x1c(%rbp)
	callq	rand@PLT
	movl	%eax, %ecx
	movl	$0x92492493, %edx
	movl	%ecx, %eax
	imull	%edx
	leal	(%rdx,%rcx), %eax
	sarl	$0x2, %eax
	movl	%eax, %edx
	movl	%ecx, %eax
	sarl	$0x1f, %eax
	subl	%eax, %edx
	movl	%edx, %eax
	movl	%eax, -0x20(%rbp)
	movl	-0x20(%rbp), %edx
	movl	%edx, %eax
	shll	$0x3, %eax
	subl	%edx, %eax
	subl	%eax, %ecx
	movl	%ecx, %eax
	movl	%eax, -0x20(%rbp)
	movl	-0x1c(%rbp), %eax
	movl	%eax, %edi
Ltmp17_inc:
	callq	_Z3inci
# FDATA: 1 main #Ltmp17_inc# 1 _Z3inci 0 0 1073
	movl	%eax, %ebx
	movl	-0x20(%rbp), %eax
	movl	%eax, %edi
Ltmp17_dup:
	callq	_Z7inc_dupi
# FDATA: 1 main #Ltmp17_dup# 1 _Z7inc_dupi 0 0 1064
	movl	%eax, %edx
	movl	$0x0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	%ebx, %eax
	addl	%eax, -0x14(%rbp)
	addl	$0x1, -0x18(%rbp)

Ltmp16:
	cmpl	$0x98967f, -0x18(%rbp)
Ltmp16_br:
	jle	Ltmp17
# FDATA: 1 main #Ltmp16_br# 1 main #Ltmp17# 0 651

	cmpl	$0x0, -0x14(%rbp)
	sete	%al
	movzbl	%al, %eax
	addq	$0x18, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size main, .-main

  .globl _Z3inci
  .type _Z3inci, %function
_Z3inci:
	.cfi_startproc
LBB00:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -0x4(%rbp)
	cmpl	$0x5, -0x4(%rbp)
LBB00_br:
	ja	Ltmp12
# FDATA: 1 _Z3inci #LBB00_br# 1 _Z3inci #Ltmp12# 189 189
# FDATA: 1 _Z3inci #LBB00_br# 1 _Z3inci #LFT0# 0 881

LFT0:
	movl	-0x4(%rbp), %eax
	movq	"JUMP_TABLE/_Z3inci.0"(,%rax,8), %rax
LFT0_br:
	jmpq	*%rax
# FDATA: 1 _Z3inci #LFT0_br# 1 _Z3inci #Ltmp0# 146 163
# FDATA: 1 _Z3inci #LFT0_br# 1 _Z3inci #Ltmp1# 140 156
# FDATA: 1 _Z3inci #LFT0_br# 1 _Z3inci #Ltmp2# 126 157
# FDATA: 1 _Z3inci #LFT0_br# 1 _Z3inci #Ltmp3# 129 148
# FDATA: 1 _Z3inci #LFT0_br# 1 _Z3inci #Ltmp4# 137 150
# FDATA: 1 _Z3inci #LFT0_br# 1 _Z3inci #Ltmp5# 134 152

Ltmp0:
	movl	total(%rip), %eax
	addl	$0x1, %eax
	movl	%eax, total(%rip)
	movl	$0x1, %eax
Ltmp0_br:
	jmp	Ltmp13
# FDATA: 1 _Z3inci #Ltmp0_br# 1 _Z3inci #Ltmp13# 0 167

Ltmp1:
	movl	total(%rip), %eax
	addl	$0x2, %eax
	movl	%eax, total(%rip)
	movl	$0x2, %eax
Ltmp1_br:
	jmp	Ltmp13
# FDATA: 1 _Z3inci #Ltmp1_br# 1 _Z3inci #Ltmp13# 0 151

Ltmp2:
	movl	total(%rip), %eax
	addl	$0x3, %eax
	movl	%eax, total(%rip)
	movl	$0x3, %eax
Ltmp2_br:
	jmp	Ltmp13
# FDATA: 1 _Z3inci #Ltmp2_br# 1 _Z3inci #Ltmp13# 0 152

Ltmp3:
	movl	total(%rip), %eax
	addl	$0x4, %eax
	movl	%eax, total(%rip)
	movl	$0x4, %eax
Ltmp3_br:
	jmp	Ltmp13
# FDATA: 1 _Z3inci #Ltmp3_br# 1 _Z3inci #Ltmp13# 0 146

Ltmp4:
	movl	total(%rip), %eax
	addl	$0x5, %eax
	movl	%eax, total(%rip)
	movl	$0x5, %eax
Ltmp4_br:
	jmp	Ltmp13
# FDATA: 1 _Z3inci #Ltmp4_br# 1 _Z3inci #Ltmp13# 0 149

Ltmp5:
	movl	total(%rip), %eax
	addl	$0x6, %eax
	movl	%eax, total(%rip)
	movl	$0x6, %eax
Ltmp5_br:
	jmp	Ltmp13
# FDATA: 1 _Z3inci #Ltmp5_br# 1 _Z3inci #Ltmp13# 0 150

Ltmp12:
	movl	-0x4(%rbp), %eax
	addl	$0x1, %eax

Ltmp13:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size _Z3inci, .-_Z3inci
# Jump tables
.section .rodata
"JUMP_TABLE/_Z3inci.0":
	.quad	Ltmp0
	.quad	Ltmp1
	.quad	Ltmp2
	.quad	Ltmp3
	.quad	Ltmp4
	.quad	Ltmp5

# BinaryData
.section .bss
"total":

	.text
  .globl _Z7inc_dupi
  .type _Z7inc_dupi, %function
_Z7inc_dupi:
	.cfi_startproc
LBB01:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -0x4(%rbp)
	cmpl	$0x5, -0x4(%rbp)
LBB01_br:
	ja	Ltmp14
# FDATA: 1 _Z7inc_dupi #LBB01_br# 1 _Z7inc_dupi #Ltmp14# 143 144
# FDATA: 1 _Z7inc_dupi #LBB01_br# 1 _Z7inc_dupi #LFT1# 0 777

LFT1:
	movl	-0x4(%rbp), %eax
	movq	"JUMP_TABLE/_Z7inc_dupi.0"(,%rax,8), %rax
LFT1_br:
	jmpq	*%rax
# FDATA: 1 _Z7inc_dupi #LFT1_br# 1 _Z7inc_dupi #Ltmp6# 130 137
# FDATA: 1 _Z7inc_dupi #LFT1_br# 1 _Z7inc_dupi #Ltmp7# 126 136
# FDATA: 1 _Z7inc_dupi #LFT1_br# 1 _Z7inc_dupi #Ltmp8# 122 130
# FDATA: 1 _Z7inc_dupi #LFT1_br# 1 _Z7inc_dupi #Ltmp9# 111 130
# FDATA: 1 _Z7inc_dupi #LFT1_br# 1 _Z7inc_dupi #Ltmp10# 122 140
# FDATA: 1 _Z7inc_dupi #LFT1_br# 1 _Z7inc_dupi #Ltmp11# 104 114

Ltmp6:
	movl	total(%rip), %eax
	addl	$0x2, %eax
	movl	%eax, total(%rip)
	movl	$0x1, %eax
Ltmp6_br:
	jmp	Ltmp15
# FDATA: 1 _Z7inc_dupi #Ltmp6_br# 1 _Z7inc_dupi #Ltmp15# 0 106

Ltmp7:
	movl	total(%rip), %eax
	addl	$0x3, %eax
	movl	%eax, total(%rip)
	movl	$0x2, %eax
Ltmp7_br:
	jmp	Ltmp15
# FDATA: 1 _Z7inc_dupi #Ltmp7_br# 1 _Z7inc_dupi #Ltmp15# 0 113

Ltmp8:
	movl	total(%rip), %eax
	addl	$0x4, %eax
	movl	%eax, total(%rip)
	movl	$0x3, %eax
Ltmp8_br:
	jmp	Ltmp15
# FDATA: 1 _Z7inc_dupi #Ltmp8_br# 1 _Z7inc_dupi #Ltmp15# 0 97

Ltmp9:
	movl	total(%rip), %eax
	addl	$0x5, %eax
	movl	%eax, total(%rip)
	movl	$0x4, %eax
Ltmp9_br:
	jmp	Ltmp15
# FDATA: 1 _Z7inc_dupi #Ltmp9_br# 1 _Z7inc_dupi #Ltmp15# 0 105

Ltmp10:
	movl	total(%rip), %eax
	addl	$0x6, %eax
	movl	%eax, total(%rip)
	movl	$0x5, %eax
Ltmp10_br:
	jmp	Ltmp15
# FDATA: 1 _Z7inc_dupi #Ltmp10_br# 1 _Z7inc_dupi #Ltmp15# 0 98

Ltmp11:
	movl	total(%rip), %eax
	addl	$0x7, %eax
	movl	%eax, total(%rip)
	movl	$0x6, %eax
Ltmp11_br:
	jmp	Ltmp15
# FDATA: 1 _Z7inc_dupi #Ltmp11_br# 1 _Z7inc_dupi #Ltmp15# 0 92

Ltmp14:
	movl	-0x4(%rbp), %eax
	addl	$0x1, %eax

Ltmp15:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq

	.cfi_endproc
.size _Z7inc_dupi, .-_Z7inc_dupi
# Jump tables
.section .rodata
"JUMP_TABLE/_Z7inc_dupi.0":
	.quad	Ltmp6
	.quad	Ltmp7
	.quad	Ltmp8
	.quad	Ltmp9
	.quad	Ltmp10
	.quad	Ltmp11
