	.text
  .globl main
  .type main, %function
main:
# FDATA: 0 [unknown] 0 1 main 0 0 1
	.cfi_startproc
LBB00: 
	subq	$0x8, %rsp
	.cfi_def_cfa_offset 16
	movl	$Input, %esi
	movl	$SYMBOLat0x4006c4, %edi
	xorl	%eax, %eax
	movl	$0x0, Input(%rip)
	callq	scanf@PLT
	movl	Input(%rip), %edx
	movl	$0xf4240, %eax
LBB00_br: 	movl	%edx, %esi
# FDATA: 1 main #LBB00_br# 1 main #Ltmp9# 0 0

Ltmp9: 
	cmpl	$0x8, %edx
Ltmp9_br: 	ja	Ltmp0
# FDATA: 1 main #Ltmp9_br# 1 main #Ltmp0# 0 0
# FDATA: 1 main #Ltmp9_br# 1 main #LFT0# 0 40

LFT0: 
	movl	%edx, %ecx
LFT0_br: 	jmpq	*"JUMP_TABLE/main.0"(,%rcx,8)
# FDATA: 1 main #LFT0_br# 1 main #Ltmp8# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp7# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp3# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp2# 0 40
# FDATA: 1 main #LFT0_br# 1 main #Ltmp1# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp5# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp4# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp6# 0 0
# FDATA: 1 main #LFT0_br# 1 main #Ltmp0# 0 0

Ltmp8: 
Ltmp8_br: 	addl	$0xa, %esi
# FDATA: 1 main #Ltmp8_br# 1 main #Ltmp10# 0 0

Ltmp10: 
	subl	$0x1, %eax
Ltmp10_br: 	jne	Ltmp9
# FDATA: 1 main #Ltmp10_br# 1 main #Ltmp9# 0 45
# FDATA: 1 main #Ltmp10_br# 1 main #LFT1# 0 0

LFT1: 
	movl	$SYMBOLat0x4006c7, %edi
	xorl	%eax, %eax
	movl	%esi, Value(%rip)
	callq	printf@PLT
	xorl	%eax, %eax
	addq	$0x8, %rsp
	.cfi_def_cfa_offset 8
	retq
	.cfi_def_cfa %rsp, 16

Ltmp7: 
	addl	$0x9, %esi
Ltmp7_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp7_br# 1 main #Ltmp10# 0 0

Ltmp3: 
	addl	$0x5, %esi
Ltmp3_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp3_br# 1 main #Ltmp10# 0 0

Ltmp2: 
	addl	$0x4, %esi
Ltmp2_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp2_br# 1 main #Ltmp10# 0 43

Ltmp1: 
	addl	$0x3, %esi
Ltmp1_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp1_br# 1 main #Ltmp10# 0 0

Ltmp5: 
	addl	$0x7, %esi
Ltmp5_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp5_br# 1 main #Ltmp10# 0 0

Ltmp4: 
	addl	$0x6, %esi
Ltmp4_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp4_br# 1 main #Ltmp10# 0 0

Ltmp6: 
	addl	$0x8, %esi
Ltmp6_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp6_br# 1 main #Ltmp10# 0 0

Ltmp0: 
	addl	$0x2, %esi
Ltmp0_br: 	jmp	Ltmp10
# FDATA: 1 main #Ltmp0_br# 1 main #Ltmp10# 0 0

	.cfi_endproc
.size main, .-main
# Jump tables
.section .rodata
"JUMP_TABLE/main.0":
	.quad	Ltmp0
	.quad	Ltmp1
	.quad	Ltmp2
	.quad	Ltmp3
	.quad	Ltmp4
	.quad	Ltmp5
	.quad	Ltmp6
	.quad	Ltmp7
	.quad	Ltmp8

# BinaryData
"SYMBOLat0x4006c4":
"SYMBOLat0x4006c7": 
.section .bss
"Value": 
"Input": 
