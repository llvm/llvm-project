  .globl _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 792
	.cfi_startproc
LBB00:
	movl	$0x0, %eax
  jmp	Ltmp0

LFT0:
	cmpl	$0x0, %eax
  jmp	Ltmp1

Ltmp1:
  jmp	LBB00
# FDATA: 1 _start #Ltmp1# 1 _start #LBB00# 13 792

Ltmp0:
  jmp	Ltmp2

Ltmp2:
  jmp	Ltmp3

Ltmp3:
  jmp	Ltmp4

Ltmp4:
  jmp	Ltmp5

Ltmp5:
  jmp	Ltmp6

Ltmp6:
  jmp	Ltmp7

Ltmp7:
  jmp	Ltmp8

Ltmp8:
  jmp	Ltmp9

Ltmp9:
  jmp	Ltmp10

Ltmp10:
  jmp	Ltmp11

Ltmp11:
	retq
	.cfi_endproc
.size _start, .-_start
