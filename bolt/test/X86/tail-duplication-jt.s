# This reproduces a bug in tail duplication when aggressiveCodeToDuplicate
# fails to handle a block with a jump table.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clangxx %cflags %t.o -o %t.exe -Wl,-q 
# RUN: llvm-bolt %t.exe -o %t.out -data %t.fdata -relocs \
# RUN:   -tail-duplication=1 -tail-duplication-aggressive=1 \
# RUN:   -print-cfg | FileCheck %s
# CHECK: Jump table JUMP_TABLE/a.0 for function a at {{.*}} with a total count of 3
  .globl main
main:
  .globl a
  .type a, %function
a:
	.cfi_startproc
b:
  jmp	c
  je	b
  movl	%esi, %edi
c:
	movb	0, %cl
d:
  jmp	e
	movq	0, %r14
f:
	je	d
  jmp	f
e:
g:
j:
	movq	%rbp, 0
h:
	cmpl	$0x41, 0
i:
  jmp	h
  jmp	i
  ja	o
	movl	%edx, 0
p:
q:
k:
  jmpq	*JT0(,%rcx,8)
# FDATA: 1 a #k# 1 a #l# 1 3
m:
	movl	0, %esi
r:
  jmpq	*JT1(,%rax,8)
	cmpl	1, %eax
  jmp	j
l:
  jmp	m
s:
  movl	6, %ebx
ak:
  jmp	e
	movl	0, %eax
am:
  jmp	p
 	jmp	q
o:
  jmp	g
n:
	xorl	%r12d, %r12d
	.cfi_endproc
.rodata
JT0:
	.quad	r
	.quad	l
	.quad	ak
JT1:
	.quad	s
	.quad	am
	.quad	n
