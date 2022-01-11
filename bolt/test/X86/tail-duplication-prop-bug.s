# This reproduces a bug in aggressive tail duplication/copy propagation.
# XFAIL: *

# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out -data %t.fdata -relocs \
# RUN:   -tail-duplication=1 -tail-duplication-aggressive=1 \
# RUN:   -tail-duplication-const-copy-propagation=1

  .globl a
a:
	.cfi_startproc
	jmpq	*JT(,%rcx,8)
b:
	jb	d
# FDATA: 1 a #b# 1 a #d# 6 60
e:
	cmpl	%eax, %ebx
f:
	jmp	g
# FDATA: 1 a #f# 1 a #g# 0 8
d:
	movl	$0x1, %ebx
	jmp	e
	jmp	g
h:
	jmp	h
i:
	jne	j
g:
	jmp	g
j:
	.cfi_endproc
.rodata
JT:
	.quad	b
	.quad	i
