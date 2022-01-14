# This reproduces a bug in TailDuplication::isInCacheLine
# with accessing BlockLayout past bounds (unreachable blocks).

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out -data %t.fdata -relocs \
# RUN:   -tail-duplication=1 -tail-duplication-aggressive=1
  .globl _start
_start:
  jmp	d
  je	_start
  movl	%esi, %edi
d:
  jmpq	*JT0(,%rcx,8)
# FDATA: 1 _start #d# 1 _start #e# 1 3
f:
	movl	0, %esi
g:
	movl	0, %esi
e:
  jmp	f

.rodata
JT0:
	.quad	g
	.quad	e
