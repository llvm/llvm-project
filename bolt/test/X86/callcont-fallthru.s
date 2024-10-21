## Ensures that a call continuation fallthrough count is set when using
## pre-aggregated perf data.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: link_fdata %s %t.exe %t.pa PREAGG
# RUN: llvm-strip --strip-unneeded %t.exe
# RUN: llvm-bolt %t.exe --pa -p %t.pa -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s

  .globl foo
  .type foo, %function
foo:
	pushq	%rbp
	movq	%rsp, %rbp
	popq	%rbp
Lfoo_ret:
	retq
.size foo, .-foo

  .globl main
  .type main, %function
main:
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$0x20, %rsp
	movl	$0x0, -0x4(%rbp)
	movl	%edi, -0x8(%rbp)
	movq	%rsi, -0x10(%rbp)
	callq	puts@PLT
# PREAGG: B X:0 #Ltmp1# 2 0
# CHECK:      callq puts@PLT
# CHECK-NEXT: count: 2

Ltmp1:
	movq	-0x10(%rbp), %rax
	movq	0x8(%rax), %rdi
	movl	%eax, -0x14(%rbp)

Ltmp4:
	cmpl	$0x0, -0x14(%rbp)
	je	Ltmp0

	movl	$0xa, -0x18(%rbp)
	callq	foo
# PREAGG: B #Lfoo_ret# #Ltmp3# 1 0
# CHECK:      callq foo
# CHECK-NEXT: count: 1

Ltmp3:
	cmpl	$0x0, -0x18(%rbp)
	jmp	Ltmp2

Ltmp2:
	movl	-0x18(%rbp), %eax
	addl	$-0x1, %eax
	movl	%eax, -0x18(%rbp)
	jmp	Ltmp3
	jmp	Ltmp4
	jmp	Ltmp1

Ltmp0:
	xorl	%eax, %eax
	addq	$0x20, %rsp
	popq	%rbp
	retq
.size main, .-main
