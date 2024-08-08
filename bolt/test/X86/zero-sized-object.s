## Check that references to local (unnamed) objects below are not
## treated as references relative to zero-sized A object.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --print-cfg \
# RUN:    --print-only=main -o %t.out | FileCheck %s

	.file	"rust_bug.c"
	.section	.rodata
.LC0:
	.string	"0"
.LC1:
	.string	"1"
.LC2:
	.string	"2"
.LC3:
	.string	"3"
.LC4:
	.string	"4"
.LC5:
	.string	"5"
	.text
	.globl	inc_dup
	.type	inc_dup, @function
inc_dup:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	-4(%rbp), %eax
	subl	$10, %eax
	cmpl	$5, %eax
	ja	.L2
	movl	%eax, %eax
	movq	.L4(,%rax,8), %rax
	jmp	*%rax
	.section	.rodata
  .align 8
  .globl A
  .type A, @object
A:
	.section	.rodata
	.align 8
	.align 4
.L4:
	.quad	.L3
	.quad	.L5
	.quad	.L6
	.quad	.L7
	.quad	.L8
	.quad	.L9
	.text
.L3:
	movl	$.LC0, %edi
	call	puts
	movl	$1, %eax
	jmp	.L10
.L5:
	movl	$.LC1, %edi
	call	puts
	movl	$2, %eax
	jmp	.L10
.L6:
	movl	$.LC2, %edi
	call	puts
	movl	$3, %eax
	jmp	.L10
.L7:
	movl	$.LC3, %edi
	call	puts
	movl	$4, %eax
	jmp	.L10
.L8:
	movl	$.LC4, %edi
	call	puts
	movl	$5, %eax
	jmp	.L10
.L9:
	movl	$.LC5, %edi
	call	puts
	movl	$6, %eax
	jmp	.L10
.L2:
	movl	-4(%rbp), %eax
	addl	$1, %eax
.L10:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	inc_dup, .-inc_dup
	.section	.rodata
.LC6:
	.string	"%d\n"
	.text
	.globl	main
	.type	main, @function
main:
# CHECK: Binary Function "main" after building cfg
.LFB1:
  .cfi_startproc
  pushq   %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset 6, -16
  movq    %rsp, %rbp
  .cfi_def_cfa_register 6
  movl    .LN(%rip), %eax
# We should not reference a data object as an offset from A
# CHECK-NOT: movl    A+{{.*}}(%rip), %eax
  movl    %eax, %esi
  movl    $.LC6, %edi
  movl    $0, %eax
  call    printf
  movl    $0, %eax
  popq    %rbp
  .cfi_def_cfa 7, 8
  ret
  .cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 4
.LN:
	.long	42
	.ident	"GCC: (GNU) 4.8.5 20150623 (Red Hat 4.8.5-44)"
	.section	.note.GNU-stack,"",@progbits
