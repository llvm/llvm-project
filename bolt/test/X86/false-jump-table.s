## Check that jump table detection does not fail on a false
## reference to a jump table.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe --print-cfg \
# RUN:    --print-only=inc_dup -o %t.out | FileCheck %s

	.file	"jump_table.c"
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
# Control flow confusing for JT detection
# CHECK: leaq    "JUMP_TABLE{{.*}}"(%rip), %rdx
	leaq	.L4(%rip), %rdx
  jmp .LJT
# CHECK: leaq    {{.*}}(%rip), %rdx
	leaq	.LC0(%rip), %rdx
  jmp .L10
.LJT:
  movslq  (%rdx,%rax,4), %rax
	addq	%rdx, %rax
# CHECK: jmpq    *%rax # UNKNOWN CONTROL FLOW
	jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L3-.L4
	.long	.L5-.L4
	.long	.L6-.L4
	.long	.L7-.L4
	.long	.L8-.L4
	.long	.L9-.L4
	.text
.L3:
	leaq	.LC0(%rip), %rdi
	call	puts@PLT
	movl	$1, %eax
	jmp	.L10
.L5:
	leaq	.LC1(%rip), %rdi
	call	puts@PLT
	movl	$2, %eax
	jmp	.L10
.L6:
	leaq	.LC2(%rip), %rdi
	call	puts@PLT
	movl	$3, %eax
	jmp	.L10
.L7:
	leaq	.LC3(%rip), %rdi
	call	puts@PLT
	movl	$4, %eax
	jmp	.L10
.L8:
	leaq	.LC4(%rip), %rdi
	call	puts@PLT
	movl	$5, %eax
	jmp	.L10
.L9:
	leaq	.LC5(%rip), %rdi
	call	puts@PLT
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
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movq	%rsi, -16(%rbp)
	movl	-4(%rbp), %eax
	addl	$9, %eax
	movl	%eax, %edi
	call	inc_dup@PLT
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.ident	"GCC: (GNU) 6.3.0"
	.section	.note.GNU-stack,"",@progbits
