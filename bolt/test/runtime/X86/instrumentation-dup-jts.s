# This reproduces a bug with instrumentation when trying to instrument
# functions that share a jump table with multiple indirect jumps. Usually,
# each indirect jump that uses a JT will have its own copy of it. When
# this does not happen, we need to duplicate the jump table safely, so
# we can split the edges correctly (each copy of the jump table may have
# different split edges). For this to happen, we need to correctly match
# the sequence of instructions that perform the indirect jump to identify
# the base address of the jump table and patch it to point to the new
# cloned JT.
#
# Here we test this variant:
#	  movq	jt.2397(,%rax,8), %rax
#   jmp	*%rax
#
# Which is suboptimal since the compiler could've avoided using an intermediary
# register, but GCC does generate this code and it triggered a bug in our
# matcher. Usual jumps in non-PIC code have this format:
#
#   jmp	*jt.2397(,%rax,8)
#
# This is the C code fed to GCC:
#  #include <stdio.h>
#int interp(char* code) {
#    static void* jt[] = { &&op_end, &&op_inc, &&do_dec };
#    int pc = 0;
#    int res = 0;
#    goto *jt[code[pc++] - '0'];
#
#op_inc:
#    res += 1;
#    printf("%d\n", res);
#    goto *jt[code[pc++] - '0'];
#do_dec:
#    res -= 1;
#    printf("%d\n", res);
#    goto *jt[code[pc++] - '0'];
#op_end:
#    return res;
#}
#int main(int argc, char** argv) {
#    return interp(argv[1]);
#}


# REQUIRES: system-linux,bolt-runtime

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -instrument -instrumentation-file=%t.fdata \
# RUN:   -o %t.instrumented

# Instrumented program needs to finish returning zero
# RUN: %t.instrumented 120

# Test that the instrumented data makes sense
# RUN:  llvm-bolt %t.exe -o %t.bolted -data %t.fdata \
# RUN:    -reorder-blocks=cache+ -reorder-functions=hfsort+ \
# RUN:    -print-only=interp -print-finalized | FileCheck %s

# RUN: %t.bolted 120

# Check that our two indirect jumps are recorded in the fdata file and that
# each has its own independent profile
# CHECK:  Successors: .Ltmp1 (mispreds: 0, count: 1), .Ltmp0 (mispreds: 0, count: 0), .Ltmp2 (mispreds: 0, count: 0)
# CHECK:  Successors: .Ltmp0 (mispreds: 0, count: 1), .Ltmp2 (mispreds: 0, count: 1), .Ltmp1 (mispreds: 0, count: 0)

	.file	"test.c"
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%d\n"
	.text
	.p2align 4,,15
	.globl	interp
	.type	interp, @function
interp:
.LFB11:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	xorl	%ebp, %ebp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	leaq	1(%rdi), %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movsbl	(%rdi), %eax
	subl	$48, %eax
	cltq
	movq	jt.2397(,%rax,8), %rax
	jmp	*%rax
	.p2align 4,,10
	.p2align 3
.L3:
	addl	$1, %ebp
.L8:
	movl	%ebp, %esi
	movl	$.LC0, %edi
	xorl	%eax, %eax
	addq	$1, %rbx
	call	printf
	movsbl	-1(%rbx), %eax
	subl	$48, %eax
	cltq
	movq	jt.2397(,%rax,8), %rax
	jmp	*%rax
	.p2align 4,,10
	.p2align 3
.L6:
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	movl	%ebp, %eax
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L4:
	.cfi_restore_state
	subl	$1, %ebp
	jmp	.L8
	.cfi_endproc
.LFE11:
	.size	interp, .-interp
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB12:
	.cfi_startproc
	movq	8(%rsi), %rdi
	jmp	interp
	.cfi_endproc
.LFE12:
	.size	main, .-main
	.section	.rodata
	.align 16
	.type	jt.2397, @object
	.size	jt.2397, 24
jt.2397:
	.quad	.L6
	.quad	.L3
	.quad	.L4
	.ident	"GCC: (GNU) 8"
	.section	.note.GNU-stack,"",@progbits
