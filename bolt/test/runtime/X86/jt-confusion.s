# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags -no-pie -nostartfiles -nostdlib -lc %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -o %t.exe.bolt --relocs=1 --lite=0

# RUN: %t.exe.bolt

## Check that BOLT's jump table detection diffrentiates between
## __builtin_unreachable() targets and function pointers.

## The test case was built from the following two source files and
## modiffied for standalone build. main became _start, etc.
## $ $(CC) a.c -O1 -S -o a.s
## $ $(CC) b.c -O0 -S -o b.s

## a.c:

## typedef int (*fptr)(int);
## void check_fptr(fptr, int);
##
## int foo(int a) {
##   check_fptr(foo, 0);
##   switch (a) {
##   default:
##     __builtin_unreachable();
##   case 0:
##     return 3;
##   case 1:
##     return 5;
##   case 2:
##     return 7;
##   case 3:
##     return 11;
##   case 4:
##     return 13;
##   case 5:
##     return 17;
##   }
##   return 0;
## }
##
## int main(int argc) {
##   check_fptr(main, 1);
##   return foo(argc);
## }
##
## const fptr funcs[2] = {foo, main};

## b.c.:

## typedef int (*fptr)(int);
## extern const fptr funcs[2];
##
## #define assert(C) { if (!(C)) (*(unsigned long long *)0) = 0; }
## void check_fptr(fptr f, int i) {
##   assert(f == funcs[i]);
## }


	.text
	.globl	foo
	.type	foo, @function
foo:
.LFB0:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movl	%edi, %ebx
	movl	$0, %esi
	movl	$foo, %edi
	call	check_fptr
	movl	%ebx, %ebx
	jmp	*.L4(,%rbx,8)
.L8:
	movl	$5, %eax
	jmp	.L1
.L7:
	movl	$7, %eax
	jmp	.L1
.L6:
	movl	$11, %eax
	jmp	.L1
.L5:
	movl	$13, %eax
	jmp	.L1
.L3:
	movl	$17, %eax
	jmp	.L1
.L10:
	movl	$3, %eax
.L1:
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE0:
	.size	foo, .-foo
	.globl	_start
	.type	_start, @function
_start:
.LFB1:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movl	%edi, %ebx
	movl	$1, %esi
	movl	$_start, %edi
	call	check_fptr
	movl	$1, %edi
	call	foo
	popq	%rbx
	.cfi_def_cfa_offset 8
  callq exit@PLT
	.cfi_endproc
.LFE1:
	.size	_start, .-_start
	.globl	check_fptr
	.type	check_fptr, @function
check_fptr:
.LFB2:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	-12(%rbp), %eax
	cltq
	movq	funcs(,%rax,8), %rax
	cmpq	%rax, -8(%rbp)
	je	.L33
	movl	$0, %eax
	movq	$0, (%rax)
.L33:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc

	.section	.rodata
	.align 8
	.align 4
.L4:
	.quad	.L10
	.quad	.L8
	.quad	.L7
	.quad	.L6
	.quad	.L5
	.quad	.L3

	.globl	funcs
	.type	funcs, @object
	.size	funcs, 16
funcs:
	.quad	foo
	.quad	_start
