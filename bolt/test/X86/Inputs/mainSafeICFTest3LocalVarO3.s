# clang++ -O3 -c main.cpp -o main.o
# extern int FooVar;
# extern int BarVar;
# [[clang::noinline]]
# int fooSub(int a, int b) {
#   return a - b;
# }
# [[clang::noinline]]
# int barSub(int a, int b) {
#   return a - b;
# }
# [[clang::noinline]]
# int fooMul(int a, int b) {
#   return a * b;
# }
# [[clang::noinline]]
# int barMul(int a, int b) {
#   return a * b;
# }
# [[clang::noinline]]
# int fooAdd(int a, int b) {
#   return a + b;
# }
# [[clang::noinline]]
# int barAdd(int a, int b) {
#   return a + b;
# }
# [[clang::noinline]]
# int helper1(int (*func)(int, int), int a, int b) {
#   if (func == barAdd)
#     return 1;
#   return func(a, b) - 4;
# }
# [[clang::noinline]]
# int helper2(int (*func)(int, int), int (*func2)(int, int), int a, int b) {
#   if (func == func2)
#     return 2;
#   return func(a, b) + func2(a, b);
# }
# int main(int argc, char **argv) {
#   static int (*funcGlobalBarAdd)(int, int) = barAdd;
#   int (*funcGlobalBarMul)(int, int) = barMul;
#   int temp = helper1(funcGlobalBarAdd, FooVar, BarVar) +
#              helper2(fooMul, funcGlobalBarMul, FooVar, BarVar) + fooSub(FooVar, BarVar) +
#              barSub(FooVar, BarVar) + fooAdd(FooVar, BarVar);
#   MY_PRINTF("val: %d", temp);
#   return temp;
# }
	.text
	.file	"main.cpp"
	.globl	_Z6fooSubii                     # -- Begin function _Z6fooSubii
	.p2align	4, 0x90
	.type	_Z6fooSubii,@function
_Z6fooSubii:                            # @_Z6fooSubii
	.cfi_startproc
# %bb.0:
	movl	%edi, %eax
	subl	%esi, %eax
	retq
.Lfunc_end0:
	.size	_Z6fooSubii, .Lfunc_end0-_Z6fooSubii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barSubii                     # -- Begin function _Z6barSubii
	.p2align	4, 0x90
	.type	_Z6barSubii,@function
_Z6barSubii:                            # @_Z6barSubii
	.cfi_startproc
# %bb.0:
	movl	%edi, %eax
	subl	%esi, %eax
	retq
.Lfunc_end1:
	.size	_Z6barSubii, .Lfunc_end1-_Z6barSubii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6fooMulii                     # -- Begin function _Z6fooMulii
	.p2align	4, 0x90
	.type	_Z6fooMulii,@function
_Z6fooMulii:                            # @_Z6fooMulii
	.cfi_startproc
# %bb.0:
	movl	%edi, %eax
	imull	%esi, %eax
	retq
.Lfunc_end2:
	.size	_Z6fooMulii, .Lfunc_end2-_Z6fooMulii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barMulii                     # -- Begin function _Z6barMulii
	.p2align	4, 0x90
	.type	_Z6barMulii,@function
_Z6barMulii:                            # @_Z6barMulii
	.cfi_startproc
# %bb.0:
	movl	%edi, %eax
	imull	%esi, %eax
	retq
.Lfunc_end3:
	.size	_Z6barMulii, .Lfunc_end3-_Z6barMulii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6fooAddii                     # -- Begin function _Z6fooAddii
	.p2align	4, 0x90
	.type	_Z6fooAddii,@function
_Z6fooAddii:                            # @_Z6fooAddii
	.cfi_startproc
# %bb.0:
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	leal	(%rdi,%rsi), %eax
	retq
.Lfunc_end4:
	.size	_Z6fooAddii, .Lfunc_end4-_Z6fooAddii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barAddii                     # -- Begin function _Z6barAddii
	.p2align	4, 0x90
	.type	_Z6barAddii,@function
_Z6barAddii:                            # @_Z6barAddii
	.cfi_startproc
# %bb.0:
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	leal	(%rdi,%rsi), %eax
	retq
.Lfunc_end5:
	.size	_Z6barAddii, .Lfunc_end5-_Z6barAddii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper1PFiiiEii              # -- Begin function _Z7helper1PFiiiEii
	.p2align	4, 0x90
	.type	_Z7helper1PFiiiEii,@function
_Z7helper1PFiiiEii:                     # @_Z7helper1PFiiiEii
	.cfi_startproc
# %bb.0:
	leaq	_Z6barAddii(%rip), %rcx
	cmpq	%rcx, %rdi
	je	.LBB6_1
# %bb.2:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movq	%rdi, %rax
	movl	%esi, %edi
	movl	%edx, %esi
	callq	*%rax
	addl	$-4, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	retq
.LBB6_1:
	movl	$1, %eax
	retq
.Lfunc_end6:
	.size	_Z7helper1PFiiiEii, .Lfunc_end6-_Z7helper1PFiiiEii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper2PFiiiES0_ii           # -- Begin function _Z7helper2PFiiiES0_ii
	.p2align	4, 0x90
	.type	_Z7helper2PFiiiES0_ii,@function
_Z7helper2PFiiiES0_ii:                  # @_Z7helper2PFiiiES0_ii
	.cfi_startproc
# %bb.0:
	cmpq	%rsi, %rdi
	je	.LBB7_1
# %bb.2:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%ecx, %ebx
	movl	%edx, %ebp
	movq	%rsi, %r14
	movq	%rdi, %rax
	movl	%edx, %edi
	movl	%ecx, %esi
	callq	*%rax
	movl	%eax, %r15d
	movl	%ebp, %edi
	movl	%ebx, %esi
	callq	*%r14
	addl	%r15d, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r14
	.cfi_restore %r15
	.cfi_restore %rbp
	retq
.LBB7_1:
	movl	$2, %eax
	retq
.Lfunc_end7:
	.size	_Z7helper2PFiiiES0_ii, .Lfunc_end7-_Z7helper2PFiiiES0_ii
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	FooVar@GOTPCREL(%rip), %r14
	movl	(%r14), %esi
	movq	BarVar@GOTPCREL(%rip), %r15
	movl	(%r15), %edx
	leaq	_Z6barAddii(%rip), %rdi
	callq	_Z7helper1PFiiiEii
	movl	%eax, %ebx
	movl	(%r14), %edx
	movl	(%r15), %ecx
	leaq	_Z6fooMulii(%rip), %rdi
	leaq	_Z6barMulii(%rip), %rsi
	callq	_Z7helper2PFiiiES0_ii
	movl	%eax, %ebp
	addl	%ebx, %ebp
	movl	(%r14), %ebx
	movl	(%r15), %r14d
	movl	%ebx, %edi
	movl	%r14d, %esi
	callq	_Z6fooSubii
	movl	%eax, %r15d
	movl	%ebx, %edi
	movl	%r14d, %esi
	callq	_Z6barSubii
	movl	%eax, %r12d
	addl	%r15d, %r12d
	addl	%ebp, %r12d
	movl	%ebx, %edi
	movl	%r14d, %esi
	callq	_Z6fooAddii
	addl	%r12d, %eax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end8:
	.size	main, .Lfunc_end8-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z6fooMulii
	.addrsig_sym _Z6barMulii
	.addrsig_sym _Z6barAddii
