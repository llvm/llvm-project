# clang++ helper.cpp -c -o
# #define MY_CONST const
# int FooVar = 1;
# int BarVar = 2;
# [[clang::noinline]]
# MY_CONST int barAddHdlper(int a, int b) {
#   return a + b;
# }
#
# MY_CONST int (*const funcGlobalBarMulExt)(int, int) = barAddHdlper;
# MY_CONST int fooGlobalFuncHelper(int a, int b) {
#   return 5 + funcGlobalBarMulExt(a, b);
# }

	.text
	.file	"helper.cpp"
	.globl	_Z12barAddHdlperii              # -- Begin function _Z12barAddHdlperii
	.p2align	4, 0x90
	.type	_Z12barAddHdlperii,@function
_Z12barAddHdlperii:                     # @_Z12barAddHdlperii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	addl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z12barAddHdlperii, .Lfunc_end0-_Z12barAddHdlperii
	.cfi_endproc
                                        # -- End function
	.globl	_Z19fooGlobalFuncHelperii       # -- Begin function _Z19fooGlobalFuncHelperii
	.p2align	4, 0x90
	.type	_Z19fooGlobalFuncHelperii,@function
_Z19fooGlobalFuncHelperii:              # @_Z19fooGlobalFuncHelperii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %edi
	movl	-8(%rbp), %esi
	callq	_Z12barAddHdlperii
	addl	$5, %eax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_Z19fooGlobalFuncHelperii, .Lfunc_end1-_Z19fooGlobalFuncHelperii
	.cfi_endproc
                                        # -- End function
	.type	FooVar,@object                  # @FooVar
	.data
	.globl	FooVar
	.p2align	2, 0x0
FooVar:
	.long	1                               # 0x1
	.size	FooVar, 4

	.type	BarVar,@object                  # @BarVar
	.globl	BarVar
	.p2align	2, 0x0
BarVar:
	.long	2                               # 0x2
	.size	BarVar, 4

	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z12barAddHdlperii
