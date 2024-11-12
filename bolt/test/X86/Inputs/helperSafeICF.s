# clang++ -c helper.cpp -o helper.o
# int FooVar = 1;
# int BarVar = 2;
#
# int fooGlobalFuncHelper(int a, int b) {
#   return 5;
# }
	.text
	.file	"helper.cpp"
	.globl	_Z19fooGlobalFuncHelperii       # -- Begin function _Z19fooGlobalFuncHelperii
	.p2align	4, 0x90
	.type	_Z19fooGlobalFuncHelperii,@function
_Z19fooGlobalFuncHelperii:              # @_Z19fooGlobalFuncHelperii
	.cfi_startproc
# %bb.0:
	movl	$5, %eax
	retq
.Lfunc_end0:
	.size	_Z19fooGlobalFuncHelperii, .Lfunc_end0-_Z19fooGlobalFuncHelperii
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
