# clang++ -O2 helper.cpp -c -o helperProf.o
# int returnFive() {
#   return 5;
# }
# int returnFourOrFive(int val) {
#   return val == 1 ? 4 : 5;
# }

	.text
	.file	"helper.cpp"
	.globl	_Z10returnFivev                 # -- Begin function _Z10returnFivev
	.p2align	4, 0x90
	.type	_Z10returnFivev,@function
_Z10returnFivev:                        # @_Z10returnFivev
	.cfi_startproc
# %bb.0:                                # %entry
	movl	$5, %eax
	retq
.Lfunc_end0:
	.size	_Z10returnFivev, .Lfunc_end0-_Z10returnFivev
	.cfi_endproc
                                        # -- End function
	.globl	_Z16returnFourOrFivei           # -- Begin function _Z16returnFourOrFivei
	.p2align	4, 0x90
	.type	_Z16returnFourOrFivei,@function
_Z16returnFourOrFivei:                  # @_Z16returnFourOrFivei
	.cfi_startproc
# %bb.0:                                # %entry
	xorl	%eax, %eax
	cmpl	$1, %edi
	sete	%al
	xorl	$5, %eax
	retq
.Lfunc_end1:
	.size	_Z16returnFourOrFivei, .Lfunc_end1-_Z16returnFourOrFivei
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
