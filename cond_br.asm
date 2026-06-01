	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_b1p0_zmmul1p0_zba1p0_zbb1p0_zbs1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	cond_br_demo                    # -- Begin function cond_br_demo
	.p2align	2
	.type	cond_br_demo,@function
cond_br_demo:                           # @cond_br_demo
	.cfi_startproc
# %bb.0:
	andi	a2, a2, 1
	beqz	a2, .LBB0_2
# %bb.1:
	ret
.LBB0_2:
	mv	a0, a1
	ret
.Lfunc_end0:
	.size	cond_br_demo, .Lfunc_end0-cond_br_demo
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
