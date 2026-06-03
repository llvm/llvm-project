	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_b1p0_zmmul1p0_zba1p0_zbb1p0_zbs1p0"
	.file	"gep.ll"
	.text
	.globl	gep_i64                         # -- Begin function gep_i64
	.p2align	2
	.type	gep_i64,@function
gep_i64:                                # @gep_i64
	.cfi_startproc
# %bb.0:
	sh3add	a0, a1, a0
	ret
.Lfunc_end0:
	.size	gep_i64, .Lfunc_end0-gep_i64
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
