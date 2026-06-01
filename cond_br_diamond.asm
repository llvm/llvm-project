	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_b1p0_zmmul1p0_zba1p0_zbb1p0_zbs1p0"
	.file	"LLVMDialectModule"
	.text
	.globl	diamond_cond_br                 # -- Begin function diamond_cond_br
	.p2align	2
	.type	diamond_cond_br,@function
diamond_cond_br:                        # @diamond_cond_br
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	mv	a3, a2
	mv	a2, a1
	sext.w	a1, a0
	sd	a3, 16(sp)                      # 8-byte Folded Spill
	sd	a2, 24(sp)                      # 8-byte Folded Spill
	li	a0, 0
	bge	a0, a1, .LBB0_2
	j	.LBB0_1
.LBB0_1:
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	j	.LBB0_3
.LBB0_2:
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	sd	a0, 8(sp)                       # 8-byte Folded Spill
	j	.LBB0_3
.LBB0_3:
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	diamond_cond_br, .Lfunc_end0-diamond_cond_br
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
