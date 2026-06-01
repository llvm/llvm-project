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
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	.cfi_remember_state
	mv	a3, a2
	mv	a2, a1
	mv	a1, a0
	andi	a0, a3, 1
	sd	a2, 0(sp)                       # 8-byte Folded Spill
	sd	a1, 8(sp)                       # 8-byte Folded Spill
	beqz	a0, .LBB0_2
	j	.LBB0_1
.LBB0_1:
	ld	a0, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.LBB0_2:
	.cfi_restore_state
	ld	a0, 0(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 16
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	cond_br_demo, .Lfunc_end0-cond_br_demo
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
