	.text
	.file	"acle_sme2_cvtl.c"
	.globl	test_cvtl_f32_x2                // -- Begin function test_cvtl_f32_x2
	.p2align	2
	.type	test_cvtl_f32_x2,@function
	.variant_pcs	test_cvtl_f32_x2
test_cvtl_f32_x2:                       // @test_cvtl_f32_x2
.Ltest_cvtl_f32_x2$local:
	.type	.Ltest_cvtl_f32_x2$local,@function
// %bb.0:                               // %entry
	str	x29, [sp, #-16]!                // 8-byte Folded Spill
	addvl	sp, sp, #-1
	ptrue	p0.h
	st1h	{ z0.h }, p0, [sp]
	ld1h	{ z0.h }, p0/z, [sp]
	fcvtl	{ z2.s, z3.s }, z0.h
	mov	z0.d, z2.d
	mov	z1.d, z3.d
	addvl	sp, sp, #1
	ldr	x29, [sp], #16                  // 8-byte Folded Reload
	ret
.Lfunc_end0:
	.size	test_cvtl_f32_x2, .Lfunc_end0-test_cvtl_f32_x2
	.size	.Ltest_cvtl_f32_x2$local, .Lfunc_end0-test_cvtl_f32_x2
                                        // -- End function
	.ident	"clang version 19.0.0git (git@github.com:Lukacma/llvm-project.git 176083b8562ef5f6b265ed14a3d4f81e4555ee6e)"
	.section	".note.GNU-stack","",@progbits
