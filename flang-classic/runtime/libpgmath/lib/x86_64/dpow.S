/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */



#include "directives.h"

/*      .file "dpow.s"
 *   double __mth_i_dpowd(double x, double y)
 */
	.text
        ALN_FUNC
        .globl  ENT(__mth_i_dpowd)

ENT(__mth_i_dpowd):	/* x ** y */

	pushq	%rbp
	movq	%rsp,%rbp
	subq	$48,%rsp
	movsd	%xmm1, 32(%rsp)	/* y */
	movsd	%xmm0, 24(%rsp)	/* x */

        /* r8 holds flags for x, in rax */
        /* r9 holds flags for y, in rcx */
        /* rdx holds 1 */
        /* Use r10 and r11 for scratch */
        xor     %r8d, %r8d
        xor     %r9d, %r9d
        movl    $1, %edx
        movq    24(%rsp), %rax
        movq    32(%rsp), %rcx

        cmpq    .C4_D101(%rip), %rax   /* test x == 1.0 */
        cmove   %edx, %r8d
        cmpq    .C4_D102(%rip), %rcx   /* test y == 0.5 */
        cmove   %edx, %r9d
        movsd	.C4_D107(%rip), %xmm2

        cmpq    .C4_D103(%rip), %rax   /* test x == 0.0 */
        cmove   %edx, %r8d
        cmpq    .C4_D104(%rip), %rcx   /* test y == 1.5 */
        cmove   %edx, %r9d

        cmpq    .C4_D101(%rip), %rcx   /* test y == 1.0 */
        cmove   %edx, %r9d
        cmpq    .C4_D105(%rip), %rcx   /* test y == 0.25 */
        cmove   %edx, %r9d
        andpd	%xmm1, %xmm2

        or      %r9d, %r8d
        jnz     .L_Special_Pow_Cases_1

        movq    .C4_D106(%rip), %r10   /* test both for infinity and nan */
        movq    .C4_D106(%rip), %r11
        andq    %rcx, %r10
        andq    %rax, %r11
        cmpq    .C4_D106(%rip), %r10
        cmove   %edx, %r9d
        cmpq    .C4_D106(%rip), %r11
        cmove   %edx, %r8d

        movq    .C4_D107(%rip), %r10   /* test y for +/- 0.0 */
        movq    .C4_D108(%rip), %r11   /* test x for negative */
        andq    %rcx, %r10
        andq    %rax, %r11
        cmpq    .C4_D103(%rip), %r10
        cmove   %edx, %r9d
        cmpq    .C4_D108(%rip), %r11
        cmove   %edx, %r8d

        or      %r9d, %r8d
        jnz     .L_Special_Pow_Cases_2
        comisd	.C4_D109(%rip), %xmm2
        ja	.L__DY_is_large
        comisd	.C4_D10A(%rip), %xmm2
        jb	.L__DY_near_zero

/* ---------------------------------------------------------------------- */
.L_noopt:
	fldl	32(%rsp)		/* y */
	fldl	24(%rsp)		/* x */
	fxch %st(1)

	xorq	%rdx,%rdx
	comisd	.C4_D10E(%rip),%xmm1
	jae	.L_skipfistl
	comisd	.C4_D10F(%rip),%xmm1
	jbe	.L_skipfistl

	fnstcw -4(%rbp)
	movl -4(%rbp),%eax
	movb $12,%ah
	movl %eax,-8(%rbp)
	fldcw -8(%rbp)
	subq $8,%rsp
	fistl (%rsp)
	popq %rdx
	fldcw -4(%rbp)

.L_skipfistl:
	fldz
	fucom %st(2)
	fnstsw %ax
	andb $68,%ah
	xorb $64,%ah
	jne .L169
	fcomp %st(1)
	fnstsw %ax
	andb $69,%ah
	cmpb $1,%ah
	jne .L157
	fstp %st(0)
	fstp %st(0)
	fldz
	jmp	.L_stack_return
.L169:
	fstp %st(0)
.L157:
	pushq %rdx
	fildl (%rsp)
	addq $4,%rsp
	fucomp %st(1)
	fnstsw %ax
	andb $68,%ah
	xorb $64,%ah
	jne .L158
	fstp %st(0)
	fld1
	testl %edx,%edx
	jne .L159
.L172:
	fstp %st(1)
	jmp	.L_stack_return
.L159:
	testl %edx,%edx
	jge .L161
	negl %edx
	fld1
	fdivp %st,%st(2)
	jmp .L161
.L171:
	fxch %st(1)
.L161:
	testb $1,%dl
	je .L164
	fmul %st(1),%st
.L164:
	shrl $1,%edx
	je .L172
	fxch %st(1)
	fmul %st(0),%st
	jmp .L171
.L158:
	fxch %st(1)
	fld1
	fxch
	fyl2x
	fmul	%st(1),%st
	fst	%st(1)
	frndint
	fxch
	fsub	%st(1),%st
	f2xm1

	fld %st(1)
	fstp %st(0)
	faddl .C4_D101(%rip)
	fscale
	fstp %st(1)
.L_stack_return:
	fstpl	24(%rsp)		/* result to memory */
	movsd	24(%rsp), %xmm0	/* result to return register */
	shlq    $63, %r8                /* Test odd bit */
        movd    %r8, %xmm1
	orpd  	%xmm1, %xmm0		/* multipy by -1.0 */

.L_Dpop_and_return:
	movq %rbp,%rsp
	popq %rbp
	ret
/* ---------------------------------------------------------------------- */
/* This part tests the special cases, and jumps appropriately */
.L_Special_Pow_Cases_1:
        /* if x == 1.0, return 1.0 */
        cmpq    .C4_D101(%rip), %rax
        je      .L__DSpecial_Case_1

        /* if y == 1.5, return x * sqrt(x) */
        cmpq    .C4_D104(%rip), %rcx
        je      .L__DSpecial_Case_2

        /* if y == 0.5, return sqrt(x) */
        cmpq    .C4_D102(%rip), %rcx
        je      .L__DSpecial_Case_3

        /* if y == 0.25, return sqrt(sqrt(x)) */
        cmpq    .C4_D105(%rip), %rcx
        je      .L__DSpecial_Case_4

.L_Special_Pow_Cases_2:
        /* if abs(y) == 0, return 1.0 */
        testq   .C4_D107(%rip), %rcx
        je      .L__DSpecial_Case_5

        /* if x == nan or inf, handle */
        movq    %rax, %rdx
        andq    .C4_D106(%rip), %rdx
        cmpq    .C4_D106(%rip), %rdx
        je      .L__DSpecial_Case_6

.L__DSpecial_Pow_Case_7:
        /* if y == nan or inf, handle */
        movq    %rcx, %rdx
        andq    .C4_D106(%rip), %rdx
        cmpq    .C4_D106(%rip), %rdx
        je      .L__DSpecial_Case_7

.L__DSpecial_Pow_Case_8:
        /* if y == 1.0, return x */
        cmpq    .C4_D101(%rip), %rcx
        je      .L_Dpop_and_return

.L__DSpecial_Pow_Case_9:
        /* If sign of x is 1, jump away */
        testq   .C4_D108(%rip), %rax
        jne     .L__DSpecial_Pow_Case_10
        /* x is 0.0 or +inf */
        testq    %rax, %rax
        jne     .L__DSpecial_Case_9b

.L__DSpecial_Case_9a:
        /* x is 0.0, test sign of y */
        testq   .C4_D108(%rip), %rcx
        cmovneq .C4_D106(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Case_9b:
        /* x is +inf, test sign of y */
        testq   .C4_D108(%rip), %rcx
        cmovneq .C4_D103(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Pow_Case_10:
        /* x is -0.0, neg, or -inf */
        /* Need to compute y is integer, even, odd, etc. */
        /* rax = x, rcx = y, use r8, r9, r10 for scratch */
        /* r8 contains result:  ==0 if not an int */
        /*                      ==1 if an odd  int */
        /*                      ==2 if an even int */
        movq    %rcx, %r8
        movq    %rcx, %r9
        movq    $1075, %r10
        andq    .C4_D106(%rip), %r8
        sarq    $52, %r8
        subq    %r8, %r10       /* 1075 - ((y && 0x7ff) >> 52) */
        jb      .L__DY_inty_2
        cmpq    $53, %r10
        jae     .L__DY_inty_0
        movq    $1, %rdx
        movq    %r10, %rcx
        shlq    %cl, %rdx
        movq    %rdx, %r10
        subq    $1, %rdx
        testq   %r9, %rdx
        jne     .L__DY_inty_0
        testq   %r9, %r10
        jne     .L__DY_inty_1
.L__DY_inty_2:
        movq    $2, %r8
        jmp     .L__DY_inty_decided
.L__DY_inty_1:
        movq    $1, %r8
        jmp     .L__DY_inty_decided
.L__DY_inty_0:
        xorq    %r8, %r8

.L__DY_inty_decided:
        movq    %r9, %rcx
        movq    %rax, %rdx
        andq    .C4_D106(%rip), %rdx
        cmpq    .C4_D106(%rip), %rdx
        je      .L__DSpecial_Case_10c  /* Jump if x is -inf */

.L__DSpecial_Case_10a:
        testq   .C4_D107(%rip), %rax
        jne     .L__DSpecial_Case_10e  /* Jump if x is nonzero */

        /* x is -0.0, test sign of y */
        cmpq    $1, %r8
        je      .L__DSpecial_Case_10b
        xorq    %rax, %rax
        testq   .C4_D108(%rip), %rcx
        cmovneq .C4_D106(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Case_10b:
        testq   .C4_D108(%rip), %rcx
        cmovneq .C4_D10B(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Case_10c:
        /* x is -inf, test sign of y */
        cmpq    $1, %r8
        je      .L__DSpecial_Case_10d
        /* x is -inf, inty != 1 */
        movq    .C4_D106(%rip), %rax
        testq   .C4_D108(%rip), %rcx
        cmovneq .C4_D103(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Case_10d:
        /* x is -inf, inty == 1 */
        testq   .C4_D108(%rip), %rcx
        cmovneq .C4_D108(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Case_10e:
        /* x is negative */
        comisd  .C4_D109(%rip), %xmm2
        ja      .L__DY_is_large
        testq   $3, %r8
        je      .L__DSpecial_Case_10f
        andq    .C4_D107(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_noopt

.L__DSpecial_Case_10f:
        movq    .C4_D10B(%rip), %rax
        orq     .C4_D10C(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

/* ---------------------------------------------------------------------- */
/* This part acts on the special cases, jumps back or returns */

/*
 * 1.0 ** y -> 1.0
 */
.L__DSpecial_Case_1:
.L__DSpecial_Case_5:
	movsd	.C4_D101(%rip), %xmm0
        jmp     .L_Dpop_and_return
/*
 * if (pw == 0x3ff80000) x ** 1.5 -> sqrt(x)*x
 */
.L__DSpecial_Case_2:
	fldl	24(%rsp)
	fld	%st(0)
	fsqrt
	fmulp	%st,%st(1)
        xor     %r8d, %r8d
	jmp	.L_stack_return
/*
 * if (pw == 0x3fe00000) x ** 0.5 -> sqrt(x)
 */
.L__DSpecial_Case_3:
	fldl	24(%rsp)
	fsqrt
        xor     %r8d, %r8d
	jmp	.L_stack_return
/*
 * if (pw == 0x3fd00000) x ** .25 -> sqrt(sqrt(x))
 */
.L__DSpecial_Case_4:
	fldl	24(%rsp)
	fsqrt
	fsqrt
        xor     %r8d, %r8d
	jmp	.L_stack_return

/* if x == nan or inf, handle.  If INF, jump back, if nan, quiet and return */
.L__DSpecial_Case_6:
        testq   .C4_D10D(%rip), %rax
        je      .L__DSpecial_Pow_Case_7
        orq     .C4_D10C(%rip), %rax
        movd    %rax, %xmm0
        jmp     .L_Dpop_and_return

.L__DSpecial_Case_7:
        testq   .C4_D10D(%rip), %rcx
        je      .L__DY_is_large
        orq     .C4_D10C(%rip), %rcx
        movd    %rcx, %xmm0
        jmp     .L_Dpop_and_return

/* This takes care of all the large Y cases */
.L__DY_is_large:
        movsd	.C4_D107(%rip), %xmm2
        comisd  .C4_D103(%rip), %xmm1
        andpd   %xmm2, %xmm0
        jb      .L__DY_large_negative
.L__DY_large_positive:
        /* If abs(x) < 1.0, return 0 */
        /* If abs(x) == 1.0, return 1.0 */
        /* If abs(x) > 1.0, return Inf */
        comisd  .C4_D101(%rip), %xmm0
        jb      .L__DY_large_pos_0
        je      .L__DY_large_pos_1
.L__DY_large_pos_i:
        movsd   .C4_D106(%rip), %xmm0
        jmp     .L_Dpop_and_return
.L__DY_large_pos_1:
        movsd   .C4_D101(%rip), %xmm0
        jmp     .L_Dpop_and_return

.L__DY_large_negative:
        /* If abs(x) < 1.0, return Inf */
        /* If abs(x) == 1.0, return 1.0 */
        /* If abs(x) > 1.0, return 0 */
        comisd  .C4_D101(%rip), %xmm0
        jb      .L__DY_large_pos_i
        je      .L__DY_large_pos_1
.L__DY_large_pos_0:
        movsd   .C4_D103(%rip), %xmm0
        jmp     .L_Dpop_and_return

.L__DY_near_zero:
        movsd   .C4_D101(%rip), %xmm0
        jmp     .L_Dpop_and_return
/* ---------------------------------------------------------------------- */
        ALN_QUAD
.C4_D100:
        .quad   0x0bff0000000000000
.C4_D101:
        .quad   0x03ff0000000000000
.C4_D102:
        .quad   0x03fe0000000000000
.C4_D103:
        .quad   0x00000000000000000
.C4_D104:
        .quad   0x03ff8000000000000
.C4_D105:
        .quad   0x03fd0000000000000
.C4_D106:
        .quad   0x07ff0000000000000
.C4_D107:
        .quad   0x07fffffffffffffff
.C4_D108:
        .quad   0x08000000000000000
.C4_D109:
        .quad   0x043dfffffffffffff
.C4_D10A:
        .quad   0x03c00000000000000
.C4_D10B:
        .quad   0x0fff0000000000000
.C4_D10C:
        .quad   0x00008000000000000
.C4_D10D:
        .quad   0x0000fffffffffffff
.C4_D10E:
        .quad   0x041e0000000000000
.C4_D10F:
        .quad   0x0c1e0000000200000

	ELF_FUNC(__mth_i_dpowd)
	ELF_SIZE(__mth_i_dpowd)
