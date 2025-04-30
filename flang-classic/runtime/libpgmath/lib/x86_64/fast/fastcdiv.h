/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * Note: This "header" file can potentially be included from fastcdiv.S
 * 	 multiple times for instruction sets VEX, FMA4, and FMA3.
 * 	 Certain routines and data structures can only be defined once
 * 	 regardless of how many times this file is included.
 *
 * 	 Thus use the CPP macro "_FASTCDIV_H" to indicate that the file
 * 	 has been referenced once.
 *
 * 	 Unlike standard headers where the typical construct to guard
 * 	 against multiple passes is:
 * 	 #ifndef	_<SOME-HEADER-MACRO_NAME>_H
 * 	 #define	<SOME-HEADER-MACRO_NAME>_H
 * 	 ...
 * 	 #endif		// #ifndef        _<SOME-HEADER-MACRO_NAME>_H
 *
 * 	 We have to #define macro "_FASTCDIV_H" as the last operation
 * 	 in this file.
 */

#if	! defined(_FASTCDIV_H)
	.text
        ALN_QUAD
.Const_z:
        .long   0xffffffff
        .long   0x7fffffff
        .long   0x00000000
        .long   0x3ff00000
        .long   0x00000000
        .long   0x00000000
        .long   0x00000000
        .long   0x80000000

.Const_mask:
        .long   0x7fffffff
        .long   0x3f800000
        .long   0x00000000
        .long   0x80000000
#endif

/*
 * ============================================================
 *
 * The PGI recommended algorithm:
 * __mth_i_cddiv(dcmplx_t *dcmplx, double ar, double ai,
 *               double br, double bi)
 * {
 *  double          r_mag, i_mag;
 *  double          bl, bs, ax, ay, dm;
 *  double          r, f, m, g, n, s, t, d, x, y;
 *
 *  r_mag = br;
 *  if (r_mag < 0)
 *      r_mag = -r_mag;
 *  i_mag = bi;
 *  if (i_mag < 0)
 *      i_mag = -i_mag;
 *
 *  if (r_mag <= i_mag) {
 *      bl = bi;
 *      bs = br;
 *      ax = ar;
 *      ay = ai;
 *      dm = 1.0;
 *  } else {
 *      bl = br;
 *      bs = bi;
 *      ax = ai;
 *      ay = ar;
 *      dm = -1.0;
 *  }
 *  r = bs / bl;
 *  f = ax * r;
 *  m = f + ay;
 *  g = ay * r;
 *  n = g - ax;
 *  s = r * r;
 *  t = 1.0 + s;
 *  d = bl * t;
 *  x = m / d;
 *  dm = dm * d;
 *  y = n / dm;
 * Alternatively (same numerics, different variables):
 *  if (r_mag <= i_mag) {
 *      bl = bi;
 *      bs = br;
 *      axm = ar;
 *      aym = ai;
 *      axp = ai;
 *      ayp = -ar;
 *  } else {
 *      bl = br;
 *      bs = bi;
 *      axm = ai;
 *      aym = -ar;
 *      axp = ar;
 *      ayp = ai;
 *  }
 *  r = bs / bl;
 *  s = r * r;
 *  t = 1.0 + s;
 *  d = t * bl;
 *  rx = r;
 *  ry = r;
 *  sx = rx * axm;
 *  sy = ry * aym;
 *  tx = sx + axp;
 *  ty = sy + ayp;
 *  x = tx / d;
 *  y = ty / d;
 */


#ifndef _FASTCDIV_H
/* ========================================================================= */
	.text
        ALN_FUNC
	.globl ENT(__fsc_div)
ENT(__fsc_div):

	RZ_PUSH
	xorq	%rax, %rax
        movlps  %xmm1, RZ_OFF(16)(%rsp)
	movss	%xmm1, RZ_OFF(8)(%rsp)
	notl	%eax
        movlps  %xmm0, RZ_OFF(24)(%rsp)
        movlps  %xmm0, RZ_OFF(40)(%rsp)
/* 
 *  Set up stack contents like this:
 *  rsp+8
 *  rsp+4
 *  rsp
 *  -8 : br
 *  -12: bi
 *  -16: br
 *  -20: ai   aym
 *  -24: ar   axm
 *  -28: -ar  ayp  aym
 *  -32: ai   ayp  axm
 *  -36: ai        ayp
 *  -40: ar        axp
 *
*/
	shrl	$1, %eax
	movl	RZ_OFF(12)(%rsp), %ecx  /* i_mag = bi */
	movl	RZ_OFF(16)(%rsp), %edx  /* r_mag = br */
	andl	%eax, %ecx              /* if (i_mag < 0) i_mag = -i_mag */
	andl	%eax, %edx              /* if (r_mag < 0) r_mag = -r_mag */
	subl	%ecx, %edx              /* r_mag <= i_mag */
	shrl    $31, %edx
	movss	RZ_OFF(12)(%rsp,%rdx,4),%xmm1  /* bs */
	divss	RZ_OFF(16)(%rsp,%rdx,4),%xmm1  /* r = bs / bl */
                                               /* (dest is numerator) */
	shrl	$24, %eax
	shll	$23, %eax               /* 1.0d0 */
	movl	%eax, RZ_OFF(8)(%rsp)
	movl	$1, %ecx
	shll	$31,%ecx
	xorl	RZ_OFF(24)(%rsp),%ecx
	movl	%ecx,RZ_OFF(28)(%rsp)
	movl	RZ_OFF(36)(%rsp),%ecx
	movl	%ecx,RZ_OFF(32)(%rsp)

	movlps	RZ_OFF(32)(%rsp,%rdx,8),%xmm0  /* axm, aym for multiply */
	movlps	RZ_OFF(40)(%rsp,%rdx,8),%xmm3  /* axp, ayp for add */

	shufps  $0,%xmm1,%xmm1          /* rx = r; ry = r; */
	mulps	%xmm1,%xmm0             /* sx = rx * axm; sy = ry * aym; */
	mulss   %xmm1,%xmm1             /* s = r * r */
	addss	RZ_OFF(8)(%rsp),%xmm1          /* t = 1 + r * r */
	mulss	RZ_OFF(16)(%rsp,%rdx,4), %xmm1 /* d = t * bl */
	shufps  $0,%xmm1,%xmm1
	addps	%xmm3,%xmm0             /* tx = sx + axp; ty = sy + ayp; */
	divps	%xmm1,%xmm0             /* x = tx / d; y = ty / d; */
	RZ_POP
	ret

	ELF_FUNC(__fsc_div)
	ELF_SIZE(__fsc_div)
#endif


#if	! defined(_FASTCDIV_H)
/* ========================================================================= */
/* 
 *  vector single precision complex div
 *
 *  Prototype:
 *
 *      single __fvc_div(float *x, float *y);
 *
 */
	.text
	ALN_FUNC
	.globl ENT(__fvc_div)
ENT(__fvc_div):

	RZ_PUSH
        xorq    %rax, %rax
        movlps  %xmm1, RZ_OFF(16)(%rsp)
        movss   %xmm1, RZ_OFF(8)(%rsp)
        notl    %eax
        movlps  %xmm0, RZ_OFF(24)(%rsp)
        movlps  %xmm0, RZ_OFF(40)(%rsp)
        movhps  %xmm1, RZ_OFF(48)(%rsp)
        movhps  %xmm1, RZ_OFF(56)(%rsp)
        movhps  %xmm0, RZ_OFF(64)(%rsp)
        movhps  %xmm0, RZ_OFF(80)(%rsp)
/*
 *  Set up stack contents like this:
 *  rsp+8
 *  rsp+4
 *  rsp
 *  -8 : br0
 *  -12: bi0
 *  -16: br0
 *  -20: ai0  aym
 *  -24: ar0  axm
 *  -28: -ar0 ayp  aym
 *  -32: ai0  ayp  axm
 *  -36: ai0       ayp
 *  -40: ar0       axp
 *
 *  -48: br1
 *  -52: bi1
 *  -56: br1
 *  -60: ai1  aym
 *  -64: ar1  axm
 *  -68: -ar1 ayp  aym
 *  -72: ai1  ayp  axm
 *  -76: ai1       ayp
 *  -80: ar1       axp
 */
        shrl    $1, %eax
        movl    RZ_OFF(12)(%rsp), %ecx  /* i_mag = bi */
        movl    RZ_OFF(16)(%rsp), %edx  /* r_mag = br */
        andl    %eax, %ecx              /* if (i_mag < 0) i_mag = -i_mag */
        andl    %eax, %edx              /* if (r_mag < 0) r_mag = -r_mag */
        subl    %ecx, %edx              /* r_mag <= i_mag */
        shrl    $31, %edx
        movss   RZ_OFF(12)(%rsp,%rdx,4),%xmm1  /* bs */
        movss   RZ_OFF(16)(%rsp,%rdx,4),%xmm2  /* bl */

        movl    RZ_OFF(52)(%rsp), %r8d  /* i_mag = bi */
        movl    RZ_OFF(56)(%rsp), %r9d  /* r_mag = br */
        andl    %eax, %r8d              /* if (i_mag < 0) i_mag = -i_mag */
        andl    %eax, %r9d              /* if (r_mag < 0) r_mag = -r_mag */
        subl    %r8d, %r9d              /* r_mag <= i_mag */
        shrl    $31, %r9d
        movhps  RZ_OFF(52)(%rsp,%r9,4),%xmm1   /* bs */
        movhps  RZ_OFF(56)(%rsp,%r9,4),%xmm2   /* bl */

        shufps  $160, %xmm1, %xmm1      /* shuffle for rx, ry */
        shufps  $160, %xmm2, %xmm2      /* shuffle for rx, ry */

        divps   %xmm2,%xmm1             /* r = bs / bl */
        shrl    $24, %eax
        shll    $23, %eax               /* 1.0d0 */
        movl    %eax, %ecx              /* 1.0d0 */
        shlq    $32, %rcx
        orq     %rcx, %rax
        movq    %rax, RZ_OFF(48)(%rsp)
        movq    %rax, RZ_OFF(56)(%rsp)
        addl    %edx, %edx              /* Add for rsp+8/rsp in reads */
        addl    %r9d, %r9d              /* Add for rsp+8/rsp in reads */

        movl    $1, %ecx
        shll    $31,%ecx

        xorl    RZ_OFF(24)(%rsp),%ecx
        movl    %ecx,RZ_OFF(28)(%rsp)
        movl    RZ_OFF(36)(%rsp),%ecx
        movl    %ecx,RZ_OFF(32)(%rsp)

        movl    $1, %ecx
        shll    $31,%ecx

        xorl    RZ_OFF(64)(%rsp),%ecx
        movl    %ecx,RZ_OFF(68)(%rsp)
        movl    RZ_OFF(76)(%rsp),%ecx
        movl    %ecx,RZ_OFF(72)(%rsp)

        movlps  RZ_OFF(32)(%rsp,%rdx,4),%xmm0  /* axm, aym for multiply */
        movhps  RZ_OFF(72)(%rsp,%r9,4),%xmm0   /* axm, aym for multiply */

        movlps  RZ_OFF(40)(%rsp,%rdx,4),%xmm3  /* axp, ayp for add */
        movhps  RZ_OFF(80)(%rsp,%r9,4),%xmm3   /* axp, ayp for add */

        mulps   %xmm1,%xmm0             /* sx = rx * axm; sy = ry * aym; */
        mulps   %xmm1,%xmm1             /* s = r * r */
        addps   RZ_OFF(56)(%rsp),%xmm1  /* t = 1 + r * r */
        mulps   %xmm2,%xmm1             /* d = t * bl */
        addps   %xmm3,%xmm0             /* tx = sx + axp; ty = sy + ayp; */
        divps   %xmm1,%xmm0             /* x = tx / d; y = ty / d; */
        RZ_POP
        ret

        ELF_FUNC(__fvc_div)
        ELF_SIZE(__fvc_div)
#endif

/* ========================================================================= */
/* 
 *  vector and scalar single precision complex div - vec and FMA4 versions.
 *
 *  Prototype:
 *
 *      single __fsc_div_[vex|fma4](%xmm0-ss, %xmm1-ss)
 *      single __fvc_div_[vex|fma4](%xmm0-ps, %xmm1-ps)
 *
 *  For purposes of bit-for-bit reproducibility and code maintenance, the
 *  scalar routines simply duplicates its input arguments from single to
 *  a packed vector and then falls through to the vector version.
 *
 */
	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fsc_div_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsc_div_,TARGET_VEX_OR_FMA)):

	vmovlhps     %xmm0,%xmm0,%xmm0
	vmovlhps     %xmm1,%xmm1,%xmm1
//	Fall though to ASM_CONCAT(__fvc_div_,TARGET_VEX_OR_FMA)
//	JMP(ASM_CONCAT(__fvc_div_,TARGET_VEX_OR_FMA))

        ELF_FUNC(ASM_CONCAT(__fsc_div_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsc_div_,TARGET_VEX_OR_FMA))

	.globl ENT(ASM_CONCAT(__fvc_div_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvc_div_,TARGET_VEX_OR_FMA)):

        vbroadcastss .Const_mask(%rip), %xmm2    /* 0x7fffffff */
        vshufps      $0xa0, %xmm1, %xmm1, %xmm3  /* br,br,br,br */
        vshufps      $0xf5, %xmm1, %xmm1, %xmm4  /* bi,bi,bi,bi */
        vshufps      $0xb1, %xmm0, %xmm0, %xmm5  /* Shuffle to pull from a */
        vandps       %xmm2, %xmm3, %xmm3         /* mask off for r_mag,r_mag */
        vandps       %xmm4, %xmm2, %xmm2         /* mask off for i_mag,i_mag */
        vcmpltps     %xmm3, %xmm2, %xmm4         /* r_mag <= i_mag */
        vmovddup     .Const_mask+8(%rip), %xmm2  /* 0x80000000 00000000 */
        vshufps      $0xa0, %xmm1, %xmm1, %xmm3  /* br,br,br,br */
        vxorps       %xmm2, %xmm5, %xmm5         /* ayp, axp */
        vshufps      $0xf5, %xmm1, %xmm1, %xmm2  /* bi,bi,bi,bi */

        vblendvps    %xmm4, %xmm2, %xmm3, %xmm1  /* bs */
        vblendvps    %xmm4, %xmm3, %xmm2, %xmm2  /* bl */
        vblendvps    %xmm4, %xmm5, %xmm0, %xmm3  /* aym, axm */
        vblendvps    %xmm4, %xmm0, %xmm5, %xmm4  /* ayp, axp */
        vbroadcastss .Const_mask+4(%rip), %xmm0  /* 0x3f800000 */

        vdivps       %xmm2, %xmm1, %xmm1         /* r */
#ifdef TARGET_FMA
#        vfmaddps     %xmm4, %xmm3, %xmm1, %xmm3
	VFMA_213PS	(%xmm4,%xmm1,%xmm3)
#        vfmaddps     %xmm0, %xmm1, %xmm1, %xmm0  /* t = r*r+1.0 */
	VFMA_231PS	(%xmm1,%xmm1,%xmm0)
#else
        vmulps       %xmm1, %xmm3, %xmm3         /* sy, sx */
        vaddps       %xmm4, %xmm3, %xmm3         /* ty, tx */
        vmulps       %xmm1, %xmm1, %xmm1         /* s = r * r */
        vaddps       %xmm0, %xmm1, %xmm0         /* t = 1.0 + s */
#endif
        vmulps       %xmm0, %xmm2, %xmm1         /* d = bl * t */
        vdivps       %xmm1, %xmm3, %xmm0         /* y, x */
        ret

        ELF_FUNC(ASM_CONCAT(__fvc_div_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvc_div_,TARGET_VEX_OR_FMA))

/* ------------------------------------------------------------------------- */
/* 
 *  vector single precision complex div
 *
 *  Prototype:
 *
 *      single __fvc_div_[vex|fma4]_256(float *x, float *y);
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvc_div_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvc_div_,TARGET_VEX_OR_FMA,_256)):

        vbroadcastss .Const_mask(%rip), %ymm2    /* 0x7fffffff */
        vshufps      $0xa0, %ymm1, %ymm1, %ymm3  /* br,br,br,br */
        vshufps      $0xf5, %ymm1, %ymm1, %ymm4  /* bi,bi,bi,bi */
        vshufps      $0xb1, %ymm0, %ymm0, %ymm5  /* Shuffle to pull from a */
        vandps       %ymm2, %ymm3, %ymm3         /* mask off for r_mag, r_mag */
        vandps       %ymm4, %ymm2, %ymm2         /* mask off for i_mag, i_mag */
        vcmpltps     %ymm3, %ymm2, %ymm4         /* r_mag <= i_mag */
        vbroadcastsd .Const_mask+8(%rip), %ymm2  /* 0x80000000 00000000 */
        vshufps      $0xa0, %ymm1, %ymm1, %ymm3  /* br,br,br,br */
        vxorps       %ymm2, %ymm5, %ymm5         /* ayp, axp */
        vshufps      $0xf5, %ymm1, %ymm1, %ymm2  /* bi,bi,bi,bi */

        vblendvps    %ymm4, %ymm2, %ymm3, %ymm1  /* bs */
        vblendvps    %ymm4, %ymm3, %ymm2, %ymm2  /* bl */
        vblendvps    %ymm4, %ymm5, %ymm0, %ymm3  /* aym, axm */
        vblendvps    %ymm4, %ymm0, %ymm5, %ymm4  /* ayp, axp */
        vbroadcastss .Const_mask+4(%rip), %ymm0  /* 0x3f800000 */

        vdivps       %ymm2, %ymm1, %ymm1         /* r */
#ifdef TARGET_FMA
#        vfmaddps     %ymm4, %ymm3, %ymm1, %ymm3
	VFMA_213PS	(%ymm4,%ymm1,%ymm3)
#        vfmaddps     %ymm0, %ymm1, %ymm1, %ymm0  /* t = r*r+1.0 */
	VFMA_231PS	(%ymm1,%ymm1,%ymm0)
#else
        vmulps       %ymm1, %ymm3, %ymm3         /* sy, sx */
        vaddps       %ymm4, %ymm3, %ymm3         /* ty, tx */
        vmulps       %ymm1, %ymm1, %ymm1         /* s = r * r */
        vaddps       %ymm0, %ymm1, %ymm0         /* t = 1.0 + s */
#endif
        vmulps       %ymm0, %ymm2, %ymm1         /* d = bl * t */
        vdivps       %ymm1, %ymm3, %ymm0         /* y, x */
        ret

        ELF_FUNC(ASM_CONCAT3(__fvc_div_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvc_div_,TARGET_VEX_OR_FMA,_256))

#if	! defined(_FASTCDIV_H)
/* ========================================================================= */
/* 
 *  Double precision complex div (scalar) using vector instructions
 *
 *  C99 calling sequence:
 *      complex double __fvz_div_c99(double complex x, double complex y)
 *	(%xmm0) = creal(x)
 *	(%xmm1) = cimag(x)
 *	(%xmm2) = creal(y)
 *	(%xmm3) = cimag(y)
 */

	.text
	ALN_FUNC
	.globl	ENT(__fsz_div_c99)
ENT(__fsz_div_c99):
	subq	$8,%rsp		/* Align stack to 16 bytes */
/*
 *	Pack upper(%xmm0) = (%xmm1)
 *	     lower(%xmm1) = (%xmm2)
 *	     upper(%xmm1) = (%xmm3)
 */
	movlhps %xmm1,%xmm0
	movlhps %xmm3,%xmm2
	movapd	%xmm2,%xmm1

	CALL(ENT(__fsz_div))
	movhlps %xmm0,%xmm1	/* Unpack real+imag */
	addq	$8,%rsp
	ret

        ELF_FUNC(__fsz_div_c99)
        ELF_SIZE(__fsz_div_c99)

/* ========================================================================= */
	.text
        ALN_FUNC
	.globl ENT(__fsz_div)
	.globl ENT(__fvz_div)
ENT(__fsz_div):
ENT(__fvz_div):

	RZ_PUSH
	xorq	%rax, %rax
        movapd  %xmm1, RZ_OFF(24)(%rsp)
	movsd	%xmm1, RZ_OFF(8)(%rsp)
	notq	%rax
        movapd  %xmm0, RZ_OFF(40)(%rsp)
        movapd  %xmm0, RZ_OFF(72)(%rsp)
/* 
 *  Set up stack contents like this:
 *  rsp+16
 *  rsp+8
 *  rsp
 *  -8 : br
 *  -16: bi
 *  -24: br
 *  -32: ai   aym
 *  -40: ar   axm
 *  -48: -ar  ayp  aym
 *  -56: ai   ayp  axm
 *  -64: ai        ayp
 *  -72: ar        axp
 *
*/
	shrq	$1, %rax
	movq	RZ_OFF(16)(%rsp), %rcx  /* i_mag = bi */
	movq	RZ_OFF(24)(%rsp), %rdx  /* r_mag = br */
	andq	%rax, %rcx              /* if (i_mag < 0) i_mag = -i_mag */
	andq	%rax, %rdx              /* if (r_mag < 0) r_mag = -r_mag */
	subq	%rcx, %rdx              /* r_mag <= i_mag */
	shrq    $63, %rdx
	movsd	RZ_OFF(16)(%rsp,%rdx,8),%xmm1  /* bs */
	divsd	RZ_OFF(24)(%rsp,%rdx,8),%xmm1  /* r = bs / bl */
	shrq	$53, %rax
	shlq	$52, %rax               /* 1.0d0 */
	movq	%rax, RZ_OFF(8)(%rsp)
	addq	%rdx, %rdx              /* Add for rsp+16/rsp in reads */
	movq	$1, %rcx
	shlq	$63,%rcx
	xorq	RZ_OFF(40)(%rsp),%rcx
	movq	%rcx,RZ_OFF(48)(%rsp)
	movq	RZ_OFF(64)(%rsp),%rcx
	movq	%rcx,RZ_OFF(56)(%rsp)
	movapd	RZ_OFF(56)(%rsp,%rdx,8),%xmm0  /* axm, aym for multiply */
	movapd	RZ_OFF(72)(%rsp,%rdx,8),%xmm3  /* axp, ayp for add */
	movapd  %xmm1,%xmm2
	shufpd  $0,%xmm1,%xmm1          /* rx = r; ry = r; */
	mulpd	%xmm1,%xmm0             /* sx = rx * axm; sy = ry * aym; */
	mulsd   %xmm2,%xmm2             /* s = r * r */
	addsd	RZ_OFF(8)(%rsp),%xmm2          /* t = 1 + r * r */
	mulsd	RZ_OFF(24)(%rsp,%rdx,4), %xmm2 /* d = t * bl */
	shufpd  $0,%xmm2,%xmm2
	addpd	%xmm3,%xmm0             /* tx = sx + axp; ty = sy + ayp; */
	divpd	%xmm2,%xmm0             /* x = tx / d; y = ty / d; */
	RZ_POP
	ret

	ELF_FUNC(__fvz_div)
	ELF_SIZE(__fvz_div)
	ELF_FUNC(__fsz_div)
	ELF_SIZE(__fsz_div)
#endif

/* ========================================================================= */
/* 
 *  Double precision complex div (scalar) using vector instructions
 *
 *  C99 calling sequence:
 *      complex double __fvz_div_vex_c99(double complex x, double complex y)
 *      complex double __fvz_div_fma4_c99(double complex x, double complex y)
 *	(%xmm0) = creal(x)
 *	(%xmm1) = cimag(x)
 *	(%xmm2) = creal(y)
 *	(%xmm3) = cimag(y)
 */

	.text
	ALN_FUNC
	.globl	ENT(ASM_CONCAT3(__fsz_div_,TARGET_VEX_OR_FMA,_c99))
ENT(ASM_CONCAT3(__fsz_div_,TARGET_VEX_OR_FMA,_c99)):

/*
 *	Pack upper(%xmm0) = (%xmm1)
 *	     lower(%xmm1) = (%xmm2)
 *	     upper(%xmm1) = (%xmm3)
 */
	vmovlhps %xmm1,%xmm0,%xmm0
	vmovlhps %xmm3,%xmm2,%xmm1

	CALL(ENT(ASM_CONCAT(__fsz_div_,TARGET_VEX_OR_FMA)))
	vmovhlps %xmm0,%xmm1,%xmm1	/* Unpack real+imag */
	ret

        ELF_FUNC(ASM_CONCAT3(__fsz_div_,TARGET_VEX_OR_FMA,_c99))
        ELF_SIZE(ASM_CONCAT3(__fsz_div_,TARGET_VEX_OR_FMA,_c99))

/* ========================================================================= */
/* 
 *  Double precision complex div (scalar) using vector instructions
 *
 *  Prototype:
 *      complex double __fvz_div_vex(%xmm0[real+imag], %xmm1[real+imag])
 *      complex double __fvz_div_fma4(%xmm0[real+imag], %xmm1[real+imag])
 */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fsz_div_,TARGET_VEX_OR_FMA))
	.globl ENT(ASM_CONCAT(__fvz_div_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsz_div_,TARGET_VEX_OR_FMA)):
ENT(ASM_CONCAT(__fvz_div_,TARGET_VEX_OR_FMA)):

	vmovddup     .Const_z(%rip), %xmm2       /* 0x7fffffff 0xffffffff */
	vshufpd      $0, %xmm1, %xmm1, %xmm3     /* br, br */
	vshufpd      $3, %xmm1, %xmm1, %xmm4     /* bi, bi */
	vshufpd      $1, %xmm0, %xmm0, %xmm5     /* Shuffle to pull from a */
	vandpd       %xmm2, %xmm3, %xmm3         /* mask off for r_mag, r_mag */
	vandpd       %xmm4, %xmm2, %xmm2         /* mask off for i_mag, i_mag */
	vxorpd       .Const_z+16(%rip), %xmm5, %xmm5  /* axp ayp */
	vcmpltpd     %xmm3, %xmm2, %xmm4         /* r_mag <= i_mag */

	vshufpd      $0, %xmm1, %xmm1, %xmm3     /* br, br */
	vshufpd      $3, %xmm1, %xmm1, %xmm2     /* bi, bi */
	vblendvpd    %xmm4, %xmm2, %xmm3, %xmm1  /* bs */
	vblendvpd    %xmm4, %xmm3, %xmm2, %xmm2  /* bl  #1free */
	vblendvpd    %xmm4, %xmm5, %xmm0, %xmm3  /* aym, axm */
	vblendvpd    %xmm4, %xmm0, %xmm5, %xmm4  /* ayp, axp */
	vmovddup     .Const_z+8(%rip), %xmm0     /* 0x3ff00000 0x00000000 */

	vdivpd       %xmm2, %xmm1, %xmm1         /* r */
#ifdef TARGET_FMA
#	vfmaddpd     %xmm4, %xmm3, %xmm1, %xmm3
	VFMA_213PD	(%xmm4,%xmm1,%xmm3)
#	vfmaddpd     %xmm0, %xmm1, %xmm1, %xmm0
	VFMA_231PD	(%xmm1,%xmm1,%xmm0)
#else
	vmulpd       %xmm1, %xmm3, %xmm3         /* sy,sx = ry*aym, rx*axm */
	vaddpd       %xmm4, %xmm3, %xmm3         /* ty, tx */
	vmulpd       %xmm1, %xmm1, %xmm1         /* s = r * r */
	vaddpd       %xmm0, %xmm1, %xmm0         /* t = 1.0 + s */
#endif
	vmulpd       %xmm0, %xmm2, %xmm1         /* d = bl * t */
	vdivpd       %xmm1, %xmm3, %xmm0         /* y, x */
        ret

        ELF_FUNC(ASM_CONCAT(__fvz_div_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvz_div_,TARGET_VEX_OR_FMA))
        ELF_FUNC(ASM_CONCAT(__fsz_div_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsz_div_,TARGET_VEX_OR_FMA))

/* ========================================================================= */
/* 
 *  vector double precision div
 *  Prototype:
 *      single __fvz_div_vex/fma4_256(double *x, double *y);
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvz_div_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvz_div_,TARGET_VEX_OR_FMA,_256)):

        vbroadcastsd .Const_z(%rip), %ymm2       /* 0x7fffffff 0xffffffff */
        vshufpd      $0, %ymm1, %ymm1, %ymm3     /* br, br */
        vshufpd      $15, %ymm1, %ymm1, %ymm4     /* bi, bi */
        vshufpd      $5, %ymm0, %ymm0, %ymm5     /* Shuffle to pull from a */
        vandpd       %ymm2, %ymm3, %ymm3         /* mask off for r_mag, r_mag */
        vandpd       %ymm4, %ymm2, %ymm2         /* mask off for i_mag, i_mag */
        vcmpltpd     %ymm3, %ymm2, %ymm4         /* r_mag <= i_mag */
        vbroadcastf128 .Const_z+16(%rip), %ymm2  /* 0x80000000 0x00000000 */
        vshufpd      $0, %ymm1, %ymm1, %ymm3     /* br, br */
	vxorpd       %ymm2, %ymm5, %ymm5         /* ayp, axp */
        vshufpd      $15, %ymm1, %ymm1, %ymm2     /* bi, bi */

        vblendvpd    %ymm4, %ymm2, %ymm3, %ymm1  /* bs */
        vblendvpd    %ymm4, %ymm3, %ymm2, %ymm2  /* bl */
        vblendvpd    %ymm4, %ymm5, %ymm0, %ymm3  /* aym, axm */
        vblendvpd    %ymm4, %ymm0, %ymm5, %ymm4  /* ayp, axp */
	vbroadcastsd .Const_z+8(%rip), %ymm0     /* 1.0d0 */

        vdivpd       %ymm2, %ymm1, %ymm1         /* r */
#ifdef TARGET_FMA
#        vfmaddpd     %ymm4, %ymm3, %ymm1, %ymm3
	VFMA_213PD	(%ymm4,%ymm1,%ymm3)
#        vfmaddpd     %ymm0, %ymm1, %ymm1, %ymm0  /* t = r*r+1.0 */	
	VFMA_231PD	(%ymm1,%ymm1,%ymm0)
#else
        vmulpd       %ymm1, %ymm3, %ymm3         /* sy, sx */
        vaddpd       %ymm4, %ymm3, %ymm3         /* ty, tx */
        vmulpd       %ymm1, %ymm1, %ymm1         /* s = r * r */
        vaddpd       %ymm0, %ymm1, %ymm0         /* t = 1.0 + s */
#endif
        vmulpd       %ymm0, %ymm2, %ymm1         /* d = bl * t */
        vdivpd       %ymm1, %ymm3, %ymm0         /* y, x */
        ret

        ELF_FUNC(ASM_CONCAT3(__fvz_div_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvz_div_,TARGET_VEX_OR_FMA,_256))


/*
 * Note! (again)
 *
 * This "header" file can be included a couple of times from fastcdiv.S.
 *
 * In the case were it is included more than once, only assemble the
 * AVX-512 single and double precision complex divide routines a single time,
 * since there is no corresponding AMD FMA4 instructions for AVX512.
 */

#if	defined(TARGET_FMA)
#if	VFMA_IS_FMA3 == 1

/* 
 *  vector single precision complex div
 *
 *  Prototype:
 *
 *      single __fvc_div_evex_512(float complex %zmm0, float complex %zmm1)
 *
 *  Note: use of AVX512F instructions only.
 *
 */

        .text
        ALN_FUNC
        .globl ENT(__fvc_div_evex_512)
ENT(__fvc_div_evex_512):

#if	! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
        vbroadcastss .Const_mask(%rip), %zmm2    /* 0x7fffffff */
        vshufps      $0xa0, %zmm1, %zmm1, %zmm3  /* br,br,br,br */
        vshufps      $0xf5, %zmm1, %zmm1, %zmm4  /* bi,bi,bi,bi */
        vshufps      $0xb1, %zmm0, %zmm0, %zmm5  /* Shuffle to pull from a */
        vpandd       %zmm2, %zmm3, %zmm3         /* mask off for r_mag, r_mag */
        vpandd       %zmm4, %zmm2, %zmm2         /* mask off for i_mag, i_mag */
        vcmpltps     %zmm3, %zmm2, %k1           /* r_mag <= i_mag */
        vbroadcastsd .Const_mask+8(%rip), %zmm2  /* 0x80000000 00000000 */
        vshufps      $0xa0, %zmm1, %zmm1, %zmm3  /* br,br,br,br */
        vpxord       %zmm2, %zmm5, %zmm5         /* ayp, axp */
        vshufps      $0xf5, %zmm1, %zmm1, %zmm2  /* bi,bi,bi,bi */

        vblendmps    %zmm2, %zmm3, %zmm1{%k1}    /* bs */
        vblendmps    %zmm3, %zmm2, %zmm2{%k1}    /* bl */
        vblendmps    %zmm5, %zmm0, %zmm3{%k1}    /* aym, axm */
        vblendmps    %zmm0, %zmm5, %zmm4{%k1}    /* ayp, axp */
        vbroadcastss .Const_mask+4(%rip), %zmm0  /* 0x3f800000 */

        vdivps       %zmm2, %zmm1, %zmm1         /* r */

#	vfmadd213ps	%zmm4,%zmm1,%zmm3
#	vfmadd231ps	%zmm1,%zmm1,%zmm0
	VFMA_213PS	(%zmm4,%zmm1,%zmm3)
	VFMA_231PS	(%zmm1,%zmm1,%zmm0)

        vmulps       %zmm0, %zmm2, %zmm1         /* d = bl * t */
        vdivps       %zmm1, %zmm3, %zmm0         /* y, x */
        ret
#else		// ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
	ud2	// No support for OSX & Windows yet - but entry points are needed
#endif		// ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)

        ELF_FUNC(__fvc_div_evex_512)
        ELF_SIZE(__fvc_div_evex_512)

/* ========================================================================= */
/* 
 *  vector double precision div
 *  Prototype:
 *      single __fvz_div_evex_512(double complex %zmm0, double complex %zmm1)
 *
 *  Note: use of AVX512F instructions only.
 */

        .text
        ALN_FUNC
        .globl ENT(__fvz_div_evex_512)
ENT(__fvz_div_evex_512):

#if	! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
        vbroadcastsd .Const_z(%rip), %zmm2      /* 0x7fffffff 0xffffffff */
        vshufpd      $0x00, %zmm1, %zmm1, %zmm3 /* br, br */
        vshufpd      $0xff, %zmm1, %zmm1, %zmm4 /* bi, bi */
        vshufpd      $0x55, %zmm0, %zmm0, %zmm5 /* Shuffle to pull from a */
        vpandq       %zmm2, %zmm3, %zmm3        /* mask off for r_mag, r_mag */
        vpandq       %zmm4, %zmm2, %zmm2        /* mask off for i_mag, i_mag */
        vcmpltpd     %zmm3, %zmm2, %k1          /* r_mag <= i_mag */
        vbroadcastf32x4 .Const_z+16(%rip), %zmm2/* 0x80000000 0x00000000 */
        vshufpd      $0, %zmm1, %zmm1, %zmm3    /* br, br */
        vpxorq       %zmm2, %zmm5, %zmm5        /* ayp, axp */
        vshufpd      $0xff, %zmm1, %zmm1, %zmm2 /* bi, bi */

        vblendmpd    %zmm2, %zmm3, %zmm1{%k1}   /* bs */
        vblendmpd    %zmm3, %zmm2, %zmm2{%k1}   /* bl */
        vblendmpd    %zmm5, %zmm0, %zmm3{%k1}   /* aym, axm */
        vblendmpd    %zmm0, %zmm5, %zmm4{%k1}   /* ayp, axp */
        vbroadcastsd .Const_z+8(%rip), %zmm0    /* 1.0d0 */

        vdivpd       %zmm2, %zmm1, %zmm1        /* r */

#	vfmadd213pd	%zmm4,%zmm1,%zmm3
#	vfmadd231pd	%zmm1,%zmm1,%zmm0
	VFMA_213PD	(%zmm4,%zmm1,%zmm3)
	VFMA_231PD	(%zmm1,%zmm1,%zmm0)

        vmulpd       %zmm0, %zmm2, %zmm1        /* d = bl * t */
        vdivpd       %zmm1, %zmm3, %zmm0        /* y, x */
        ret

#else		// ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
	ud2	// No support for OSX & Windows yet - but entry points are needed
#endif		// ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)

        ELF_FUNC(__fvz_div_evex_512)
        ELF_SIZE(__fvz_div_evex_512)
#endif	// if	VFMA_IS_FMA3 == 1
#endif	// if	VFMA_IS_FMA3


#define	_FASTCDIV_H	1
