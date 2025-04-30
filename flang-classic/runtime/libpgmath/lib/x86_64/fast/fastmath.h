/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* ============================================================
 * Copyright (c) 2004 Advanced Micro Devices, Inc.
 * 
 * All rights reserved.
 * 
 * Redistribution and  use in source and binary  forms, with or
 * without  modification,  are   permitted  provided  that  the
 * following conditions are met:
 * 
 *  Redistributions  of source  code  must  retain  the  above
 *   copyright  notice,  this   list  of   conditions  and  the
 *   following disclaimer.
 * 
 *  Redistributions  in binary  form must reproduce  the above
 *   copyright  notice,  this   list  of   conditions  and  the
 *   following  disclaimer in  the  documentation and/or  other
 *   materials provided with the distribution.
 * 
 *  Neither the  name of Advanced Micro Devices,  Inc. nor the
 *   names  of  its contributors  may  be  used  to endorse  or
 *   promote  products  derived   from  this  software  without
 *   specific prior written permission.
 * 
 * THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
 * CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
 * MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
 * DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
 * INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
 * INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
 * INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
 * GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
 * BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
 * LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
 * OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * It is  licensee's responsibility  to comply with  any export
 * regulations applicable in licensee's jurisdiction.
 * 
 * ============================================================
 */

#if	defined(TARGET_INTERIX_X8664)
#error	TARGET_INTERIX_X8664 is no longer supported
#endif

/*
 *	Some of AMD's recommendation for SSE128 optimizations:
 *	1) use movupd or movddup for movlpd/movhpd pairs
 *	2) use movsd-MEM instead of movlpd-MEM
 *	   (movsd-MEM will zero upper 64-bits)
 *	3) use movapd instead of movsd-REG
 */

#undef	movsdRR
#undef	movssRR
#undef	movlpdMR
#if	defined(GH_TARGET)
#define	movsdRR	movapd		// Register to register
#define	movssRR	movaps		// Register to register
#define	movlpdMR	movsd	// Memory to register
#else
#define	movsdRR	movsd		// Register to register
#define	movssRR	movss		// Register to register
#define	movlpdMR	movlpd	// Memory to register
#endif

/*
 *	Macros to manage labels when GH_TARGET is and
 *	is not specified.
 *
 *	ENT_GH(name) is used when a label has the suffix "_gh"
 *	when GH_TARGET is defined and without the suffix "_gh"
 *	when GH_TARGET is not defined.
 *
 *	IF_GH(...) is used when a statement is only to be
 *	included/assembled when GH_TARGET is defined.
 *
 *	The two macros are primarily used in function entry and
 *	ELF definitions.
 *	Given an original entry and ELF specification for a function as:
 *
 *	#ifdef GH_TARGET
 *		.globl ENT(__fss_exp)
 *		.globl ENT(__fmth_i_exp_gh)
 *	ENT(__fss_exp):
 *	ENT(__fmth_i_exp_gh):
 *	#else
 *		.globl ENT(__fmth_i_exp)
 *	ENT(__fmth_i_exp):
 *	#endif
 *
 *	Is now replaced with the more manageable sequence
 *		IF_GH(.globl ENT(__fss_exp))
 *		.globl	ENT_GH(__fmth_i_exp)
 *	IF_GH(ENT(__fss_exp):)
 *	ENT_GH(__fmth_i_exp):
 *
 *	That is the there is always one of __fmth_i_exp/__fmth_i_exp_gh
 *	defined and optionally if GH_TARGET is defined __fss_exp.
 *
 *	Similarily ELF information was:
 *	
 *	#ifdef GH_TARGET
 *	        ELF_FUNC(__fmth_i_exp_gh)
 *	        ELF_SIZE(__fmth_i_exp_gh)
 *		ELF_FUNC(__fss_exp)
 *		ELF_SIZE(__fss_exp)
 *	#else
 *		ELF_FUNC(__fmth_i_exp)
 *		ELF_SIZE(__fmth_i_exp)
 *	#endif
 *
 *	And is:
 *	        ELF_FUNC(ENT_GH(__fmth_i_exp))
 *	        ELF_SIZE(ENT_GH(__fmth_i_exp))
 *	        IF_GH(ELF_FUNC(__fss_exp))
 *	        IF_GH(ELF_SIZE(__fss_exp))
 *
 */

#undef	ENT_GH
#undef	IF_GH
#ifdef	GH_TARGET
#define	ENT_GH(name)	ENT(name##_gh)
#define	IF_GH(...)	__VA_ARGS__
#else
#define	ENT_GH(name)	ENT(name)
#define	IF_GH(...)
#endif

/*
 * Assume that if _SX0 is defined the whole block is defined.
 */
#ifndef	_SX0
#define _SX0 0
#define _SX1 4
#define _SX2 8
#define _SX3 12

#define _SY0 16
#define _SY1 20
#define _SY2 24
#define _SY3 28

#define _SR0 32
#define _SR1 36
#define _SR2 40
#define _SR3 44
#endif		// #ifndef _SX0

#ifndef LNUM
#define LNUM 001
#endif
#undef	NNN
#define NNN LNUM


#ifdef TABLE_TARGET 

        ALN_QUAD

/* ============================================================
 *
 *     Constants for exponential functions
 *
 * ============================================================
 */

.L__real_3fe0000000000000:	.quad 0x03fe0000000000000	/* 1/2 */
				.quad 0x03fe0000000000000
.L__real_infinity:		.quad 0x07ff0000000000000	/* +inf */
				.quad 0x00008000000000000
.L__real_ninfinity:		.quad 0x0fff0000000000000	/* -inf */
				.quad 0x0fff8000000000000
.L__real_thirtytwo_by_log2: 	.quad 0x040471547652b82fe	/* thirtytwo_by_log2 */
				.quad 0x040471547652b82fe
.L__real_log2_by_32_lead:	.quad 0x03f962e42fe000000	/* log2_by_32_lead */
				.quad 0x03f962e42fe000000
.L__real_log2_by_32_tail:	.quad 0x0Bdcf473de6af278e	/* -log2_by_32_tail */
				.quad 0x0Bdcf473de6af278e
.L__real_log2_by_32:		.quad 0x03f962e42fefa39ef	/* log2_by_32 */
				.quad 0x03f962e42fefa39ef
.L__real_3f56c1728d739765:	.quad 0x03f56c1728d739765	/* 1.38889490863777199667e-03 */
				.quad 0x03f56c1728d739765
.L__real_3F811115B7AA905E:	.quad 0x03F811115B7AA905E	/* 8.33336798434219616221e-03 */
				.quad 0x03F811115B7AA905E
.L__real_3FA5555555545D4E:	.quad 0x03FA5555555545D4E	/* 4.16666666662260795726e-02 */
				.quad 0x03FA5555555545D4E
.L__real_3FC5555555548F7C:	.quad 0x03FC5555555548F7C	/* 1.66666666665260878863e-01 */
				.quad 0x03FC5555555548F7C
.L__real_ln_max_doubleval:	.quad 0x040862e42fefa39ef	/* 709.... */
				.quad 0x040862e42fefa39ef
.L__real_ln_min_doubleval:	.quad 0x0c0874910d52d3052	/* 709.... */
				.quad 0x0c0874910d52d3052
.L_sinh_max_doubleval:          .quad 0x0408633CE8FB9F87E       /*... 710.475860073944 ...*/
                                .quad 0x0408633CE8FB9F87E
.L_sinh_min_doubleval:          .quad 0x0C08633CE8FB9F87E       /*... -710.475860073944 ...*/
                                .quad 0x0C08633CE8FB9F87E
.L__real_mask_unsign:         	.quad 0x07fffffffffffffff	/* Mask for unsigned */
				.quad 0x07fffffffffffffff
.L__ps_mask_unsign:             .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
.L__sp_ln_max_singleval:	.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
.L_sp_sinh_max_singleval:       .long 0x42B2D4FC                /*... 89.41598629223294 ...*/
                                .long 0x42B2D4FC
                                .long 0x42B2D4FC
                                .long 0x42B2D4FC
.L__two_to_jby32_table:
        .quad   0x03FF0000000000000     /* 1.0000000000000000 */
        .quad   0x03FF059B0D3158574     /* 1.0218971486541166 */
        .quad   0x03FF0B5586CF9890F     /* 1.0442737824274138 */
        .quad   0x03FF11301D0125B51     /* 1.0671404006768237 */
        .quad   0x03FF172B83C7D517B     /* 1.0905077326652577 */
        .quad   0x03FF1D4873168B9AA     /* 1.1143867425958924 */
        .quad   0x03FF2387A6E756238     /* 1.1387886347566916 */
        .quad   0x03FF29E9DF51FDEE1     /* 1.1637248587775775 */
        .quad   0x03FF306FE0A31B715     /* 1.1892071150027210 */
        .quad   0x03FF371A7373AA9CB     /* 1.2152473599804690 */
        .quad   0x03FF3DEA64C123422     /* 1.2418578120734840 */
        .quad   0x03FF44E086061892D     /* 1.2690509571917332 */
        .quad   0x03FF4BFDAD5362A27     /* 1.2968395546510096 */
        .quad   0x03FF5342B569D4F82     /* 1.3252366431597413 */
        .quad   0x03FF5AB07DD485429     /* 1.3542555469368927 */
        .quad   0x03FF6247EB03A5585     /* 1.3839098819638320 */
        .quad   0x03FF6A09E667F3BCD     /* 1.4142135623730951 */
        .quad   0x03FF71F75E8EC5F74     /* 1.4451808069770467 */
        .quad   0x03FF7A11473EB0187     /* 1.4768261459394993 */
        .quad   0x03FF82589994CCE13     /* 1.5091644275934228 */
        .quad   0x03FF8ACE5422AA0DB     /* 1.5422108254079407 */
        .quad   0x03FF93737B0CDC5E5     /* 1.5759808451078865 */
        .quad   0x03FF9C49182A3F090     /* 1.6104903319492543 */
        .quad   0x03FFA5503B23E255D     /* 1.6457554781539649 */
        .quad   0x03FFAE89F995AD3AD     /* 1.6817928305074290 */
        .quad   0x03FFB7F76F2FB5E47     /* 1.7186192981224779 */
        .quad   0x03FFC199BDD85529C     /* 1.7562521603732995 */
        .quad   0x03FFCB720DCEF9069     /* 1.7947090750031072 */
        .quad   0x03FFD5818DCFBA487     /* 1.8340080864093424 */
        .quad   0x03FFDFC97337B9B5F     /* 1.8741676341103000 */
        .quad   0x03FFEA4AFA2A490DA     /* 1.9152065613971474 */
        .quad   0x03FFF50765B6E4540     /* 1.9571441241754002 */

.L__dp_max_singleval:		.quad 0x040562e42e0000000
                      		.quad 0x040562e42e0000000
.L__dp_min_singleval:		.quad 0x0c059fe36a0000000

.L_sp_real_infinity:		.long 0x7f800000	/* +inf */
.L_sp_real_ninfinity:		.long 0xff800000	/* -inf */
.L_sp_real_nanfinity:		.long 0xffc00000	/* -inf */
.L_real_min_singleval:		.long 0xc2cff1b5
.L_sp_sinh_min_singleval:       .long 0xC2B2D4FC	/*... -89.41598629223294 ...*/
.L_real_cvt_nan:      		.long 0x00400000

        ALN_QUAD
.L__dble_cdexp_by_pi:   	.quad 0x040145f306dc9c883      /* 16.0 / pi */
.L__dble_cdexp_log2: 		.quad 0x040471547652b82fe      /* thirtytwo_by_log2 */
.L__real_ln_max_doubleval1:     .quad 0x040862e42fefa39ef       /* 709.... */
                                .quad 0x040862e42fefa39ef
.L__real_ln_min_doubleval1:	.quad 0x0c0874910d52d3052	/* 709.... */
				.quad 0x0c0874910d52d3052
.L__cdexp_log2_by_32_lead:	.quad 0x03f962e42fe000000 /* log2_by_32_lead */
.L__cdexp_log2_by_32_tail:	.quad 0x03dcf473de6af278e /* log2_by_32_tail */


/* ============================================================
 *
 *     Constants for logarithm functions, single precision
 *
 * ============================================================
 */

	ALN_QUAD
.L4_380:
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
        .long   0x007fffff      /* mantissa mask */
.L4_381:
	.long	0x003504f3	/* mantissa Bits for 1 / sqrt(2.0) */
	.long	0x003504f3	/* mantissa Bits for 1 / sqrt(2.0) */
	.long	0x003504f3	/* mantissa Bits for 1 / sqrt(2.0) */
	.long	0x003504f3	/* mantissa Bits for 1 / sqrt(2.0) */
.L4_382:
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
	.long	0x0000007E	/* 126 */
.L4_383:
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
        .long   0x3f000000      /*  0.5 */
.L4_384:
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
        .long   0x00800000      /* explicit mantissa 1 bit */
.L4_385:
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
        .long   0x007f0000      /* table lookup mask */
.L4_386:
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
.L4_387:
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
	.long	0x7f800000	/* +inf */
.L4_388:
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
	.long	0x3f318000	/* 6.93359375E-1 */
.L4_389:
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
	.long	0xb95e8083	/* -2.12194442E-4 */
.L4_390:
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
	.long   0xffc00000      /* -nan */
.L4_391:
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
	.long   0xff800000      /* -inf */
.L4_392:
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
	.long   0x4b000000      /* 2**23 */
.L4_393:
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
	.long   0x00000017      /* 23 */
.L4_394:
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */
	.long   0x00400000      /* convert nan */

.L4_395:
	.long   0x3ede5bd9      /* log10 cvt */
	.long   0x3ede5bd9      /* log10 cvt */
	.long   0x3ede5bd9      /* log10 cvt */
	.long   0x3ede5bd9      /* log10 cvt */

.L4_396:
	.long   0x3e9a2000      /* log10 cvt */
	.long   0x3e9a2000      /* log10 cvt */
	.long   0x3e9a2000      /* log10 cvt */
	.long   0x3e9a2000      /* log10 cvt */

.L4_397:
	.long   0x369a84fc      /* log10 cvt */
	.long   0x369a84fc      /* log10 cvt */
	.long   0x369a84fc      /* log10 cvt */
	.long   0x369a84fc      /* log10 cvt */

.L_STATICS1:
	.long	0xbd9cdc1f	/* -7.65917227E-2 */
	.long	0xbd0d279b	/* -3.44615988E-2 */
	.long	0xbb862ed6	/* -4.09493875E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd9589dd	/* -7.30168596E-2 */
	.long	0xbd04b0fb	/* -3.23953442E-2 */
	.long	0xbb78cc6f	/* -3.79636488E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd8e7c7f	/* -6.95733950E-2 */
	.long	0xbcf94c63	/* -3.04319318E-2 */
	.long	0xbb6674dd	/* -3.51648708E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd87b202	/* -6.62574917E-2 */
	.long	0xbcea05ae	/* -2.85671614E-2 */
	.long	0xbb55464b	/* -3.25431186E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd812871	/* -6.30654171E-2 */
	.long	0xbcdb8552	/* -2.67969705E-2 */
	.long	0xbb4530d2	/* -3.00889136E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd75bbcf	/* -5.99935614E-2 */
	.long	0xbccdc318	/* -2.51174420E-2 */
	.long	0xbb362543	/* -2.77932058E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd69a114	/* -5.70383817E-2 */
	.long	0xbcc0b711	/* -2.35247929E-2 */
	.long	0xbb281520	/* -2.56473571E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd5dfd16	/* -5.41964397E-2 */
	.long	0xbcb45996	/* -2.20153742E-2 */
	.long	0xbb1af296	/* -2.36431276E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd52cc53	/* -5.14643900E-2 */
	.long	0xbca8a345	/* -2.05856655E-2 */
	.long	0xbb0eb077	/* -2.17726617E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd480b61	/* -4.88389768E-2 */
	.long	0xbc9d8cfe	/* -1.92322694E-2 */
	.long	0xbb034232	/* -2.00284692E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd3db6ea	/* -4.63170186E-2 */
	.long	0xbc930fe2	/* -1.79519095E-2 */
	.long	0xbaf13794	/* -1.84034044E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd33cbb1	/* -4.38954271E-2 */
	.long	0xbc89254d	/* -1.67414192E-2 */
	.long	0xbadd63ab	/* -1.68906653E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd2a468d	/* -4.15711887E-2 */
	.long	0xbc7f8db7	/* -1.55977523E-2 */
	.long	0xbacaf2e7	/* -1.54837675E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd21246a	/* -3.93413678E-2 */
	.long	0xbc6ddcc1	/* -1.45179639E-2 */
	.long	0xbab9d094	/* -1.41765410E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd186248	/* -3.72031033E-2 */
	.long	0xbc5d2bd1	/* -1.34992162E-2 */
	.long	0xbaa9e8fa	/* -1.29631092E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd0ffd3a	/* -3.51536050E-2 */
	.long	0xbc4d6f6c	/* -1.25387721E-2 */
	.long	0xba9b2955	/* -1.18378794E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd07f267	/* -3.31901573E-2 */
	.long	0xbc3e9c7f	/* -1.16339913E-2 */
	.long	0xba8d7fce	/* -1.07955351E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbd003f08	/* -3.13101113E-2 */
	.long	0xbc30a85f	/* -1.07823303E-2 */
	.long	0xba80db6e	/* -9.83102014E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbcf1c0ce	/* -2.95108818E-2 */
	.long	0xbc2388c1	/* -9.98133514E-3 */
	.long	0xba6a5829	/* -8.93952849E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbce3a7c0	/* -2.77899504E-2 */
	.long	0xbc1733bb	/* -9.22864210E-3 */
	.long	0xba54c4e2	/* -8.11649603E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbcd62dc0	/* -2.61448622E-2 */
	.long	0xbc0b9fbe	/* -8.52197222E-3 */
	.long	0xba40dff1	/* -7.35758862E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbcc94dc6	/* -2.45732181E-2 */
	.long	0xbc00c394	/* -7.85912946E-3 */
	.long	0xba2e8dbc	/* -6.65869331E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbcbd02ec	/* -2.30726823E-2 */
	.long	0xbbed2cbc	/* -7.23799877E-3 */
	.long	0xba1db418	/* -6.01591077E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbcb1486a	/* -2.16409750E-2 */
	.long	0xbbda1f23	/* -6.65654382E-3 */
	.long	0xba0e3a31	/* -5.42554131E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbca61995	/* -2.02758703E-2 */
	.long	0xbbc84de6	/* -6.11280184E-3 */
	.long	0xba000887	/* -4.88408317E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc9b71df	/* -1.89751964E-2 */
	.long	0xbbb7a92e	/* -5.60488459E-3 */
	.long	0xb9e611ae	/* -4.38821909E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc914cd6	/* -1.77368335E-2 */
	.long	0xbba821bc	/* -5.13097458E-3 */
	.long	0xb9ce4c22	/* -3.93481052E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc87a625	/* -1.65587161E-2 */
	.long	0xbb99a8e4	/* -4.68932278E-3 */
	.long	0xb9b89895	/* -3.52088973E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc7cf31c	/* -1.54388212E-2 */
	.long	0xbb8c3089	/* -4.27824678E-3 */
	.long	0xb9a4d167	/* -3.14365345E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc6b85e0	/* -1.43751800E-2 */
	.long	0xbb7f5631	/* -3.89612862E-3 */
	.long	0xb992d30d	/* -2.80045351E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc5afc82	/* -1.33658666E-2 */
	.long	0xbb68170c	/* -3.54141276E-3 */
	.long	0xb9827c01	/* -2.48879223E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc4b4f20	/* -1.24090016E-2 */
	.long	0xbb528a8e	/* -3.21260421E-3 */
	.long	0xb9675956	/* -2.20631569E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc3c7607	/* -1.15027493E-2 */
	.long	0xbb3e98a1	/* -2.90826731E-3 */
	.long	0xb94c8e8c	/* -1.95080589E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc2e69b2	/* -1.06453169E-2 */
	.long	0xbb2c2a23	/* -2.62702326E-3 */
	.long	0xb9345fa2	/* -1.72017637E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc2122c7	/* -9.83495172E-3 */
	.long	0xbb1b28de	/* -2.36754818E-3 */
	.long	0xb91e97f0	/* -1.51246553E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc149a19	/* -9.06994287E-3 */
	.long	0xbb0b7f83	/* -2.12857197E-3 */
	.long	0xb90b0603	/* -1.32583125E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbc08c8a6	/* -8.34861957E-3 */
	.long	0xbafa3349	/* -1.90887705E-3 */
	.long	0xb8f2f6ef	/* -1.15854542E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbbfb4f23	/* -7.66934594E-3 */
	.long	0xbadfc758	/* -1.70729589E-3 */
	.long	0xb8d399ae	/* -1.00898891E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbbe66055	/* -7.03052664E-3 */
	.long	0xbac795ac	/* -1.52271008E-3 */
	.long	0xb8b7a2e8	/* -8.75646365E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbbd2b7ca	/* -6.43060077E-3 */
	.long	0xbab17a57	/* -1.35404884E-3 */
	.long	0xb89ec690	/* -7.57101225E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbbc048b7	/* -5.86804328E-3 */
	.long	0xba9d52f8	/* -1.20028760E-3 */
	.long	0xb888bdab	/* -6.52031376E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbbaf0697	/* -5.34136174E-3 */
	.long	0xba8afeaf	/* -1.06044661E-3 */
	.long	0xb86a8c21	/* -5.59204527E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb9ee532	/* -4.84909955E-3 */
	.long	0xba74bc21	/* -9.33589472E-4 */
	.long	0xb848445a	/* -4.77473732E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb8fd891	/* -4.38983040E-3 */
	.long	0xba56a63d	/* -8.18822358E-4 */
	.long	0xb82a3195	/* -4.05773353E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb81d503	/* -3.96216055E-3 */
	.long	0xba3b8273	/* -7.15292234E-4 */
	.long	0xb80fe9b0	/* -3.43114953E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb699e30	/* -3.56472656E-3 */
	.long	0xba231a33	/* -6.22186053E-4 */
	.long	0xb7f214d4	/* -2.88583469E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb517740	/* -3.19619477E-3 */
	.long	0xba0d3986	/* -5.38729480E-4 */
	.long	0xb7ca71ef	/* -2.41333310E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb3b1f56	/* -2.85526132E-3 */
	.long	0xb9f35df4	/* -4.64185723E-4 */
	.long	0xb7a84348	/* -2.00584909E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb268108	/* -2.54064985E-3 */
	.long	0xb9d09725	/* -3.97854630E-4 */
	.long	0xb78aeed9	/* -1.65621041E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb138766	/* -2.25111237E-3 */
	.long	0xb9b1c567	/* -3.39071470E-4 */
	.long	0xb763ce9f	/* -1.35783621E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbb021df1	/* -1.98542723E-3 */
	.long	0xb9969428	/* -2.87206145E-4 */
	.long	0xb739569f	/* -1.10470273E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbae46136	/* -1.74239906E-3 */
	.long	0xb97d66a3	/* -2.41661954E-4 */
	.long	0xb7158999	/* -8.91312902E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbac7578a	/* -1.52085838E-3 */
	.long	0xb953ae60	/* -2.01874878E-4 */
	.long	0xb6ef217d	/* -7.12665360E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xbaacf873	/* -1.31966022E-3 */
	.long	0xb92f70a1	/* -1.67312581E-4 */
	.long	0xb6bd529c	/* -5.64225593E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xba951e59	/* -1.13768422E-3 */
	.long	0xb91026c4	/* -1.37473515E-4 */
	.long	0xb69446aa	/* -4.41897009E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xba7f48d8	/* -9.73833259E-4 */
	.long	0xb8eaa465	/* -1.11886104E-4 */
	.long	0xb6658219	/* -3.41993859E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xba58cd48	/* -8.27033538E-4 */
	.long	0xb8bcf84b	/* -9.01078674E-5 */
	.long	0xb62f4c5b	/* -2.61214768E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xba368373	/* -6.96233648E-4 */
	.long	0xb8966ae5	/* -7.17246803E-5 */
	.long	0xb603f28b	/* -1.96616998E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xba182641	/* -5.80404012E-4 */
	.long	0xb86c594e	/* -5.63499561E-5 */
	.long	0xb5c36901	/* -1.45591923E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb9fae416	/* -4.78536531E-4 */
	.long	0xb836f8ce	/* -4.36238988E-5 */
	.long	0xb58e0eb4	/* -1.05841036E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb9cc491c	/* -3.89643828E-4 */
	.long	0xb80b4df6	/* -3.32127893E-5 */
	.long	0xb54a4625	/* -7.53529378E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb9a3f9cb	/* -3.12758930E-4 */
	.long	0xb7d01b57	/* -2.48082633E-5 */
	.long	0xb50c9bf8	/* -5.23810286E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb9817701	/* -2.46934622E-4 */
	.long	0xb7980ea2	/* -1.81266259E-5 */
	.long	0xb4be2b8d	/* -3.54219452E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb9488869	/* -1.91243031E-4 */
	.long	0xb758904b	/* -1.29081991E-5 */
	.long	0xb4790d41	/* -2.31947539E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb917cec7	/* -1.44775127E-4 */
	.long	0xb71598ca	/* -8.91666605E-6 */
	.long	0xb41cfd4e	/* -1.46207839E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb8dfa413	/* -1.06640298E-4 */
	.long	0xb6c742e6	/* -5.93845016E-6 */
	.long	0xb3bd1174	/* -8.80417304E-8 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb89f4fe1	/* -7.59658942E-5 */
	.long	0xb67dd02b	/* -3.78211075E-6 */
	.long	0xb3574eda	/* -5.01303035E-8 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb859abb9	/* -5.18967609E-5 */
	.long	0xb618db75	/* -2.27775058E-6 */
	.long	0xb2e49912	/* -2.66123017E-8 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb80ce838	/* -3.35948716E-5 */
	.long	0xb5ab5271	/* -1.27644864E-6 */
	.long	0xb25dc201	/* -1.29080044E-8 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb7a9c6a5	/* -2.02388710E-5 */
	.long	0xb52e6767	/* -6.49705214E-7 */
	.long	0xb1bea891	/* -5.54889157E-9 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb738f281	/* -1.10237170E-5 */
	.long	0xb49b1ae1	/* -2.88905568E-7 */
	.long	0xb10a52e0	/* -2.01287520E-9 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb6ad2653	/* -5.16026330E-6 */
	.long	0xb3e10d61	/* -1.04798126E-7 */
	.long	0xb01b46ab	/* -5.64890967E-10 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb5fba525	/* -1.87490207E-6 */
	.long	0xb2e7d4f6	/* -2.69887828E-8 */
	.long	0xaee1cb05	/* -1.02678789E-10 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb4dbaf02	/* -4.09192637E-7 */
	.long	0xb16d1190	/* -3.44980222E-9 */
	.long	0xad04ebf2	/* -7.55572167E-12 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0xb2a1cc60	/* -1.88358058E-8 */
	.long	0xae303d23	/* -4.00720672E-11 */
	.long	0x00000000	/* 0.0 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x34480cf5	/*  1.86311652E-7 */
	.long	0xb07c06b1	/* -9.16865750E-10 */
	.long	0x2b5ddf14	/*  7.88245554E-13 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3667765f	/*  3.44905834E-6 */
	.long	0xb38021f5	/* -5.96664123E-8 */
	.long	0x2f93b6dd	/*  2.68690764E-10 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x377de0db	/*  1.51323284E-5 */
	.long	0xb4ed0607	/* -4.41490755E-7 */
	.long	0x316a3501	/*  3.40816109E-9 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3829fd6c	/*  4.05287574E-5 */
	.long	0xb5deaa1d	/* -1.65897984E-6 */
	.long	0x329b04b0	/*  1.80465065E-8 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x38b1b22b	/*  8.47320407E-5 */
	.long	0xb695a9f8	/* -4.46033300E-6 */
	.long	0x33863f7a	/*  6.25140757E-8 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x39200f8f	/*  1.52645851E-4 */
	.long	0xb724b65f	/* -9.81762332E-6 */
	.long	0x3434b4c7	/*  1.68295728E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x39828b2c	/*  2.48992234E-4 */
	.long	0xb79ead30	/* -1.89157145E-5 */
	.long	0x34cdc403	/*  3.83268429E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x39c65937	/*  3.78319732E-4 */
	.long	0xb80b01b9	/* -3.31417868E-5 */
	.long	0x354ff7ab	/*  7.74739135E-7 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3a0edf0f	/*  5.45010844E-4 */
	.long	0xb862cf12	/* -5.40754481E-5 */
	.long	0x35c03d52	/*  1.43229613E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3a457864	/*  7.53289321E-4 */
	.long	0xb8af11a5	/* -8.34793682E-5 */
	.long	0x3625caa1	/*  2.47048615E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3a8404ee	/*  1.00722699E-3 */
	.long	0xb9014785	/* -1.23290418E-4 */
	.long	0x368744d9	/*  4.03132844E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3aabcd7c	/*  1.31075038E-3 */
	.long	0xb9382450	/* -1.75611349E-4 */
	.long	0x36d2f224	/*  6.28667658E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3ada94ef	/*  1.66764657E-3 */
	.long	0xb97e7e08	/* -2.42702779E-4 */
	.long	0x371e625f	/*  9.44043768E-6 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3b086af2	/*  2.08156975E-3 */
	.long	0xb9ab6df2	/* -3.26975773E-4 */
	.long	0x37665cbb	/*  1.37306588E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3b278354	/*  2.55604554E-3 */
	.long	0xb9e1f5c8	/* -4.30984655E-4 */
	.long	0x37a300cc	/*  1.94314853E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3b4accbb	/*  3.09447828E-3 */
	.long	0xba121fd9	/* -5.57420368E-4 */
	.long	0x37e146b2	/*  2.68550102E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3b727e49	/*  3.70015414E-3 */
	.long	0xba39e32a	/* -7.09104002E-4 */
	.long	0x381879bd	/*  3.63530016E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3b8f669e	/*  4.37624659E-3 */
	.long	0xba690a7e	/* -8.88980809E-4 */
	.long	0x384aa9a5	/*  4.83185468E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3ba7f684	/*  5.12582250E-3 */
	.long	0xba9031b6	/* -1.10011431E-3 */
	.long	0x38848391	/*  6.31875664E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3bc307ae	/*  5.95184322E-3 */
	.long	0xbab06192	/* -1.34568126E-3 */
	.long	0x38aacaea	/*  8.14402738E-5 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3be0b221	/*  6.85717212E-3 */
	.long	0xbad58306	/* -1.62896584E-3 */
	.long	0x38d94533	/*  1.03602557E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c00868a	/*  7.84457661E-3 */
	.long	0xbb0003dd	/* -1.95335527E-3 */
	.long	0x390892f8	/*  1.30247208E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c12177d	/*  8.91673286E-3 */
	.long	0xbb183251	/* -2.32233503E-3 */
	.long	0x3929dd3f	/*  1.61995165E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c2516c4	/*  1.00762285E-2 */
	.long	0xbb3388eb	/* -2.73948419E-3 */
	.long	0x39513558	/*  1.99516653E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c398edf	/*  1.13255670E-2 */
	.long	0xbb524539	/* -3.20847169E-3 */
	.long	0x397f5cb1	/*  2.43532253E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c4f89f7	/*  1.26671707E-2 */
	.long	0xbb74a637	/* -3.73305171E-3 */
	.long	0x399a9140	/*  2.94813886E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c6711e1	/*  1.41033838E-2 */
	.long	0xbb8d7621	/* -4.31706058E-3 */
	.long	0x39b9b202	/*  3.54185759E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c801810	/*  1.56364739E-2 */
	.long	0xbba2ac82	/* -4.96441219E-3 */
	.long	0x39dd8664	/*  4.22525336E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c8d76f6	/*  1.72686391E-2 */
	.long	0xbbba17b6	/* -5.67909610E-3 */
	.long	0x3a0345b5	/*  5.00764058E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3c9baa17	/*  1.90020036E-2 */
	.long	0xbbd3d9cc	/* -6.46517240E-3 */
	.long	0x3a1aa2bc	/*  5.89888310E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3caab5c7	/*  2.08386313E-2 */
	.long	0xbbf01567	/* -7.32677011E-3 */
	.long	0x3a352033	/*  6.90940011E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3cba9e35	/*  2.27805171E-2 */
	.long	0xbc0776db	/* -8.26808345E-3 */
	.long	0x3a5307cc	/*  8.05017306E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3ccb676f	/*  2.48295944E-2 */
	.long	0xbc184337	/* -9.29336902E-3 */
	.long	0x3a74a70f	/*  9.33275500E-4 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3cdd1562	/*  2.69877352E-2 */
	.long	0xbc2a81e2	/* -1.04069430E-2 */
	.long	0x3a8d27b0	/*  1.07692741E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3cefabdf	/*  2.92567592E-2 */
	.long	0xbc3e4535	/* -1.16131799E-2 */
	.long	0x3aa22b05	/*  1.23724400E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d01974c	/*  3.16384286E-2 */
	.long	0xbc539fc4	/* -1.29165091E-2 */
	.long	0x3ab98a24	/*  1.41555490E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d0bd090	/*  3.41344476E-2 */
	.long	0xbc6aa459	/* -1.43214101E-2 */
	.long	0x3ad373ab	/*  1.61324942E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d168379	/*  3.67464758E-2 */
	.long	0xbc81b2fc	/* -1.58324167E-2 */
	.long	0x3af01836	/*  1.83177623E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d21b1b5	/*  3.94761153E-2 */
	.long	0xbc8efbea	/* -1.74541064E-2 */
	.long	0x3b07d535	/*  2.07264465E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d2d5ce9	/*  4.23249342E-2 */
	.long	0xbc9d36ab	/* -1.91911068E-2 */
	.long	0x3b192f78	/*  2.33742408E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d3986a9	/*  4.52944376E-2 */
	.long	0xbcac6d09	/* -2.10480858E-2 */
	.long	0x3b2c3643	/*  2.62774597E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d46307f	/*  4.83860932E-2 */
	.long	0xbcbca8e8	/* -2.30297595E-2 */
	.long	0x3b4105fa	/*  2.94530252E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d535bec	/*  5.16013354E-2 */
	.long	0xbccdf43f	/* -2.51408797E-2 */
	.long	0x3b57bc0b	/*  3.29184788E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d610a61	/*  5.49415387E-2 */
	.long	0xbce0591a	/* -2.73862369E-2 */
	.long	0x3b7076f2	/*  3.66919907E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d6f3d49	/*  5.84080555E-2 */
	.long	0xbcf3e19c	/* -2.97706649E-2 */
	.long	0x3b85ab1b	/*  4.07923525E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d7df600	/*  6.20021820E-2 */
	.long	0xbd044bfb	/* -3.22990231E-2 */
	.long	0x3b943d35	/*  4.52389801E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d869aee	/*  6.57251924E-2 */
	.long	0xbd0f4339	/* -3.49762179E-2 */
	.long	0x3ba4029b	/*  5.00519341E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d8e7f14	/*  6.95783198E-2 */
	.long	0xbd1adbb3	/* -3.78071778E-2 */
	.long	0x3bb50ca6	/*  5.52518945E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d96a813	/*  7.35627636E-2 */
	.long	0xbd271a9d	/* -4.07968648E-2 */
	.long	0x3bc76d3b	/*  6.08601933E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3d9f1686	/*  7.76796788E-2 */
	.long	0xbd340536	/* -4.39502820E-2 */
	.long	0x3bdb36c6	/*  6.68987911E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3da7cb06	/*  8.19302052E-2 */
	.long	0xbd41a0c1	/* -4.72724475E-2 */
	.long	0x3bf07c3f	/*  7.33903004E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3db0c625	/*  8.63154307E-2 */
	.long	0xbd4ff28a	/* -5.07684126E-2 */
	.long	0x3c03a893	/*  8.03579669E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */
	.long	0x3dba0875	/*  9.08364430E-2 */
	.long	0xbd5effe5	/* -5.44432588E-2 */
	.long	0x3c0fe4c4	/*  8.78256932E-3 */
	.long	0x3eaaaaab	/* 1.0/3.0 */

/* ============================================================
 *
 *     Constants for logrithm functions, double precision
 *
 * ============================================================
 */

	ALN_QUAD
.L__real_nan:			.quad 0x0fff8000000000000	/* NaN */
.L__real_qnanbit:		.quad 0x00008000000000000	/* quiet nan bit */
.L__real_inf:			.quad 0x07ff0000000000000	/* +inf */
.L__real_ninf:			.quad 0x0fff0000000000000	/* -inf */
.L__real_notsign:		.quad 0x07ffFFFFFFFFFFFFF	/* ^sign bit */
				.quad 0x07ffFFFFFFFFFFFFF
.L__real_one:			.quad 0x03ff0000000000000	/* 1.0 */
				.quad 0x03ff0000000000000
.L__real_two:			.quad 0x04000000000000000	/* 2.0 */
				.quad 0x04000000000000000
.L__mask_1075:			.quad 0x00000000000000433	/*  */
				.quad 0x00000000000000433
.L__mask_1023:			.quad 0x000000000000003ff	/*  */
				.quad 0x000000000000003ff
.L__mask_040:			.quad 0x00000000000000040	/*  */
				.quad 0x00000000000000040
.L__mask_001:			.quad 0x00000000000000001	/*  */
				.quad 0x00000000000000001
.L__real_ca1:			.quad 0x03fb55555555554e6	/* 8.33333333333317923934e-02 */
				.quad 0x03fb55555555554e6
.L__real_ca2:			.quad 0x03f89999999bac6d4	/* 1.25000000037717509602e-02 */
				.quad 0x03f89999999bac6d4
.L__real_ca3:			.quad 0x03f62492307f1519f	/* 2.23213998791944806202e-03 */
				.quad 0x03f62492307f1519f
.L__real_ca4:			.quad 0x03f3c8034c85dfff0	/* 4.34887777707614552256e-04 */
				.quad 0x03f3c8034c85dfff0
.L__real_cb1:			.quad 0x03fb5555555555557	/* 8.33333333333333593622e-02 */
				.quad 0x03fb5555555555557
.L__real_cb2:			.quad 0x03f89999999865ede	/* 1.24999999978138668903e-02 */
				.quad 0x03f89999999865ede
.L__real_cb3:			.quad 0x03f6249423bd94741	/* 2.23219810758559851206e-03 */
				.quad 0x03f6249423bd94741
.L__real_log2_lead:  		.quad 0x03fe62e42e0000000	/* 6.93147122859954833984e-01 */
				.quad 0x03fe62e42e0000000
.L__real_log2_tail: 		.quad 0x03e6efa39ef35793c	/* 5.76999904754328540596e-08 */
				.quad 0x03e6efa39ef35793c
.L__real_scale:			.quad 0x04330000000000000	/* 2 ** 52 */
				.quad 0x04330000000000000
.L__real_threshold:		.quad 0x03F9EB85000000000	/* .03 */
				.quad 0x03F9EB85000000000
.L__real_mant:			.quad 0x0000FFFFFFFFFFFFF	/* mantissa bits */
				.quad 0x0000FFFFFFFFFFFFF
.L__real_3f80000000000000:	.quad 0x03f80000000000000	/* 0.0078125 = 1/128 */
				.quad 0x03f80000000000000
.L__real_half:			.quad 0x03fe0000000000000	/* 1/2 */
				.quad 0x03fe0000000000000
.L__real_maxfp:			.quad 0x07fefffffffffffff	/* 1/2 */
				.quad 0x07fefffffffffffff
.L__real_fffffffff8000000:	.quad 0x0fffffffff8000000	/* high part */
                            	.quad 0x0fffffffff8000000
.L__real_mindp:			.quad 0x00010000000000000	/* small dp */
				.quad 0x00010000000000000

	ALN_QUAD

.L__np_ln_lead_table:
	.quad	0x0000000000000000 		/* 0.00000000000000000000e+00 */
	.quad	0x3f8fc0a800000000		/* 1.55041813850402832031e-02 */
	.quad	0x3f9f829800000000		/* 3.07716131210327148438e-02 */
	.quad	0x3fa7745800000000		/* 4.58095073699951171875e-02 */
	.quad	0x3faf0a3000000000		/* 6.06245994567871093750e-02 */
	.quad	0x3fb341d700000000		/* 7.52233862876892089844e-02 */
	.quad	0x3fb6f0d200000000		/* 8.96121263504028320312e-02 */
	.quad	0x3fba926d00000000		/* 1.03796780109405517578e-01 */
	.quad	0x3fbe270700000000		/* 1.17783010005950927734e-01 */
	.quad	0x3fc0d77e00000000		/* 1.31576299667358398438e-01 */
	.quad	0x3fc2955280000000		/* 1.45181953907012939453e-01 */
	.quad	0x3fc44d2b00000000		/* 1.58604979515075683594e-01 */
	.quad	0x3fc5ff3000000000		/* 1.71850204467773437500e-01 */
	.quad	0x3fc7ab8900000000		/* 1.84922337532043457031e-01 */
	.quad	0x3fc9525a80000000		/* 1.97825729846954345703e-01 */
	.quad	0x3fcaf3c900000000		/* 2.10564732551574707031e-01 */
	.quad	0x3fcc8ff780000000		/* 2.23143517971038818359e-01 */
	.quad	0x3fce270700000000		/* 2.35566020011901855469e-01 */
	.quad	0x3fcfb91800000000		/* 2.47836112976074218750e-01 */
	.quad	0x3fd0a324c0000000		/* 2.59957492351531982422e-01 */
	.quad	0x3fd1675c80000000		/* 2.71933674812316894531e-01 */
	.quad	0x3fd22941c0000000		/* 2.83768117427825927734e-01 */
	.quad	0x3fd2e8e280000000		/* 2.95464158058166503906e-01 */
	.quad	0x3fd3a64c40000000		/* 3.07025015354156494141e-01 */
	.quad	0x3fd4618bc0000000		/* 3.18453729152679443359e-01 */
	.quad	0x3fd51aad80000000		/* 3.29753279685974121094e-01 */
	.quad	0x3fd5d1bd80000000		/* 3.40926527976989746094e-01 */
	.quad	0x3fd686c800000000		/* 3.51976394653320312500e-01 */
	.quad	0x3fd739d7c0000000		/* 3.62905442714691162109e-01 */
	.quad	0x3fd7eaf800000000		/* 3.73716354370117187500e-01 */
	.quad	0x3fd89a3380000000		/* 3.84411692619323730469e-01 */
	.quad	0x3fd9479400000000		/* 3.94993782043457031250e-01 */
	.quad	0x3fd9f323c0000000		/* 4.05465066432952880859e-01 */
	.quad	0x3fda9cec80000000		/* 4.15827870368957519531e-01 */
	.quad	0x3fdb44f740000000		/* 4.26084339618682861328e-01 */
	.quad	0x3fdbeb4d80000000		/* 4.36236739158630371094e-01 */
	.quad	0x3fdc8ff7c0000000		/* 4.46287095546722412109e-01 */
	.quad	0x3fdd32fe40000000		/* 4.56237375736236572266e-01 */
	.quad	0x3fddd46a00000000		/* 4.66089725494384765625e-01 */
	.quad	0x3fde744240000000		/* 4.75845873355865478516e-01 */
	.quad	0x3fdf128f40000000		/* 4.85507786273956298828e-01 */
	.quad	0x3fdfaf5880000000		/* 4.95077252388000488281e-01 */
	.quad	0x3fe02552a0000000		/* 5.04556000232696533203e-01 */
	.quad	0x3fe0723e40000000		/* 5.13945698738098144531e-01 */
	.quad	0x3fe0be72e0000000		/* 5.23248136043548583984e-01 */
	.quad	0x3fe109f380000000		/* 5.32464742660522460938e-01 */
	.quad	0x3fe154c3c0000000		/* 5.41597247123718261719e-01 */
	.quad	0x3fe19ee6a0000000		/* 5.50647079944610595703e-01 */
	.quad	0x3fe1e85f40000000		/* 5.59615731239318847656e-01 */
	.quad	0x3fe23130c0000000		/* 5.68504691123962402344e-01 */
	.quad	0x3fe2795e00000000		/* 5.77315330505371093750e-01 */
	.quad	0x3fe2c0e9e0000000		/* 5.86049020290374755859e-01 */
	.quad	0x3fe307d720000000		/* 5.94707071781158447266e-01 */
	.quad	0x3fe34e2880000000		/* 6.03290796279907226562e-01 */
	.quad	0x3fe393e0c0000000		/* 6.11801505088806152344e-01 */
	.quad	0x3fe3d90260000000		/* 6.20240390300750732422e-01 */
	.quad	0x3fe41d8fe0000000		/* 6.28608644008636474609e-01 */
	.quad	0x3fe4618bc0000000		/* 6.36907458305358886719e-01 */
	.quad	0x3fe4a4f840000000		/* 6.45137906074523925781e-01 */
	.quad	0x3fe4e7d800000000		/* 6.53301239013671875000e-01 */
	.quad	0x3fe52a2d20000000		/* 6.61398470401763916016e-01 */
	.quad	0x3fe56bf9c0000000		/* 6.69430613517761230469e-01 */
	.quad	0x3fe5ad4040000000		/* 6.77398800849914550781e-01 */
	.quad	0x3fe5ee02a0000000		/* 6.85303986072540283203e-01 */
	.quad	0x3fe62e42e0000000		/* 6.93147122859954833984e-01 */
	.quad 0					/* for alignment */


.L__np_ln_tail_table:
	.quad	0x00000000000000000 		/* 0.00000000000000000000e+00 */
	.quad	0x03e361f807c79f3db		/* 5.15092497094772879206e-09 */
	.quad	0x03e6873c1980267c8		/* 4.55457209735272790188e-08 */
	.quad	0x03e5ec65b9f88c69e		/* 2.86612990859791781788e-08 */
	.quad	0x03e58022c54cc2f99		/* 2.23596477332056055352e-08 */
	.quad	0x03e62c37a3a125330		/* 3.49498983167142274770e-08 */
	.quad	0x03e615cad69737c93		/* 3.23392843005887000414e-08 */
	.quad	0x03e4d256ab1b285e9		/* 1.35722380472479366661e-08 */
	.quad	0x03e5b8abcb97a7aa2		/* 2.56504325268044191098e-08 */
	.quad	0x03e6f34239659a5dc		/* 5.81213608741512136843e-08 */
	.quad	0x03e6e07fd48d30177		/* 5.59374849578288093334e-08 */
	.quad	0x03e6b32df4799f4f6		/* 5.06615629004996189970e-08 */
	.quad	0x03e6c29e4f4f21cf8		/* 5.24588857848400955725e-08 */
	.quad	0x03e1086c848df1b59		/* 9.61968535632653505972e-10 */
	.quad	0x03e4cf456b4764130		/* 1.34829655346594463137e-08 */
	.quad	0x03e63a02ffcb63398		/* 3.65557749306383026498e-08 */
	.quad	0x03e61e6a6886b0976		/* 3.33431709374069198903e-08 */
	.quad	0x03e6b8abcb97a7aa2		/* 5.13008650536088382197e-08 */
	.quad	0x03e6b578f8aa35552		/* 5.09285070380306053751e-08 */
	.quad	0x03e6139c871afb9fc		/* 3.20853940845502057341e-08 */
	.quad	0x03e65d5d30701ce64		/* 4.06713248643004200446e-08 */
	.quad	0x03e6de7bcb2d12142		/* 5.57028186706125221168e-08 */
	.quad	0x03e6d708e984e1664		/* 5.48356693724804282546e-08 */
	.quad	0x03e556945e9c72f36		/* 1.99407553679345001938e-08 */
	.quad	0x03e20e2f613e85bda		/* 1.96585517245087232086e-09 */
	.quad	0x03e3cb7e0b42724f6		/* 6.68649386072067321503e-09 */
	.quad	0x03e6fac04e52846c7		/* 5.89936034642113390002e-08 */
	.quad	0x03e5e9b14aec442be		/* 2.85038578721554472484e-08 */
	.quad	0x03e6b5de8034e7126		/* 5.09746772910284482606e-08 */
	.quad	0x03e6dc157e1b259d3		/* 5.54234668933210171467e-08 */
	.quad	0x03e3b05096ad69c62		/* 6.29100830926604004874e-09 */
	.quad	0x03e5c2116faba4cdd		/* 2.61974119468563937716e-08 */
	.quad	0x03e665fcc25f95b47		/* 4.16752115011186398935e-08 */
	.quad	0x03e5a9a08498d4850		/* 2.47747534460820790327e-08 */
	.quad	0x03e6de647b1465f77		/* 5.56922172017964209793e-08 */
	.quad	0x03e5da71b7bf7861d		/* 2.76162876992552906035e-08 */
	.quad	0x03e3e6a6886b09760		/* 7.08169709942321478061e-09 */
	.quad	0x03e6f0075eab0ef64		/* 5.77453510221151779025e-08 */
	.quad	0x03e33071282fb989b		/* 4.43021445893361960146e-09 */
	.quad	0x03e60eb43c3f1bed2		/* 3.15140984357495864573e-08 */
	.quad	0x03e5faf06ecb35c84		/* 2.95077445089736670973e-08 */
	.quad	0x03e4ef1e63db35f68		/* 1.44098510263167149349e-08 */
	.quad	0x03e469743fb1a71a5		/* 1.05196987538551827693e-08 */
	.quad	0x03e6c1cdf404e5796		/* 5.23641361722697546261e-08 */
	.quad	0x03e4094aa0ada625e		/* 7.72099925253243069458e-09 */
	.quad	0x03e6e2d4c96fde3ec		/* 5.62089493829364197156e-08 */
	.quad	0x03e62f4d5e9a98f34		/* 3.53090261098577946927e-08 */
	.quad	0x03e6467c96ecc5cbe		/* 3.80080516835568242269e-08 */
	.quad	0x03e6e7040d03dec5a		/* 5.66961038386146408282e-08 */
	.quad	0x03e67bebf4282de36		/* 4.42287063097349852717e-08 */
	.quad	0x03e6289b11aeb783f		/* 3.45294525105681104660e-08 */
	.quad	0x03e5a891d1772f538		/* 2.47132034530447431509e-08 */
	.quad	0x03e634f10be1fb591		/* 3.59655343422487209774e-08 */
	.quad	0x03e6d9ce1d316eb93		/* 5.51581770357780862071e-08 */
	.quad	0x03e63562a19a9c442		/* 3.60171867511861372793e-08 */
	.quad	0x03e54e2adf548084c		/* 1.94511067964296180547e-08 */
	.quad	0x03e508ce55cc8c97a		/* 1.54137376631349347838e-08 */
	.quad	0x03e30e2f613e85bda		/* 3.93171034490174464173e-09 */
	.quad	0x03e6db03ebb0227bf		/* 5.52990607758839766440e-08 */
	.quad	0x03e61b75bb09cb098		/* 3.29990737637586136511e-08 */
	.quad	0x03e496f16abb9df22		/* 1.18436010922446096216e-08 */
	.quad	0x03e65b3f399411c62		/* 4.04248680368301346709e-08 */
	.quad	0x03e586b3e59f65355		/* 2.27418915900284316293e-08 */
	.quad	0x03e52482ceae1ac12		/* 1.70263791333409206020e-08 */
	.quad	0x03e6efa39ef35793c		/* 5.76999904754328540596e-08 */
	.quad 0					/* for alignment */

/* ============================================================
 *
 *     Constants for sin/cos functions
 *
 * ============================================================
 */

        ALN_QUAD
.L__dble_pq4:           .quad 0x03ec71dc0d6e93878      /* p4 */
                        .quad 0x03ec71dc0d6e93878      /* p4 */
                        .quad 0x03efa017307437b32      /* q4 */
                        .quad 0x03efa017307437b32      /* q4 */
.L__dble_pq3:           .quad 0x0bf2a01a019e1edc4      /* p3 */
                        .quad 0x0bf2a01a019e1edc4      /* p3 */
                        .quad 0x0bf56c16c169b8d77      /* q3 */
                        .quad 0x0bf56c16c169b8d77      /* q3 */
.L__dble_pq2:           .quad 0x03f81111111111105      /* p2 */
                        .quad 0x03f81111111111105      /* p2 */
                        .quad 0x03fa555555555553c      /* q2 */
                        .quad 0x03fa555555555553c      /* q2 */
.L__dble_pq1:           .quad 0x0bfc5555555555555      /* p1 */
                        .quad 0x0bfc5555555555555      /* p1 */
                        .quad 0x0bfe0000000000000      /* q1 (-0.5) */
                        .quad 0x0bfe0000000000000      /* q1 (-0.5) */
.L__dble_sixteen_by_pi: .quad 0x040145f306dc9c883      /* 16.0 / pi */
.L__dble_pi_by_16_ms:   .quad 0x03fc921fb54400000
.L__dble_pi_by_16_ls:   .quad 0x03da0b4611a600000
.L__dble_pi_by_16_us:   .quad 0x03b73198a2e037073
.L__dble_sincostbl:     .quad 0x03FF0000000000000      /* cos( 0*pi/16) */
                        .quad 0x00000000000000000      /* rem( 0*pi/16) */
                        .quad 0x03FEF6297CF000000      /* cos( 1*pi/16) */
                        .quad 0x03E1EEB96055885CB      /* rem( 1*pi/16) */
                        .quad 0x03FED906BCF000000      /* cos( 2*pi/16) */
                        .quad 0x03DF946A31457E610      /* rem( 2*pi/16) */
                        .quad 0x03FEA9B6629000000      /* cos( 3*pi/16) */
                        .quad 0x03DDD4346067D8C3A      /* rem( 3*pi/16) */
                        .quad 0x03FE6A09E66000000      /* cos( 4*pi/16) */
                        .quad 0x03E0FCEF32422CBEC      /* rem( 4*pi/16) */
                        .quad 0x03FE1C73B39000000      /* cos( 5*pi/16) */
                        .quad 0x03E15CD190D92EE93      /* rem( 5*pi/16) */
                        .quad 0x03FD87DE2A6000000      /* cos( 6*pi/16) */
                        .quad 0x03E05D52C5A34C48B      /* rem( 6*pi/16) */
                        .quad 0x03FC8F8B83C000000      /* cos( 7*pi/16) */
                        .quad 0x03DEA6982AD92E646      /* rem( 7*pi/16) */
                        .quad 0x00000000000000000      /* cos( 8*pi/16) */
                        .quad 0x00000000000000000      /* rem( 8*pi/16) */
                        .quad 0x0BFC8F8B83C000000      /* cos( 9*pi/16) */
                        .quad 0x0BDEA6982AD92E646      /* rem( 9*pi/16) */
                        .quad 0x0BFD87DE2A6000000      /* cos(10*pi/16) */
                        .quad 0x0BE05D52C5A34C48B      /* rem(10*pi/16) */
                        .quad 0x0BFE1C73B39000000      /* cos(11*pi/16) */
                        .quad 0x0BE15CD190D92EE93      /* rem(11*pi/16) */
                        .quad 0x0BFE6A09E66000000      /* cos(12*pi/16) */
                        .quad 0x0BE0FCEF32422CBEC      /* rem(12*pi/16) */
                        .quad 0x0BFEA9B6629000000      /* cos(13*pi/16) */
                        .quad 0x0BDDD4346067D8C3A      /* rem(13*pi/16) */
                        .quad 0x0BFED906BCF000000      /* cos(14*pi/16) */
                        .quad 0x0BDF946A31457E610      /* rem(14*pi/16) */
                        .quad 0x0BFEF6297CF000000      /* cos(15*pi/16) */
                        .quad 0x0BE1EEB96055885CB      /* rem(15*pi/16) */
                        .quad 0x0BFF0000000000000      /* cos(16*pi/16) */
                        .quad 0x00000000000000000      /* rem(16*pi/16) */
.L__dble_dsin_c6:       .quad 0x03de5e0b2f9a43bb8      /* 1.59181e-010 */
			.quad 0x03de5e0b2f9a43bb8      /* 1.59181e-010 */
.L__dble_dsin_c5:       .quad 0x0be5ae600b42fdfa7      /* -2.50511e-008 */
.L__dble_dsin_c4:       .quad 0x03ec71de3796cde01      /* 2.75573e-006 */
			.quad 0x03ec71de3796cde01      /* 2.75573e-006 */
.L__dble_dsin_c3:       .quad 0x0bf2a01a019e83e5c      /* -0.000198413 */
			.quad 0x0bf2a01a019e83e5c      /* -0.000198413 */
.L__dble_dsin_c2:       .quad 0x03f81111111110bb3      /* 0.00833333 */
			.quad 0x03f81111111110bb3      /* 0.00833333 */
.L__dble_dcos_c6:       .quad 0x0bDA907DB47258AA7      /* -1.13826e-011 */
			.quad 0x0bDA907DB47258AA7      /* -1.13826e-011 */
.L__dble_dcos_c5:       .quad 0x03E21EEB690382EEC      /* 2.08761e-009 */
.L__dble_dcos_c4:       .quad 0x0bE927E4FA17F667B      /* -2.75573e-007 */
			.quad 0x0bE927E4FA17F667B      /* -2.75573e-007 */
.L__dble_dcos_c3:       .quad 0x03EFA01A019F4EC91      /* 2.48016e-005 */
			.quad 0x03EFA01A019F4EC91      /* 2.48016e-005 */
.L__dble_dcos_c2:       .quad 0x0bf56c16c16c16967      /* -0.00138889 */
			.quad 0x0bf56c16c16c16967      /* -0.00138889 */
.L__dble_dcos_c1:       .quad 0x03fa5555555555555      /* 0.0416667 */
			.quad 0x03fa5555555555555      /* 0.0416667 */


        ALN_QUAD
.L__sngl_mask_unsign:   .long 0x07fffffff      /* Mask for unsigned */
                        .long 0x07fffffff      /* Mask for unsigned */
                        .long 0x07fffffff      /* Mask for unsigned */
                        .long 0x07fffffff      /* Mask for unsigned */
.L__sngl_pi_over_fours: .long 0x03f490fdb
                        .long 0x03f490fdb
.L__sngl_sixteen_by_pi: .long 0x040a2f983  
                        .long 0x040a2f983  
.L__sngl_needs_argreds: .long 0x049800000
                        .long 0x049800000


        ALN_QUAD
.L__dble_pi_over_fours: .quad 0x03fe921fb54442d18
.L__dble_needs_argreds: .quad 0x04130000000000000


/* ============================================================
 *
 *     Constants for sinh/cosh functions
 *
 * ============================================================
 */

        ALN_QUAD
.L__dsinh_shortval_y7:
        .quad   0x03d6ae7f3e733b81f
.L__dsinh_shortval_y6:
        .quad   0x03de6124613a86d09
.L__dsinh_shortval_y5:
        .quad   0x03e5ae64567f544e4
.L__dsinh_shortval_y4:
        .quad   0x03ec71de3a556c734
.L__dsinh_shortval_y3:
        .quad   0x03f2a01a01a01a01a
.L__dsinh_shortval_y2:
        .quad   0x03f81111111111111
.L__dsinh_shortval_y1:
        .quad   0x03fc5555555555555
.L__dsinh_too_small:
        .quad   0x03fea800000000000
.L__ps_vssinh_too_small:        .long 0x3d000000
                                .long 0x3d000000
                                .long 0x3d000000
                                .long 0x3d000000

/* ============================================================
 *
 *     Constants for pow functions, single precision
 *
 * ============================================================
 */

        ALN_QUAD
.L4_100:
        .long   0x3f000000
        .long   0x3fc00000
        .long   0x3f800000
        .long   0x00000000
.L4_101:
        .long   0x3f800000
        .long   0x3e800000
        .long   0xbf800000
        .long   0x80000000
.L4_102:
        .long   0x7fffffff
        .long   0x7f800000
        .long   0x80000000
        .long   0x7f800000
.L4_103:
        .long   0x00000000
        .long   0x7f800000
        .long   0x80000000
        .long   0x7f800000
.L4_104:
        .long   0x4f7fffff
.L4_105:
        .long   0x2e800000
.L4_106:
        .long   0x007fffff
.L4_107:
        .long   0x00400000
.L4_108:
        .long   0xff800000

        ALN_QUAD
.L4_fvspow_infinity_mask:
        .long   0x7f800000
        .long   0x7f800000
        .long   0x7f800000
        .long   0x7f800000

/* ============================================================
 *
 *     Constants for pow functions, double precision
 *
 * ============================================================
 */

        ALN_QUAD
.L4_D100:
        .quad   0x03fe0000000000000
        .quad   0x03ff0000000000000
.L4_D101:
        .quad   0x03ff8000000000000
        .quad   0x00000000000000000
.L4_D102:
        .quad   0x03ff0000000000000
        .quad   0x0bff0000000000000
.L4_D103:
        .quad   0x03fd0000000000000
        .quad   0x08000000000000000
.L4_D104:
        .quad   0x07ff0000000000000
        .quad   0x07ff0000000000000
.L4_D105:
        .quad   0x07fffffffffffffff
        .quad   0x08000000000000000
.L4_D106:
        .quad   0x00000000000000000
        .quad   0x08000000000000000
.L4_D107:
        .quad   0x043dfffffffffffff
.L4_D108:
        .quad   0x03c00000000000000
.L4_D109:
        .quad   0x0fff0000000000000
.L4_D10a:
        .quad   0x00008000000000000
.L4_D10b:
        .quad   0x0000fffffffffffff

        ALN_QUAD
.L4_fvdpow_infinity_mask:
        .quad   0x7ff0000000000000
        .quad   0x7ff0000000000000

/* ============================================================
 *
 *     Constants for logarithm(base 10) functions, double precision
 *
 * ============================================================
 */

        ALN_QUAD
.L__log10_multiplier:   .quad 0x3fdbcb7b1526e50e
                        .quad 0x3fdbcb7b1526e50e
.L__log10_multiplier1:  .quad 0x3fdbcb7b00000000
                        .quad 0x3fdbcb7b00000000
.L__log10_multiplier2:  .quad 0x3e5526e50e32a6ab
                        .quad 0x3e5526e50e32a6ab

/* ==============================================================
 *     
 *    Constants for mask intrinsic functions
 *
 * ==============================================================
 */
	ALN_QUAD
.L_zeromask:	.quad 0xFFFFFFFFFFFFFFFF
	        .quad 0xFFFFFFFFFFFFFFFF
        	.quad 0xFFFFFFFFFFFFFFFF
	        .quad 0xFFFFFFFFFFFFFFFF
.L_s_zeromask:	.long 0xFFFFFFFF
		.long 0xFFFFFFFF
		.long 0xFFFFFFFF
		.long 0xFFFFFFFF
		.long 0xFFFFFFFF
		.long 0xFFFFFFFF
		.long 0xFFFFFFFF
		.long 0xFFFFFFFF
        ALN_QUAD
.L_one_for_mask_fvd:
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        .quad 0x03FF0000000000000     /* 1.0000000000000000 */
        ALN_QUAD
.L_one_for_mask_fvs:
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
.L_dpow_mask_two:
        .quad 0x4000000000000000   /* 2.0 */
        .quad 0x4000000000000000   /* 2.0 */
        .quad 0x4000000000000000   /* 2.0 */
        .quad 0x4000000000000000   /* 2.0 */
.L_spow_mask_two:
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */
        .long 0x40000000           /* 2.0 */


#else

#ifdef VEX_TARGET

#include "fastmath_vex.h"
#include "fastmath_vex_mask.h"

#elif defined (HELPER_TARGET)


/* ============================================================ */
/* This routine takes a double input, and produces a 
   single precision output

** MAKE SURE THE STACK HERE MATCHES THE STACK IN __fmth_i_exp

*/
	.text
        ALN_FUNC
ENT(__fss_exp_dbl):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	movsdRR	%xmm0, %xmm2
	mulsd	%xmm0, %xmm3 

	/* Set n = nearest integer to r */
	comisd	.L__dp_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_dp_inf)
	comisd	.L__dp_min_singleval(%rip), %xmm0
#ifdef GH_TARGET
	jnbe    .L__fmth_i_exp_dbl_entry_gh
#else
	jnbe	.L__fmth_i_exp_dbl_entry
#endif

LBL(.L_sp_dp_ninf):
	xor	%eax, %eax
	movd	%eax,%xmm0
	jmp	LBL(.L_sp_dp_final_check)

LBL(.L_sp_dp_inf):
	movlps	.L_sp_real_infinity(%rip),%xmm0
LBL(.L_sp_dp_final_check):
	RZ_POP
	rep
	ret

	ELF_FUNC(__fss_exp_dbl)
	ELF_SIZE(__fss_exp_dbl)


/* ============================================================ */
/* This routine takes two doubles input, and produces a 
   double precision output

** MAKE SURE THE STACK HERE MATCHES THE STACK IN __fmth_i_dexp

*/
	.text
        ALN_FUNC
ENT(__fsd_exp_long):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm0,%xmm3 

	/* Set n = nearest integer to r */
	comisd	.L__real_ln_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf_exp_long)
	comisd	.L__real_ln_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_ninf_exp_long)
	cvtpd2dq %xmm3,%xmm3	/* convert to integer */
	cvtdq2pd %xmm3,%xmm5	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
	movlpdMR	.L__real_log2_by_32_lead(%rip),%xmm2
	mulsd	%xmm5,%xmm2
	movd	%xmm3,%ecx
	subsd	%xmm2,%xmm0	/* r1 in xmm0, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
	mulsd	.L__real_log2_by_32_tail(%rip),%xmm5
	addsd	%xmm1, %xmm5

	/* j = n & 0x0000001f; */
	movq	$0x1f,%rax
	and	%ecx,%eax
	movsdRR	%xmm0,%xmm2	/* r1 */

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%eax,%ecx
	sar	$5,%ecx
	addsd	%xmm5,%xmm2    /* r = r1 + r2 */
	shufpd	$0, %xmm0, %xmm5	/* Store r1 and r2 */

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */

	/* r in %xmm2, r1, r2 in %xmm5 */

	movsdRR	%xmm2,%xmm1
	movlpdMR	.L__real_3f56c1728d739765(%rip),%xmm3
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm0
	mulsd	%xmm2,%xmm3
	mulsd	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	movsdRR	%xmm1,%xmm4
	addsd	.L__real_3F811115B7AA905E(%rip),%xmm3
	addsd	.L__real_3fe0000000000000(%rip),%xmm0
	mulsd	%xmm1,%xmm4
	mulsd	%xmm2,%xmm3
	mulsd	%xmm1,%xmm0
	addsd	.L__real_3FA5555555545D4E(%rip),%xmm3
	addsd	%xmm5,%xmm0
	shufpd	$1, %xmm5, %xmm5	/* Store r1 and r2 */
	mulsd	%xmm4,%xmm3
	addsd	%xmm3,%xmm0
	addsd	%xmm5,%xmm0

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%rdx,%rax,8),%xmm5
	/* deal with infinite results */
        movslq	%ecx,%rcx
	mulsd	%xmm5,%xmm0
	addsd	%xmm5,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* deal with denormal results */
	movq	$1, %rdx
	movq	$1, %rax
        addq	$1022, %rcx	/* add bias */
	cmovleq	%rcx, %rdx
	cmovleq	%rax, %rcx
        shlq	$52,%rcx        /* build 2^n */
        addq	$1023, %rdx	/* add bias */
        shlq	$52,%rdx        /* build 2^n */
	movq	%rdx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

LBL(.Lfinal_check_exp_long):
	RZ_POP
	ret

LBL(.L_inf_exp_long):
        movlpdMR  .L__real_infinity(%rip),%xmm0
        jmp     LBL(.Lfinal_check_exp_long)
LBL(.L_ninf_exp_long):
        jp      LBL(.L_cvt_nan_exp_long)
        xorq    %rax, %rax
        movd    %rax,%xmm0
        jmp     LBL(.Lfinal_check_exp_long)
LBL(.L_cvt_nan_exp_long):
        xorq    %rax, %rax
        movd    %rax,%xmm1
        movlpdMR  .L__real_infinity+8(%rip),%xmm1
        orpd    %xmm1, %xmm0
        jmp     LBL(.Lfinal_check_exp_long)


	ELF_FUNC(__fsd_exp_long)
	ELF_SIZE(__fsd_exp_long)

/**** Here starts the main calculations  ****/
/* This is the extra precision log(x) calculation, for use in dpow */

/* Input argument comes in xmm0
 * Output arguments go in xmm0, xmm1
 */
	.text
	ALN_FUNC
ENT(__fsd_log_long):

	comisd  .L__real_mindp(%rip), %xmm0
	movd 	%xmm0, %rdx
	movq	$0x07fffffffffffffff,%rcx
	jb      LBL(.L__z_or_n_long)

	andq	%rdx, %rcx       /* rcx is ax */
	shrq	$52,%rcx
	sub	$1023,%ecx

LBL(.L__100_long):
	/* log_thresh1 = 9.39412117004394531250e-1 = 0x3fee0faa00000000
	   log_thresh2 = 1.06449508666992187500 = 0x3ff1082c00000000 */
	/* if (ux >= log_thresh1 && ux <= log_thresh2) */
#ifdef GH_TARGET
	movd %ecx, %xmm6
	cvtdq2pd %xmm6, %xmm6
#else
	cvtsi2sd %ecx,%xmm6	/* xexp */
#endif
	movq	$0x03fee0faa00000000,%r8
	cmpq	%r8,%rdx
	jb	LBL(.L__pl1_long)
	movq	$0x03ff1082c00000000,%r8
	cmpq	%r8,%rdx
	jl	LBL(.L__near_one_long)

LBL(.L__pl1_long):
	/* Store the exponent of x in xexp and put f into the range [0.5,1) */
	/* compute the index into the log tables */
	movq	$0x0000fffffffffffff, %rax
	andq	%rdx,%rax
	movq	%rax,%r8

	/* Now  x = 2**xexp  * f,  1/2 <= f < 1. */
	shrq	$45,%r8
	movq	%r8,%r9
	shr	$1,%r8d
	add	$0x040,%r8d
	and	$1,%r9d
	add	%r9d,%r8d

	/* reduce and get u */
	cvtsi2sd %r8d,%xmm1	/* convert index to float */
	movq	$0x03fe0000000000000,%rdx
	orq	%rdx,%rax
	movd 	%rax,%xmm2	/* f */

	movlpdMR	.L__real_half(%rip),%xmm5	/* .5 */
	mulsd	.L__real_3f80000000000000(%rip),%xmm1	/* f1 = index/128 */
	leaq	.L__np_ln_lead_table(%rip),%r9
	movlpdMR	 -512(%r9,%r8,8),%xmm0			/* z1 */
	subsd	%xmm1,%xmm2				/* f2 = f - f1 */
	mulsd	%xmm2,%xmm5
	addsd	%xmm5,%xmm1				/* denominator */
	divsd	%xmm1,%xmm2				/* u */

	/* solve for ln(1+u) */
	movsdRR	%xmm2,%xmm1				/* u */
	mulsd	%xmm2,%xmm2				/* u^2 */
	movlpdMR	.L__real_cb3(%rip),%xmm3
	mulsd	%xmm2,%xmm3				/* Cu2 */
	movsdRR	%xmm2,%xmm5
	mulsd	%xmm1,%xmm5				/* u^3 */
	addsd	.L__real_cb2(%rip),%xmm3 		/* B+Cu2 */
	movsdRR	%xmm2,%xmm4
	mulsd	%xmm5,%xmm4				/* u^5 */
	movlpdMR	.L__real_log2_lead(%rip),%xmm2
	mulsd	.L__real_cb1(%rip),%xmm5 		/* Au3 */
	mulsd	%xmm3,%xmm4				/* u5(B+Cu2) */
	addsd	%xmm5,%xmm4				/* Au3+u5(B+Cu2) */

	/* recombine */
	mulsd	%xmm6,%xmm2				/* xexp * log2_lead */
	addsd	%xmm2,%xmm0				/* r1,A */
	mulsd	.L__real_log2_tail(%rip),%xmm6	/* xexp * log2_tail */
	leaq	.L__np_ln_tail_table(%rip),%r9
        addsd   -512(%r9,%r8,8),%xmm4                   /* z2+=q ,C */
	addsd	%xmm4,%xmm6				/* C */

	/* redistribute the result */
	movsdRR	%xmm1,%xmm2		/* B */
	addsd	%xmm6,%xmm1		/* 0xB = B+C */
	subsd	%xmm1,%xmm2		/* -0xC = B-Bh */
	addsd	%xmm2,%xmm6		/* Ct = C-0xC */

	movsdRR	%xmm0,%xmm3
	addsd	%xmm1,%xmm0		/* H = A+0xB */
	subsd	%xmm0,%xmm3		/* -Bhead = A-H */
	addsd	%xmm3,%xmm1		/* +Btail = 0xB-Bhead */

	movsdRR	%xmm0,%xmm4
	 
	andpd	.L__real_fffffffff8000000(%rip),%xmm0	/* Head */
	subsd	%xmm0,%xmm4		/* Ht = H - Head */
	addsd	%xmm4,%xmm1		/* tail = Btail +Ht */


	addsd	%xmm6,%xmm1		/* Tail = tail + ct */
	ret

LBL(.L__near_one_long):
	/* saves 10 cycles */
	/* r = x - 1.0; */
	movlpdMR	.L__real_two(%rip),%xmm2
	subsd	.L__real_one(%rip),%xmm0	/* r */

	/* u = r / (2.0 + r); */
	addsd	%xmm0,%xmm2
	movsdRR	%xmm0,%xmm1
	divsd	%xmm2,%xmm1			/* u */
	movlpdMR	.L__real_ca4(%rip),%xmm4	/* D */
	movlpdMR	.L__real_ca3(%rip),%xmm5	/* C */
	/* correction = r * u; */
	movsdRR	%xmm0,%xmm6
	mulsd	%xmm1,%xmm6			/* correction */

	/* u = u + u; */
	addsd	%xmm1,%xmm1			/* u */
	movsdRR	%xmm1,%xmm2
	mulsd	%xmm2,%xmm2			/* v =u^2 */

	/* r2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction); */
	mulsd	%xmm1,%xmm5			/* Cu */
	movsdRR	%xmm1,%xmm3
	mulsd	%xmm2,%xmm3			/* u^3 */
	mulsd	.L__real_ca2(%rip),%xmm2	/* Bu^2 */
	mulsd	%xmm3,%xmm4			/* Du^3 */

	addsd	.L__real_ca1(%rip),%xmm2	/* +A */
	movsdRR	%xmm3,%xmm1
	mulsd	%xmm1,%xmm1			/* u^6 */
	addsd	%xmm4,%xmm5			/* Cu+Du3 */
	subsd	%xmm6,%xmm0			/*  -correction	part A */

	mulsd	%xmm3,%xmm2			/* u3(A+Bu2)	part B */
	movsdRR	%xmm0,%xmm4
	mulsd	%xmm5,%xmm1			/* u6(Cu+Du3)	part C */

	/* we now have 3 terms, develop a head and tail for the sum */

	movsdRR	%xmm2,%xmm3			/* B */
	addsd	%xmm3,%xmm0			/* H = A+B */
	subsd	%xmm0,%xmm4			/* 0xB = A - H */
	addsd	%xmm4,%xmm2			/* Bt = B-0xB */

	movsdRR	%xmm0,%xmm3			/* split the top term */
	andpd	.L__real_fffffffff8000000(%rip),%xmm0		/* Head */
	subsd	%xmm0,%xmm3			/* Ht = H - Head */
	addsd	%xmm3,%xmm2			/* Tail = Bt +Ht */

	addsd	%xmm2,%xmm1			/* Tail = tail + C */
	ret

	/* Start here for all the conditional cases */
	/* we have a zero, a negative number, denorm, or nan. */
LBL(.L__z_or_n_long):
	jp      LBL(.L__lnan)
	xorpd   %xmm1, %xmm1
	comisd  %xmm1, %xmm0
	je      LBL(.L__zero)
	jbe     LBL(.L__negative_x)

	/* A Denormal input, scale appropriately */
	mulsd   .L__real_scale(%rip), %xmm0
	movd 	%xmm0, %rdx
	movq	$0x07fffffffffffffff,%rcx
	andq	%rdx, %rcx       /* rcx is ax */
	shrq	$52,%rcx
	sub	$1075,%ecx
	jmp     LBL(.L__100_long)

        /* x == +/-0.0 */
LBL(.L__zero):
        movlpdMR  .L__real_ninf(%rip),%xmm0  /* C99 specs -inf for +-0 */
        jmp     LBL(.L__finish)

        /* x < 0.0 */
LBL(.L__negative_x):
        movlpdMR  .L__real_nan(%rip),%xmm0
        jmp     LBL(.L__finish)

        /* NaN */
LBL(.L__lnan):
        xorpd   %xmm1, %xmm1
        movlpdMR  .L__real_qnanbit(%rip), %xmm1   /* convert to quiet */
        orpd    %xmm1, %xmm0
        jmp     LBL(.L__finish)

LBL(.L__finish):
#if defined(_WIN64)
        movdqa  RZ_OFF(24)(%rsp), %xmm6
#endif

        RZ_POP
        rep
        ret

        ELF_FUNC(__fsd_log_long)
        ELF_SIZE(__fsd_log_long)


/* ============================================================ */
/* This routine takes a 4 doubles input, and produces 4 
   single precision outputs

** MAKE SURE THE STACK HERE MATCHES THE STACK IN __fvsexp

*/
	.text
        ALN_FUNC
ENT(__fvs_exp_dbl):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
#endif

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
        /* Step 1. Reduce the argument. */
        /* r = x * thirtytwo_by_logbaseof2; */
        movapd  .L__real_thirtytwo_by_log2(%rip),%xmm3
        movapd  .L__real_thirtytwo_by_log2(%rip),%xmm4
        movapd  %xmm0, %xmm2
        movapd  %xmm1, %xmm6
        mulpd   %xmm0, %xmm3
        mulpd   %xmm1, %xmm4
        andpd   .L__real_mask_unsign(%rip), %xmm2
        andpd   .L__real_mask_unsign(%rip), %xmm6

        /* Compare input with max */
        cmppd    $6, .L__dp_max_singleval(%rip), %xmm2
        cmppd    $6, .L__dp_max_singleval(%rip), %xmm6

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm3,%xmm5    /* convert to integer */
        orpd    %xmm6, %xmm2    /* Or masks together */
        cvtpd2dq %xmm4,%xmm6    /* convert to integer */
        movmskps %xmm2, %r8d
        cvtdq2pd %xmm5,%xmm3    /* and back to float. */
        cvtdq2pd %xmm6,%xmm4    /* and back to float. */
        movapd  %xmm0, %xmm2    /* Move input */
        test     $3, %r8d
#ifdef GH_TARGET
	jz      .L__fvsexp_dbl_entry_gh
#else
        jz      .L__fvsexp_dbl_entry
#endif

#define _DX0 0
#define _DX1 8
#define _DX2 16
#define _DX3 24

LBL(.L__Scalar_fvsexp_dbl):
        pushq   %rbp                    /* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _DX0(%rsp)
        movapd  %xmm1, _DX2(%rsp)

        CALL(ENT(__fss_exp_dbl))
        movss   %xmm0, _SR0(%rsp)

        movsd   _DX1(%rsp), %xmm0
        CALL(ENT(__fss_exp_dbl))
        movss   %xmm0, _SR1(%rsp)

        movsd   _DX2(%rsp), %xmm0
        CALL(ENT(__fss_exp_dbl))
        movss   %xmm0, _SR2(%rsp)

        movsd   _DX3(%rsp), %xmm0
        CALL(ENT(__fss_exp_dbl))
        movss   %xmm0, _SR3(%rsp)

        movaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp

	/* Done */
#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif

	RZ_POP
	ret

        ELF_FUNC(__fvs_exp_dbl)
        ELF_SIZE(__fvs_exp_dbl)


/* ============================================================ 
 * 
 *  Prototype:
 * 
 *      __fvdexp_long(__m128d x);
 * 
 *    Computes e raised to the x power.
 *  Does not perform error checking.   Denormal results are truncated to 0.
 * 
 */
        .text
        ALN_FUNC
ENT(__fvd_exp_long):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	%xmm0, %xmm2
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulpd	%xmm0,%xmm3 

	/* save x for later. */
	andpd	.L__real_mask_unsign(%rip), %xmm2

        /* Set n = nearest integer to r */
	cvtpd2dq %xmm3,%xmm3
	cmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2
	leaq	.L__two_to_jby32_table(%rip),%r11
	cvtdq2pd %xmm3,%xmm5
	movmskpd %xmm2, %r8d

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	movapd	.L__real_log2_by_32_lead(%rip),%xmm2
	mulpd	%xmm5,%xmm2
	movq	 %xmm3,RZ_OFF(24)(%rsp)
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdexp_long)

	/* r2 =   - n * logbaseof2_by_32_trail; */
	subpd	%xmm2,%xmm0	/* r1 in xmm0, */
	mulpd	.L__real_log2_by_32_tail(%rip),%xmm5 	/* r2 in xmm5 */
	addpd	%xmm1, %xmm5

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	mov	RZ_OFF(24)(%rsp),%ecx
	and	%ecx,%r9d

	mov	RZ_OFF(20)(%rsp),%edx
	and	%edx,%r8d
	movapd	%xmm0,%xmm2	/* xmm2 = r1 for now */
	                   	/* xmm0 = r1 for good */

	/* f1 = two_to_jby32_lead_table[j]; */
	/* f2 = two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%r9d,%ecx
	sar	$5,%ecx
	sub	%r8d,%edx
	sar	$5,%edx
	addpd	%xmm5,%xmm2    /* xmm2 = r = r1 + r2 */

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
#endif
	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	 * r*r*( 5.00000000000000008883e-01 +
	 * r*( 1.66666666665260878863e-01 +
	 * r*( 4.16666666662260795726e-02 +
	 * r*( 8.33336798434219616221e-03 +
	 * r*( 1.38889490863777199667e-03 ))))));
	 * q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movapd	%xmm2,%xmm1
	movapd	.L__real_3f56c1728d739765(%rip),%xmm3
	movapd	.L__real_3FC5555555548F7C(%rip),%xmm6

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

	mulpd	%xmm2,%xmm3	/* *x */
	mulpd	%xmm2,%xmm6	/* *x */
	mulpd	%xmm2,%xmm1	/* x*x */
	movapd	%xmm1,%xmm4

	addpd	 .L__real_3F811115B7AA905E(%rip),%xmm3
	addpd	 .L__real_3fe0000000000000(%rip),%xmm6
	mulpd	%xmm1,%xmm4	/* x^4 */
	mulpd	%xmm2,%xmm3	/* *x */

	mulpd	%xmm1,%xmm6	/* *x^2 */
	addpd	.L__real_3FA5555555545D4E(%rip),%xmm3
	addpd	%xmm5,%xmm6	/* + x  */
	mulpd	%xmm4,%xmm3	/* *x^4 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1023,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	addpd	%xmm3,%xmm6	/* q = final sum */
	addpd	%xmm6,%xmm0	/* final sum */

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	movhpd	(%r11,%r8,8),%xmm5 	/* f1 + f2 */


	mulpd	%xmm5,%xmm0
	addpd	%xmm5,%xmm0		/* z = z1 + z2 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r11		/* 1 */
	addq	$1022,%rdx		/* add bias */
	cmovleq	%rdx, %r11
	cmovleq	%rax, %rdx
	addq	$1023,%r11		/* add bias */
	shlq	$52, %r11		/* build 2^n */

	/* Step 3. Reconstitute. */
	movq	%r10,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r11,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(24)(%rsp),%xmm0  /* result*= 2^n */

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
#endif

LBL(.L__final_check_long):
	RZ_POP
	rep
	ret

#define _DT0 48
#define _DT1 56

#define _DR0 32
#define _DR1 40

LBL(.L__Scalar_fvdexp_long):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _DX0(%rsp)
        movapd  %xmm1, _DT0(%rsp)

        CALL(ENT(__fsd_exp_long))
        movsd   %xmm0, _DR0(%rsp)

        movsd   _DX1(%rsp), %xmm0
        movsd   _DT1(%rsp), %xmm1
        CALL(ENT(__fsd_exp_long))
        movsd   %xmm0, _DR1(%rsp)

        movapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check_long)

	ELF_FUNC(__fvd_exp_long)
	ELF_SIZE(__fvd_exp_long)


/* ======================================================================== */

    	.text
    	ALN_FUNC
ENT(__fvd_log_long):
	RZ_PUSH

	movdqa	%xmm0, RZ_OFF(40)(%rsp)	/* save the input values */
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm4
	pxor	%xmm1, %xmm1
	cmppd	$6, .L__real_maxfp(%rip), %xmm2
	cmppd 	$1, .L__real_mindp(%rip), %xmm4
	movdqa	%xmm0, %xmm3
	psrlq	$52, %xmm3
	orpd	%xmm2, %xmm4
	psubq	.L__mask_1023(%rip),%xmm3
	movmskpd %xmm4, %r8d
	packssdw %xmm1, %xmm3
	cvtdq2pd %xmm3, %xmm6		/* xexp */
	movdqa	%xmm0, %xmm2
	xorq	%rax, %rax
	subpd	.L__real_one(%rip), %xmm2
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdlog_long)

	movdqa	%xmm0,%xmm3
	andpd	.L__real_notsign(%rip),%xmm2
	pand	.L__real_mant(%rip),%xmm3
	movdqa	%xmm3,%xmm4
	movapd	.L__real_half(%rip),%xmm5	/* .5 */

	cmppd	$1,.L__real_threshold(%rip),%xmm2
	movmskpd %xmm2,%r10d
	cmp	$3,%r10d
	jz	LBL(.Lall_nearone_long)

	test	$3,%r10d
	jnz	LBL(.L__Scalar_fvdlog_long)

	psrlq	$45,%xmm3
	movdqa	%xmm3,%xmm2
	psrlq	$1,%xmm3
	paddq	.L__mask_040(%rip),%xmm3
	pand	.L__mask_001(%rip),%xmm2
	paddq	%xmm2,%xmm3

	packssdw %xmm1,%xmm3
	cvtdq2pd %xmm3,%xmm1
	xorq	 %rcx,%rcx
	movq	 %xmm3,RZ_OFF(24)(%rsp)

	por	.L__real_half(%rip),%xmm4
	movdqa	%xmm4,%xmm2
	mulpd	.L__real_3f80000000000000(%rip),%xmm1	/* f1 = index/128 */

	leaq	.L__np_ln_lead_table(%rip),%rdx
	mov	RZ_OFF(24)(%rsp),%eax
	mov	RZ_OFF(20)(%rsp),%ecx

	subpd	%xmm1,%xmm2				/* f2 = f - f1 */
	mulpd	%xmm2,%xmm5
	addpd	%xmm5,%xmm1

	divpd	%xmm1,%xmm2				/* u */

	movlpdMR	 -512(%rdx,%rax,8),%xmm0		/* z1 */
	movhpd	 -512(%rdx,%rcx,8),%xmm0		/* z1 */


	movapd	%xmm2,%xmm1				/* u */
	mulpd	%xmm2,%xmm2				/* u^2 */
	movapd	%xmm2,%xmm5
	movapd	.L__real_cb3(%rip),%xmm3
	mulpd	%xmm2,%xmm3				/* Cu2 */
	mulpd	%xmm1,%xmm5				/* u^3 */
	addpd	.L__real_cb2(%rip),%xmm3 		/* B+Cu2 */

	mulpd	%xmm5,%xmm2				/* u^5 */
	movapd	.L__real_log2_lead(%rip),%xmm4
	mulpd	.L__real_cb1(%rip),%xmm5 		/* Au3 */
	mulpd	%xmm3,%xmm2				/* u5(B+Cu2) */
	addpd	%xmm5,%xmm2				/* u+Au3 */

	/* table lookup */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	movlpdMR	 -512(%rdx,%rax,8),%xmm3		/* z2+=q */
	movhpd	 -512(%rdx,%rcx,8),%xmm3		/* z2+=q */


	/* recombine */
	mulpd	%xmm6,%xmm4				/* xexp * log2_lead */
	addpd	%xmm4,%xmm0				/* r1 */
	mulpd	.L__real_log2_tail(%rip),%xmm6
	addpd	%xmm3, %xmm2
	addpd	%xmm2, %xmm6

	/* redistribute the result */
	movapd	%xmm1,%xmm2		/* B */
	addpd	%xmm6,%xmm1		/* 0xB = B+C */
	subpd	%xmm1,%xmm2		/* -0xC = B-Bh */
	addpd	%xmm2,%xmm6		/* Ct = C-0xC */

	movapd	%xmm0,%xmm3
	addpd	%xmm1,%xmm0		/* H = A+0xB */
	subpd	%xmm0,%xmm3		/* -Bhead = A-H */
	addpd	%xmm3,%xmm1		/* +Btail = 0xB-Bhead */

	movapd	%xmm0,%xmm4
	 
	andpd	.L__real_fffffffff8000000(%rip),%xmm0	/* Head */
	subpd	%xmm0,%xmm4		/* Ht = H - Head */
	addpd	%xmm4,%xmm1		/* tail = Btail +Ht */
	addpd	%xmm6,%xmm1		/* Tail = tail + ct */

LBL(.Lfinishn1_long):
	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.Lall_nearone_long):
	movapd	.L__real_two(%rip),%xmm2
	subpd	.L__real_one(%rip),%xmm0	/* r */

	addpd	%xmm0,%xmm2
	movapd	%xmm0,%xmm1
	divpd	%xmm2,%xmm1			/* u */
	movapd	.L__real_ca4(%rip),%xmm4  	/* D */
	movapd	.L__real_ca3(%rip),%xmm5 	/* C */

	movapd	%xmm0,%xmm6
	mulpd	%xmm1,%xmm6			/* correction */

	addpd	%xmm1,%xmm1			/* u */
	movapd	%xmm1,%xmm2
	mulpd	%xmm2,%xmm2			/* v =u^2 */

	mulpd	%xmm1,%xmm5			/* Cu */
	movapd	%xmm1,%xmm3
	mulpd	%xmm2,%xmm3			/* u^3 */
	mulpd	.L__real_ca2(%rip),%xmm2	/* Bu^2 */
	mulpd	%xmm3,%xmm4			/* Du^3 */

	addpd	.L__real_ca1(%rip),%xmm2	/* +A */
	movapd	%xmm3,%xmm1
	mulpd	%xmm1,%xmm1			/* u^6 */
	addpd	%xmm4,%xmm5			/* Cu+Du3 */
	subpd	%xmm6,%xmm0			/*  -correction	part A */

	mulpd	%xmm3,%xmm2			/* u3(A+Bu2)	part B */
	movapd	%xmm0,%xmm4
	mulpd	%xmm5,%xmm1			/* u6(Cu+Du3)	part C */

	/* we now have 3 terms, develop a head and tail for the sum */

	movapd	%xmm2,%xmm3			/* B */
	addpd	%xmm3,%xmm0			/* H = A+B */
	subpd	%xmm0,%xmm4			/* 0xB = A - H */
	addpd	%xmm4,%xmm2			/* Bt = B-0xB */

	movapd	%xmm0,%xmm3			/* split the top term */
	andpd	.L__real_fffffffff8000000(%rip),%xmm0		/* Head */
	subpd	%xmm0,%xmm3			/* Ht = H - Head */
	addpd	%xmm3,%xmm2			/* Tail = Bt +Ht */
	addpd	%xmm2,%xmm1			/* Tail = tail + C */
	jmp	LBL(.Lfinishn1_long)

#define _X0 0
#define _X1 8

#define _T0 48
#define _T1 56

#define _R0 32
#define _R1 40

LBL(.L__Scalar_fvdlog_long):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _X0(%rsp)

        CALL(ENT(__fsd_log_long))
        movsd   %xmm0, _R0(%rsp)
        movsd   %xmm1, _T0(%rsp)

        movsd   _X1(%rsp), %xmm0
        CALL(ENT(__fsd_log_long))
        movsd   %xmm0, _R1(%rsp)
        movsd   %xmm1, _T1(%rsp)

        movapd  _R0(%rsp), %xmm0
        movapd  _T0(%rsp), %xmm1
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.Lfinishn1_long)

        ELF_FUNC(__fvd_log_long)
        ELF_SIZE(__fvd_log_long)




#else

/* ============================================================
 *  fastexpf.s
 * 
 *  An implementation of the expf libm function.
 * 
 *  Prototype:
 * 
 *      float fastexpf(float x);
 * 
 *    Computes e raised to the x power.
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fss_exp))
	.globl	ENT_GH(__fmth_i_exp)
IF_GH(ENT(__fss_exp):)
ENT_GH(__fmth_i_exp):
	RZ_PUSH

        comiss         %xmm0, %xmm0
        jp     LBL(.LB_NZERO_SS)

        comiss .L__np_ln_lead_table(%rip), %xmm0        /* Equal to 0.0? */
        jne     LBL(.LB_NZERO_SS)
	movss .L4_386(%rip), %xmm0
        RZ_POP
        rep
	ret

LBL(.LB_NZERO_SS):


        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
#ifdef GH_TARGET
	unpcklps %xmm0, %xmm0
	cvtps2pd %xmm0, %xmm2
#else
	cvtss2sd %xmm0, %xmm2
#endif
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm2,%xmm3 

	/* Set n = nearest integer to r */
	comiss	.L__sp_ln_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_inf)
	comiss	.L_real_min_singleval(%rip), %xmm0
	jbe	LBL(.L_sp_ninf)

#ifdef GH_TARGET
.L__fmth_i_exp_dbl_entry_gh:
#else
.L__fmth_i_exp_dbl_entry:
#endif
	cvtpd2dq %xmm3,%xmm4	/* convert to integer */
	cvtdq2pd %xmm4,%xmm1	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
	mulsd	.L__real_log2_by_32(%rip),%xmm1
	movd	%xmm4,%ecx
	subsd	%xmm1,%xmm2	/* r1 in xmm2, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* j = n & 0x0000001f; */
	movq	$0x1f,%rax
	and	%ecx,%eax

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%eax,%ecx
	sar	$5,%ecx

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm1
	movsdRR	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	mulsd	%xmm2,%xmm2
	addsd	.L__real_3fe0000000000000(%rip),%xmm1
	mulsd	%xmm1,%xmm2
	movlpdMR	(%rdx,%rax,8),%xmm4
	addsd	%xmm0,%xmm2

	/* *z2 = f2 + ((f1 + f2) * q); */
        add	$1023, %ecx	/* add bias */
	/* deal with infinite results */
	/* deal with denormal results */
	mulsd	%xmm4,%xmm2
        shlq	$52,%rcx        /* build 2^n */
	addsd	%xmm4,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm2	/* result *= 2^n */
#ifdef GH_TARGET
	unpcklpd %xmm2, %xmm2
	cvtpd2ps %xmm2, %xmm0
#else
	cvtsd2ss %xmm2,%xmm0
#endif

LBL(.L_sp_final_check):
	RZ_POP
	rep
	ret

LBL(.L_sp_inf):
	movlps	.L_sp_real_infinity(%rip),%xmm0
	jmp	LBL(.L_sp_final_check)

LBL(.L_sp_ninf):
	jp	LBL(.L_sp_cvt_nan)
	xor	%eax, %eax
	movd	%eax,%xmm0
	jmp	LBL(.L_sp_final_check)

LBL(.L_sp_sinh_ninf):
        jp      LBL(.L_sp_cvt_nan)
        movlps  .L_sp_real_ninfinity(%rip),%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_cosh_ninf):
        jp      LBL(.L_sp_cvt_nan)
        movlps  .L_sp_real_infinity(%rip),%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_cvt_nan):
	xor	%eax, %eax
	movd	%eax,%xmm1
	movlpdMR	.L_real_cvt_nan(%rip),%xmm1
	orps	%xmm1, %xmm0
	jmp	LBL(.L_sp_final_check)

        ELF_FUNC(ENT_GH(__fmth_i_exp))
        ELF_SIZE(ENT_GH(__fmth_i_exp))
        IF_GH(ELF_FUNC(__fss_exp))
        IF_GH(ELF_SIZE(__fss_exp))


/* ============================================================
 *  fastexp.s
 * 
 *  An implementation of the exp libm function.
 * 
 *  Prototype:
 * 
 *      double fastexp(double x);
 * 
 *    Computes e raised to the x power.
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fsd_exp))
	.globl ENT_GH(__fmth_i_dexp)
IF_GH(ENT(__fsd_exp):)
ENT_GH(__fmth_i_dexp):
	RZ_PUSH

        comisd         %xmm0, %xmm0
        jp     LBL(.LB_NZERO_SD)

        comisd .L__np_ln_lead_table(%rip), %xmm0        /* Equal to 0.0? */
        jne     LBL(.LB_NZERO_SD)
        movsd  .L__two_to_jby32_table(%rip), %xmm0
        RZ_POP
        rep
        ret

LBL(.LB_NZERO_SD):


        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm0,%xmm3 

	/* Set n = nearest integer to r */
	comisd	.L__real_ln_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	comisd	.L__real_ln_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_ninf)
	cvtpd2dq %xmm3,%xmm4	/* convert to integer */
	cvtdq2pd %xmm4,%xmm1	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
	movlpdMR	.L__real_log2_by_32_lead(%rip),%xmm2
	mulsd	%xmm1,%xmm2
	movd	%xmm4,%ecx
	subsd	%xmm2,%xmm0	/* r1 in xmm0, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
	mulsd	.L__real_log2_by_32_tail(%rip),%xmm1

	/* j = n & 0x0000001f; */
	movq	$0x1f,%rax
	andl	%ecx,%eax
	movsdRR	%xmm0,%xmm2

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%eax,%ecx
	sarl	$5,%ecx
	addsd	%xmm1,%xmm2    /* r = r1 + r2 */

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movsdRR	%xmm2,%xmm1
	movlpdMR	.L__real_3f56c1728d739765(%rip),%xmm3
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm0
	mulsd	%xmm2,%xmm3
	mulsd	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	movsdRR	%xmm1,%xmm4
	addsd	.L__real_3F811115B7AA905E(%rip),%xmm3
	addsd	.L__real_3fe0000000000000(%rip),%xmm0
	mulsd	%xmm1,%xmm4
	mulsd	%xmm2,%xmm3
	mulsd	%xmm1,%xmm0
	addsd	.L__real_3FA5555555545D4E(%rip),%xmm3
	addsd	%xmm2,%xmm0
	mulsd	%xmm4,%xmm3
	addsd	%xmm3,%xmm0

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%rdx,%rax,8),%xmm5
	/* deal with infinite results */
        movslq	%ecx,%rcx
	mulsd	%xmm5,%xmm0
	addsd	%xmm5,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* deal with denormal results */
	movq	$1, %rdx
	movq	$1, %rax
        addq	$1022, %rcx	/* add bias */
	cmovleq	%rcx, %rdx
	cmovleq	%rax, %rcx
        shlq	$52,%rcx        /* build 2^n */
        addq	$1023, %rdx	/* add bias */
        shlq	$52,%rdx        /* build 2^n */
	movq	%rdx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

LBL(.Lfinal_check):
	RZ_POP
	rep
	ret
LBL(.L_inf):
	movlpdMR	.L__real_infinity(%rip),%xmm0
	jmp	LBL(.Lfinal_check)

LBL(.L_ninf):
	jp	LBL(.L_cvt_nan)
	xorq	%rax, %rax
	movd 	%rax,%xmm0
	jmp	LBL(.Lfinal_check)

LBL(.L_sinh_ninf):
        jp      LBL(.L_cvt_nan)
        movlpdMR  .L__real_ninfinity(%rip),%xmm0
        jmp     LBL(.Lfinal_check)

LBL(.L_cosh_ninf):
        jp      LBL(.L_cvt_nan)
        movlpdMR  .L__real_infinity(%rip),%xmm0
        jmp     LBL(.Lfinal_check)

LBL(.L_cvt_nan):
	xorq	%rax, %rax
	movd 	%rax,%xmm1
	movlpdMR	.L__real_infinity+8(%rip),%xmm1
	orpd	%xmm1, %xmm0
	jmp	LBL(.Lfinal_check)


        ELF_FUNC(ENT_GH(__fmth_i_dexp))
        ELF_SIZE(ENT_GH(__fmth_i_dexp))
	IF_GH(ELF_FUNC(__fsd_exp))
	IF_GH(ELF_SIZE(__fsd_exp))


/* ------------------------------------------------------------------------- */

/*
 *	#ifdef GH_TARGET
 *	double complex __fsz_exp_c99(double complex carg)
 *	double complex __mth_i_cdexp_gh_c99(double complex carg)
 *	#else
 *	double complex __mth_i_cdexp_c99(double complex carg)
 *	#endif
 *
 *	Allocate dcmplx_t structure on stack and then call:
 *	void __mth_i_cdexp(dcmplx_t *, creal(carg), cimag(carg))
 *
 *	dcmplx_t defined as:
 *	typedef struct {
 *		double	real;
 *		double	imag;
 *	} dcmplx_t;
 *
 *	Linux86-64/OSX64:
 *		Entry:
 *		(%xmm0)		REAL(carg)
 *		(%xmm1)		IMAG(carg)
 *		Exit:
 *		(%xmm0)		REAL(exp(carg))
 *		(%xmm1)		IMAG(exp(carg))
 *	Windows:
 *		Entry:
 *		(%rcx)		pointer to return struct
 *		(%xmm1)		REAL(carg)
 *		(%xmm2)		IMAG(carg)
 *		Exit:
 *		0(%rcx)		REAL(exp(carg))
 *		8(%rcx)		IMAG(exp(carg))
 */

        .text
        ALN_FUNC
#ifdef GH_TARGET
        .globl ENT(ASM_CONCAT(__fsz_exp,__MTH_C99_CMPLX_SUFFIX))
        .globl ENT(ASM_CONCAT(__mth_i_cdexp_gh,__MTH_C99_CMPLX_SUFFIX))
ENT(ASM_CONCAT(__fsz_exp,__MTH_C99_CMPLX_SUFFIX)):
ENT(ASM_CONCAT(__mth_i_cdexp_gh,__MTH_C99_CMPLX_SUFFIX)):
#else
        .globl ENT(ASM_CONCAT(__mth_i_cdexp,__MTH_C99_CMPLX_SUFFIX))
ENT(ASM_CONCAT(__mth_i_cdexp,__MTH_C99_CMPLX_SUFFIX)):
#endif

#ifdef	_WIN64
	/*
	 * Return structure in (%rcx).
	 * Will be managed by macro I1.
	 */
	jmp	LBL(.L__fsz_exp_win64)
#else
	subq	$24,%rsp
	movq	%rsp,%rdi		/* Allocate valid dcmplx_t structure on stack */
#ifdef GH_TARGET
	CALL(ENT(__mth_i_cdexp_gh))
#else
	CALL(ENT(__mth_i_cdexp))
#endif

	movsd	(%rsp), %xmm0		/* Real */
	movsd	8(%rsp), %xmm1		/* Imaginary */
	addq	$24,%rsp
        ret
#endif

#ifdef GH_TARGET
        ELF_FUNC(ASM_CONCAT(__mth_i_cdexp_gh,__MTH_C99_CMPLX_SUFFIX))
        ELF_SIZE(ASM_CONCAT(__mth_i_cdexp_gh,__MTH_C99_CMPLX_SUFFIX))
        ELF_FUNC(ASM_CONCAT(__fsz_exp,__MTH_C99_CMPLX_SUFFIX))
        ELF_SIZE(ASM_CONCAT(__fsz_exp,__MTH_C99_CMPLX_SUFFIX))
#else
        ELF_FUNC(ASM_CONCAT(__mth_i_cdexp,__MTH_C99_CMPLX_SUFFIX))
        ELF_SIZE(ASM_CONCAT(__mth_i_cdexp,__MTH_C99_CMPLX_SUFFIX))
#endif

/*
 *	double complex __fsz_exp_1v(%xmm0-pd)
 *
 *	compute double precision complex EXP(real + I*imag) with
 *	arguments stored as packed double in %xmm0.
 *
 *	Linux86-64/OSX64 - ONLY!
 *	Entry:
 *	(%xmm0-lower)	REAL(carg)
 *	(%xmm0-upper)	IMAG(carg)
 *
 *	Exit:
 *	(%xmm0-lower)	REAL(cexp(carg))
 *	(%xmm0-upper)	IMAG(cexp(carg))
 */
	.text
	ALN_FUNC
        IF_GH(.globl ENT(__fsz_exp_1v))
        .globl ENT_GH(__mth_i_cdexp_1v)
IF_GH(ENT(__fsz_exp_1v):)
ENT_GH(__mth_i_cdexp_1v):
	subq	$24,%rsp
	movq	%rsp,%rdi	# (%rdi) = return structure
	movhlps	%xmm0,%xmm1	# (%xmm1) = imag

#ifdef GH_TARGET
	CALL(ENT(__fsz_exp))
#else
	CALL(ENT(__mth_i_cdexp))
#endif

	movapd	(%rsp),%xmm0	# (%xmm0) = real + I*imag
	addq	$24,%rsp
	ret

        IF_GH(ELF_FUNC(__fsz_exp_1v))
        IF_GH(ELF_SIZE(__fsz_exp_1v))
        ELF_FUNC(ENT_GH(__mth_i_cdexp_1v))
        ELF_SIZE(ENT_GH(__mth_i_cdexp_1v))

/*
 *	#ifdef	GH_TARGET
 *	void __fsz_exp(dcmplx_t *, double real, double imag)
 *	void __mth_i_cdexp_gh(dcmplx_t *, double real, double imag)
 *	#else
 *	void __mth_i_cdexp(dcmplx_t *, double real, double imag)
 *	#endif
 *
 *	compute double precision complex EXP(real + I*imag)
 *
 *	Return:
 *	0(%rdi) = real
 *	8(%rdi) = imag
 */

        .text
#       ALN_FUNC
        IF_GH(.globl ENT(__fsz_exp))
        .globl ENT_GH(__mth_i_cdexp)
IF_GH(ENT(__fsz_exp):)
ENT_GH(__mth_i_cdexp):

#if defined(_WIN64) 
	/*
	 *	WIN64 ONLY:
	 *	Jump entry point into routine from __fsz_exp_c99.
	 */
LBL(.L__fsz_exp_win64):
	movsdRR   %xmm1, %xmm0
	movsdRR   %xmm2, %xmm1
#endif

	comisd	.L__real_ln_max_doubleval1(%rip), %xmm0  /* compare to max */
	ja	LBL(.L_cdexp_inf)
        movd    %xmm1, %rax                             /* Move imag to gp */
	shufpd	$0, %xmm0, %xmm1                        /* pack real & imag */

        mov     $0x03fe921fb54442d18,%rdx
        movapd  .L__dble_cdexp_by_pi(%rip),%xmm4        /* For exp & sincos */
        andq    .L__real_mask_unsign(%rip), %rax       /* abs(imag) in gp */
	comisd	.L__real_ln_min_doubleval1(%rip), %xmm0  /* compare to min */
	jbe	LBL(.L_cdexp_ninf)

        cmpq    %rdx,%rax
        jle     LBL(.L__fmth_cdexp_shortcuts)

        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     LBL(.L__fmth_cdexp_hard)

        /* Step 1. Reduce the argument x. */
        /* For sincos, the closest integer to 16x / pi */
        /* For exp, the closest integer to x * 32 / ln(2) */
        mulpd   %xmm1,%xmm4                             /* Mpy to scale both */

        RZ_PUSH
#if defined(_WIN64)
        movdqa  %xmm6, RZ_OFF(40)(%rsp)
        movdqa  %xmm7, RZ_OFF(56)(%rsp)
        movdqa  %xmm8, RZ_OFF(72)(%rsp)
        movdqa  %xmm9, RZ_OFF(88)(%rsp)
	movq	I1,RZ_OFF(104)(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5                          /* convert to integer */
        movlpdMR  .L__dble_pi_by_16_ms(%rip), %xmm9
	movhpd  .L__cdexp_log2_by_32_lead(%rip), %xmm9

        movlpdMR  .L__dble_pi_by_16_ls(%rip), %xmm2
	movhpd  .L__cdexp_log2_by_32_tail(%rip), %xmm2

        movlpdMR  .L__dble_pi_by_16_us(%rip), %xmm3
        cvtdq2pd %xmm5,%xmm4                          /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulpd   %xmm4,%xmm9     /* n * p1 */
        mulpd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
	movq	%rcx,%r8      /* Save in r8 */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        movsdRR   %xmm1,%xmm6     /* x in xmm6 */
        subpd   %xmm9,%xmm1     /* x - n * p1 == rh */
        subsd   %xmm9,%xmm6     /* x - n * p1 == rh == c */

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subpd   %xmm2,%xmm1     /* rh = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */
        
        subsd   %xmm1,%xmm6     /* (c - rh) */
        movsdRR   %xmm1,%xmm9     /* Move rh */
        movsdRR   %xmm1,%xmm4     /* Move rh */
        movapd  %xmm1,%xmm8
        subsd   %xmm2,%xmm6     /* ((c - rh) - rt) */

        movlpdMR  .L__real_3FC5555555548F7C(%rip),%xmm0
        movhpd  .L__real_3f56c1728d739765(%rip),%xmm0

        movsdRR   %xmm1,%xmm5     /* Move rh */

        movlpdMR  .L__real_3fe0000000000000(%rip),%xmm7
        movhpd  .L__real_3F811115B7AA905E(%rip),%xmm7

	shufpd  $3, %xmm8, %xmm8

        subsd   %xmm6,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        movsdRR   %xmm9,%xmm2     /* Move rh */
	mulpd   %xmm8,%xmm0     /* r/720, r/6 */

        subsd   %xmm3,%xmm1     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm4     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm5     /* c = rh - rt aka r */
	addpd   %xmm7,%xmm0     /* r/720 + 1/120, r/6 + 1/2 */
	mulsd   %xmm8,%xmm8     /* r, r^2 */
        subsd   %xmm1,%xmm9     /* (rh - c) */

        mulpd   %xmm1,%xmm1     /* r^2 in both halves */
        movsdRR   %xmm4,%xmm6     /* r in xmm6 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm5,%xmm7     /* r in xmm7 */
        mulsd   %xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm1, xmm4, xmm5 have r^2, xmm9, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulsd   .L__dble_pq4(%rip), %xmm1     /* p4 * r^2 */
        subsd   %xmm6,%xmm2                   /* (rh - c) */
        mulsd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        subsd   %xmm3,%xmm9                   /* (rh - c) - rt aka rr */
	shufpd  $0, %xmm8, %xmm5              /* r^2 in both halves */

	mulpd   %xmm8, %xmm0    /* r^2/720 + r/120, r^3/6 + r^2/2 */
	shufpd  $1, %xmm8, %xmm8              /* r^2, r */

        addsd   .L__dble_pq3(%rip), %xmm1     /* + p3 */
        addsd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        subsd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */

        mulpd   %xmm5,%xmm1                   /* r^4, (p4 * r^2 + p3) * r^2 */
        mulsd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        mulsd   %xmm5,%xmm7                   /* xmm7 = r^3 */

        movhpd  .L__real_3FA5555555545D4E(%rip),%xmm8
        movsdRR   %xmm9,%xmm3                   /* Move rr */
        mulsd   %xmm5,%xmm9                   /* r * r * rr */

        addsd   .L__dble_pq2(%rip), %xmm1     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        mulsd   .L__dble_pq1+16(%rip), %xmm9  /* r * r * rr * 0.5 */
        mulsd   %xmm6, %xmm3                  /* r * rr */
	addpd   %xmm8, %xmm0    /* r^2/720 + r/120 + 1/24, r^3/6 + r^2/2 + r */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        mulsd   %xmm5,%xmm1                   /* * r^2 */
        mulsd   %xmm5,%xmm4                   /* * r^2 */
        addsd   %xmm9,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        movlpdMR  8(%rdx,%rax,8),%xmm8          /* ds2 in xmm8 */
        movlpdMR  8(%rdx,%rcx,8),%xmm9          /* dc2 in xmm9 */
        /* xmm1 has dp, xmm4 has dq,
           xmm9 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3
           xmm8 is ds2 */

        addsd   .L__dble_pq1(%rip), %xmm1     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulsd   %xmm7,%xmm1                   /* * r^3 */
        mulsd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addsd   %xmm2,%xmm1                   /* + cs  == dp aka p(r) */
        subsd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */
        movsdRR   %xmm9,%xmm3                   /* dc2 in xmm3 */
        movlpdMR   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        movlpdMR   (%rdx,%rcx,8),%xmm7          /* dc1 in xmm7 */
        addsd   %xmm6,%xmm1                   /* + r   == dp aka p(r) */
        movsdRR   %xmm8,%xmm2                   /* ds2 in xmm2 */

        mulsd   %xmm4,%xmm8                   /* ds2 * dq */
        mulsd   %xmm4,%xmm9                   /* dc2 * dq */
	movhpd  (%rdx),%xmm2                  /* high half is 1.0 */
        movq    %r8, %rcx

        addsd   %xmm2,%xmm8                   /* ds2 + ds2*dq */
        addsd   %xmm3,%xmm9                   /* dc2 + dc2*dq */
        shrq    $32, %rcx
        leaq    .L__two_to_jby32_table(%rip),%rdx

        mulsd   %xmm1,%xmm3                   /* dc2 * dp */
        mulsd   %xmm1,%xmm2                   /* ds2 * dp */
        movq    $0x1f,%rax
	movsdRR	%xmm4,%xmm6                   /* xmm6 = dq */
        andl    %ecx,%eax

        addsd   %xmm3,%xmm8                   /* (ds2 + ds2*dq) + dc2*dp */
        subsd   %xmm2,%xmm9                   /* (dc2 + dc2*dq) - ds2*dp */

        subl    %eax,%ecx

	movsdRR	%xmm5,%xmm3                   /* xmm3 = ds1 */
	shufpd  $3, %xmm1,%xmm2               /* r^4, 1.0 */
        sarl    $5,%ecx

        mulsd   %xmm5,%xmm4                   /* ds1 * dq */
        mulsd   %xmm1,%xmm5                   /* ds1 * dp */
	mulsd   %xmm7,%xmm6                   /* dc1 * dq */

        mulsd   %xmm7,%xmm1                   /* dc1 * dp */
        addsd   %xmm4,%xmm8                   /* ((ds2...) + dc2*dp) + ds1*dq */
        subsd   %xmm5,%xmm9                   /* (() - ds2*dp) - ds1*dp */

	mulpd   %xmm0,%xmm2          /* r^6/720+r^5/120+r^4/24, r^3/6+r^2/2+r */
        movlpdMR  (%rdx,%rax,8),%xmm5
        addsd   %xmm3,%xmm8                   /* + ds1 */
        addsd   %xmm6,%xmm9                   /* + dc1*dq */

        shufpd  $1, %xmm2, %xmm2
        addsd   %xmm8,%xmm1                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        addsd   %xmm7,%xmm9                   /* cos(x) = (C + Cq(r)) + Sq(r) */

        /* Now start exp */

        addsd   %xmm2, %xmm0

        /* Step 2. Compute the polynomial. */
        /* q = r1 + (r2 +
           r*r*( 5.00000000000000008883e-01 +
           r*( 1.66666666665260878863e-01 +
           r*( 4.16666666662260795726e-02 +
           r*( 8.33336798434219616221e-03 +
           r*( 1.38889490863777199667e-03 ))))));
           q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */

        /* *z2 = f2 + ((f1 + f2) * q); */

        /* deal with infinite results */
        movslq  %ecx,%rcx
        mulsd   %xmm5,%xmm0
        addsd   %xmm5,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */

        /* deal with denormal results */
        movq    $1, %rdx
        movq    $1, %rax
        addq    $1022, %rcx     /* add bias */
        cmovleq %rcx, %rdx
        cmovleq %rax, %rcx
        shlq    $52,%rcx        /* build 2^n */
        addq    $1023, %rdx     /* add bias */
        shlq    $52,%rdx        /* build 2^n */
        movq    %rdx,RZ_OFF(24)(%rsp)   /* get 2^n to memory */
        mulsd   RZ_OFF(24)(%rsp),%xmm0  /* result *= 2^n */

        /* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
        /* Step 3. Reconstitute. */
        movq    %rcx,RZ_OFF(24)(%rsp)   /* get 2^n to memory */
        mulsd   RZ_OFF(24)(%rsp),%xmm0  /* result *= 2^n */
        mulsd   %xmm0,%xmm1
        mulsd   %xmm9,%xmm0

#if defined(_WIN64)
	movq	RZ_OFF(104)(%rsp),I1
        movdqa  RZ_OFF(40)(%rsp),%xmm6
        movdqa  RZ_OFF(56)(%rsp),%xmm7
        movdqa  RZ_OFF(72)(%rsp),%xmm8
        movdqa  RZ_OFF(88)(%rsp),%xmm9
#endif
	movlpd  %xmm1,8(I1)
	movlpd  %xmm0,(I1)

        RZ_POP
        ret

LBL(.L__fmth_cdexp_shortcuts):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $32, %rsp

	movapd	%xmm1,(%rsp)
	movq	%rax, 16(%rsp)
	movq	I1, 24(%rsp)

#ifdef GH_TARGET
	CALL(ENT(__fsd_exp))
#else
	CALL(ENT(__fmth_i_dexp))
#endif

	movsdRR	%xmm0,%xmm5
	movlpdMR  (%rsp),%xmm0
        movlpdMR  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
	movq	16(%rsp),%rax
        movsdRR   %xmm0,%xmm2
        movsdRR   %xmm0,%xmm3
        shrq    $48,%rax
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fmth_cdexp_small)
        movsdRR   %xmm0,%xmm4
        mulsd   %xmm0,%xmm0
        mulsd   %xmm2,%xmm2
        mulsd   %xmm4,%xmm4

        mulsd   .L__dble_dsin_c6(%rip),%xmm0    /* x2 * s6 */
        mulsd   .L__dble_dcos_c6(%rip),%xmm2    /* x2 * c6 */
        addsd   .L__dble_dsin_c5(%rip),%xmm0    /* + s5 */
        addsd   .L__dble_dcos_c5(%rip),%xmm2    /* + c5 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s5 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dsin_c4(%rip),%xmm0    /* + s4 */
        addsd   .L__dble_dcos_c4(%rip),%xmm2    /* + c4 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s4 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + s3 */
        addsd   .L__dble_dcos_c3(%rip),%xmm2    /* + c3 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s3 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + s2 */
        addsd   .L__dble_dcos_c2(%rip),%xmm2    /* + c2 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s2 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c2 + ...) */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + s1 */
        addsd   .L__dble_dcos_c1(%rip),%xmm2    /* + c1 */
        mulsd   %xmm4,%xmm0                     /* x3 * (s1 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c1 + ...) */
        mulsd   %xmm3,%xmm0                     /* x3 */
        addsd   .L__dble_pq1+16(%rip),%xmm2     /* - 0.5 */
        mulsd   %xmm4,%xmm2                     /* x2 * (0.5 + ...) */
        addsd   %xmm3,%xmm0                     /* x + x3 * (...) done */
        addsd   %xmm2,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
	jmp	LBL(.L__fmth_cdexp_done1)

LBL(.L__fmth_cdexp_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fmth_cdexp_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_pq1(%rip),%xmm3
        mulsd   %xmm2,%xmm3
        mulsd   .L__dble_pq1+16(%rip),%xmm2
        addsd   %xmm3,%xmm0
        addsd   %xmm2,%xmm1

LBL(.L__fmth_cdexp_done1):
	movq	24(%rsp),I1
	mulsd	%xmm5,%xmm0
	mulsd	%xmm5,%xmm1
	movlpd  %xmm0,8(I1)
	movlpd  %xmm1,(I1)
	movq	%rbp, %rsp
	popq	%rbp
        ret

LBL(.L_cdexp_inf):
	movlpdMR	.L__real_infinity(%rip), %xmm0
	movlpd  %xmm0,8(I1)
	movlpd  %xmm0,(I1)
        ret

LBL(.L_cdexp_ninf):
	jp	LBL(.L_cdexp_cvt_nan)
	xorq	%rax, %rax
	movq	%rax, 8(I1)
	movq	%rax, (I1)
        ret

LBL(.L_cdexp_cvt_nan):
	movlpdMR	.L__real_infinity+8(%rip), %xmm1
	orpd	%xmm1, %xmm0
	movlpd  %xmm0,8(I1)
	movlpd  %xmm0,(I1)
        ret

LBL(.L__fmth_cdexp_hard):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $32, %rsp

	movlpd	%xmm1,(%rsp)
	movq	I1, 24(%rsp)

#ifdef GH_TARGET
	CALL(ENT(__fsd_exp))
#else
	CALL(ENT(__fmth_i_dexp))
#endif

	movlpd	%xmm0,8(%rsp)
	movlpdMR  (%rsp),%xmm0
	CALL(ENT(__mth_i_dsincos))

        movlpdMR  8(%rsp), %xmm5
        jmp LBL(.L__fmth_cdexp_done1)

        ELF_FUNC(ENT_GH(__mth_i_cdexp))
        ELF_SIZE(ENT_GH(__mth_i_cdexp))
        IF_GH(ELF_FUNC(__fsz_exp))
        IF_GH(ELF_SIZE(__fsz_exp))


/* ============================================================
 *  vector fastexpf.s
 * 
 *  An implementation of the expf libm function.
 * 
 *  Prototype:
 * 
 *      float fastexpf(float x);
 * 
 *    Computes e raised to the x power.
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fvs_exp))
	.globl ENT_GH(__fvsexp)
IF_GH(ENT(__fvs_exp):)
ENT_GH(__fvsexp):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
#endif

	/* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movhlps  %xmm0, %xmm1
	movaps	 %xmm0, %xmm5
	cvtps2pd %xmm0, %xmm2			/* xmm2 = dble(a(2)), dble(a(1)) */
	cvtps2pd %xmm1, %xmm1			/* xmm1 = dble(a(4)), dble(a(3)) */
	andps	 .L__ps_mask_unsign(%rip), %xmm5
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm4
	cmpps	$6, .L__sp_ln_max_singleval(%rip), %xmm5
	mulpd	%xmm2, %xmm3 
	mulpd	%xmm1, %xmm4 
	movmskps %xmm5, %r8d

	/* Set n = nearest integer to r */
	cvtpd2dq %xmm3,%xmm5	/* convert to integer */
	cvtpd2dq %xmm4,%xmm6	/* convert to integer */
	test	 $15, %r8d
	cvtdq2pd %xmm5,%xmm3	/* and back to float. */
	cvtdq2pd %xmm6,%xmm4	/* and back to float. */
	jnz	LBL(.L__Scalar_fvsexp)

#ifdef GH_TARGET
.L__fvsexp_dbl_entry_gh:
#else
.L__fvsexp_dbl_entry:
#endif
	/* r1 = x - n * logbaseof2_by_32_lead; */
	mulpd	.L__real_log2_by_32(%rip),%xmm3
	mulpd	.L__real_log2_by_32(%rip),%xmm4
	movq	%xmm5,RZ_OFF(16)(%rsp)
	movq	%xmm6,RZ_OFF(24)(%rsp)
	subpd	%xmm3,%xmm2	/* r1 in xmm2, */
	subpd	%xmm4,%xmm1	/* r1 in xmm1, */
	leaq	.L__two_to_jby32_table(%rip),%rax

	/* j = n & 0x0000001f; */
	mov	RZ_OFF(12)(%rsp),%r8d
	mov	RZ_OFF(16)(%rsp),%r9d
	mov	RZ_OFF(20)(%rsp),%r10d
	mov	RZ_OFF(24)(%rsp),%r11d
	movq	$0x1f, %rcx
	and 	%r8d, %ecx
	movq	$0x1f, %rdx
	and 	%r9d, %edx
	movapd	%xmm2,%xmm0
	movapd	%xmm1,%xmm3
	movapd	%xmm2,%xmm4
	movapd	%xmm1,%xmm5

	movq	$0x1f, %rsi
	and 	%r10d, %esi
	movq	$0x1f, %rdi
	and 	%r11d, %edi

	sub 	%ecx,%r8d
	sar 	$5,%r8d
	sub 	%edx,%r9d
	sar 	$5,%r9d

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	mulpd	.L__real_3FC5555555548F7C(%rip),%xmm0
	mulpd	.L__real_3FC5555555548F7C(%rip),%xmm1

	sub 	%esi,%r10d
	sar 	$5,%r10d
	sub 	%edi,%r11d
	sar 	$5,%r11d

	mulpd	%xmm2,%xmm2
	mulpd	%xmm3,%xmm3
	addpd	.L__real_3fe0000000000000(%rip),%xmm0
	addpd	.L__real_3fe0000000000000(%rip),%xmm1
	mulpd	%xmm0,%xmm2
	mulpd	%xmm1,%xmm3
	movlpdMR	(%rax,%rdx,8),%xmm0
	movhpd	(%rax,%rcx,8),%xmm0

	movlpdMR	(%rax,%rdi,8),%xmm1
	movhpd	(%rax,%rsi,8),%xmm1

	addpd	%xmm4,%xmm2
	addpd	%xmm5,%xmm3

	/* *z2 = f2 + ((f1 + f2) * q); */
        add 	$1023, %r8d	/* add bias */
        add 	$1023, %r9d	/* add bias */
        add 	$1023, %r10d	/* add bias */
        add 	$1023, %r11d	/* add bias */

	/* deal with infinite and denormal results */
	mulpd	%xmm0,%xmm2
	mulpd	%xmm1,%xmm3
        shlq	$52,%r8
        shlq	$52,%r9
        shlq	$52,%r10
        shlq	$52,%r11
	addpd	%xmm0,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
	addpd	%xmm1,%xmm3  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%r8,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(24)(%rsp),%xmm2	/* result *= 2^n */

	movq	%r11,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r10,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm3	/* result *= 2^n */

	cvtpd2ps %xmm2,%xmm0
	cvtpd2ps %xmm3,%xmm1
	shufps	$68,%xmm1,%xmm0

LBL(.L_vsp_final_check):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif

	RZ_POP
	rep
	ret

LBL(.L__Scalar_fvsexp):
#if defined(_WIN64)
	/* Need to restore callee-saved regs can do here for this path
	 * because entry was only thru fvs_exp/fvsexp_gh
	 */
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif
        pushq   %rbp			/* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        movaps  %xmm0, _SX0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fss_exp))
#else
        CALL(ENT(__fmth_i_exp))
#endif
        movss   %xmm0, _SR0(%rsp)

        movss   _SX1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_exp))
#else
        CALL(ENT(__fmth_i_exp))
#endif
        movss   %xmm0, _SR1(%rsp)

        movss   _SX2(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_exp))
#else
        CALL(ENT(__fmth_i_exp))
#endif
        movss   %xmm0, _SR2(%rsp)

        movss   _SX3(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_exp))
#else
        CALL(ENT(__fmth_i_exp))
#endif
        movss   %xmm0, _SR3(%rsp)

        movaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp

	RZ_POP
	rep
	ret

        ELF_FUNC(ENT_GH(__fvsexp))
        ELF_SIZE(ENT_GH(__fvsexp))
        IF_GH(ELF_FUNC(__fvs_exp))
        IF_GH(ELF_SIZE(__fvs_exp))

/* 
 * ============================================================
 *  exp.asm
 * 
 *  A vector implementation of the exp libm function.
 * 
 *  Prototype:
 * 
 *      __m128d __fvdexp(__m128d x);
 * 
 *    Computes e raised to the x power.
 *  Does not perform error checking.   Denormal results are truncated to 0.
 * 
 */
        .text
        ALN_FUNC
	IF_GH(.globl ENT(__fvd_exp))
	.globl ENT_GH(__fvdexp)
IF_GH(ENT(__fvd_exp):)
ENT_GH(__fvdexp):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	%xmm0, %xmm2
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulpd	%xmm0,%xmm3 

	/* save x for later. */
	andpd	.L__real_mask_unsign(%rip), %xmm2

        /* Set n = nearest integer to r */
	cvtpd2dq %xmm3,%xmm4
	cmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2
	leaq	.L__two_to_jby32_table(%rip),%r11
	cvtdq2pd %xmm4,%xmm1
	movmskpd %xmm2, %r8d

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	movapd	.L__real_log2_by_32_lead(%rip),%xmm2
	mulpd	%xmm1,%xmm2
	movq	 %xmm4,RZ_OFF(24)(%rsp)
	testl	$3, %r8d
	jnz	LBL(.L__Scalar_fvdexp)

	/* r2 =   - n * logbaseof2_by_32_trail; */
	subpd	%xmm2,%xmm0	/* r1 in xmm0, */
	mulpd	.L__real_log2_by_32_tail(%rip),%xmm1 	/* r2 in xmm1 */

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	movl	RZ_OFF(24)(%rsp),%ecx
	andl	%ecx,%r9d

	movl	RZ_OFF(20)(%rsp),%edx
	andl	%edx,%r8d
	movapd	%xmm0,%xmm2

	/* f1 = two_to_jby32_lead_table[j]; */
	/* f2 = two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%r9d,%ecx
	sarl	$5,%ecx
	subl	%r8d,%edx
	sarl	$5,%edx
	addpd	%xmm1,%xmm2    /* r = r1 + r2 */

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	 * r*r*( 5.00000000000000008883e-01 +
	 * r*( 1.66666666665260878863e-01 +
	 * r*( 4.16666666662260795726e-02 +
	 * r*( 8.33336798434219616221e-03 +
	 * r*( 1.38889490863777199667e-03 ))))));
	 * q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movapd	%xmm2,%xmm1
	movapd	.L__real_3f56c1728d739765(%rip),%xmm3
	movapd	.L__real_3FC5555555548F7C(%rip),%xmm0

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

	mulpd	%xmm2,%xmm3	/* *x */
	mulpd	%xmm2,%xmm0	/* *x */
	mulpd	%xmm2,%xmm1	/* x*x */
	movapd	%xmm1,%xmm4

	addpd	 .L__real_3F811115B7AA905E(%rip),%xmm3
	addpd	 .L__real_3fe0000000000000(%rip),%xmm0
	mulpd	%xmm1,%xmm4	/* x^4 */
	mulpd	%xmm2,%xmm3	/* *x */

	mulpd	%xmm1,%xmm0	/* *x^2 */
	addpd	.L__real_3FA5555555545D4E(%rip),%xmm3
	addpd	%xmm2,%xmm0	/* + x  */
	mulpd	%xmm4,%xmm3	/* *x^4 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1023,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	addpd	%xmm3,%xmm0	/* q = final sum */

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	movhpd	(%r11,%r8,8),%xmm5 	/* f1 + f2 */

	/* shufpd	$0,%xmm4,%xmm5 */

	mulpd	%xmm5,%xmm0
	addpd	%xmm5,%xmm0		/* z = z1 + z2 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r11		/* 1 */
	addq	$1022,%rdx		/* add bias */
	cmovleq	%rdx, %r11
	cmovleq	%rax, %rdx
	addq	$1023,%r11		/* add bias */
	shlq	$52, %r11		/* build 2^n */

	/* Step 3. Reconstitute. */
	movq	%r10,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r11,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(24)(%rsp),%xmm0  /* result*= 2^n */

LBL(.L__final_check):
	RZ_POP
	rep
	ret

#define _DX0 0
#define _DX1 8
#define _DX2 16
#define _DX3 24

#define _DR0 32
#define _DR1 40

LBL(.L__Scalar_fvdexp):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _DX0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fsd_exp))
#else
        CALL(ENT(__fmth_i_dexp))
#endif
        movsd   %xmm0, _DR0(%rsp)

        movsd   _DX1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fsd_exp))
#else
        CALL(ENT(__fmth_i_dexp))
#endif
        movsd   %xmm0, _DR1(%rsp)

        movapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

        ELF_FUNC(ENT_GH(__fvdexp))
        ELF_SIZE(ENT_GH(__fvdexp))
	IF_GH(ELF_FUNC(__fvd_exp))
	IF_GH(ELF_SIZE(__fvd_exp))


/*
 * This version uses SSE and one table lookup which pulls out 3 values,
 * for the polynomial approximation ax**2 + bx + c, where x has been reduced
 * to the range [ 1/sqrt(2), sqrt(2) ] and has had 1.0 subtracted from it.
 * The bulk of the answer comes from the taylor series
 *   log(x) = (x-1) - (x-1)**2/2 + (x-1)**3/3 - (x-1)**4/4
 * Method for argument reduction and result reconstruction is from
 * Cody & Waite.
 *
 * 5/04/04  B. Leback
 *
 */

/*
 *  float __fmth_i_alogx(float f)
 *
 *  Expects its argument f in %xmm0 instead of on the floating point
 *  stack, and also returns the result in %xmm0 instead of on the
 *  floating point stack.
 *
 *   stack usage:
 *   +---------+
 *   |   ret   | 12    <-- prev %esp
 *   +---------+
 *   |         | 8
 *   +--     --+
 *   |   lcl   | 4
 *   +--     --+
 *   |         | 0     <-- %esp  (8-byte aligned)
 *   +---------+
 *
 */

	.text
	ALN_FUNC
	IF_GH(.globl	ENT(__fss_log))
	.globl	ENT_GH(__fmth_i_alog)
IF_GH(ENT(__fss_log):)
ENT_GH(__fmth_i_alog):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(24)(%rsp)
#endif
	/* First check for valid input:
	 * if (a .gt. 0.0) then !!! Also check if not +infinity */

	/* Get started:
	 * ra = a
	 * ma = IAND(ia,'007fffff'x)
	 * ms = ma - '3504f3'x
	 * ig = IOR(ma,'3f000000'x)
	 * nx = ISHFT(ia,-23) - 126
	 * mx = IAND(ms,'00800000'x)
	 * ig = IOR(ig,mx)
	 * nx = nx - ISHFT(mx,-23)
         * ms = IAND(ms,'007f0000'x)
         * mt1 = ISHFT(ms,-16)
         * mt2 = ISHFT(ms,-15)
         * mt = mt1 + mt2 */

	movss	%xmm0, RZ_OFF(4)(%rsp)
        movss	.L4_384(%rip), %xmm2	/* Move smallest normalized number */
	movl	RZ_OFF(4)(%rsp), %ecx
	andl	$8388607, %ecx		/* ma = IAND(ia,'007fffff'x) */
	leaq 	-3474675(%rcx), %rdx	/* ms = ma - '3504f3'x */
	orl	$1056964608, %ecx	/* ig = IOR(ma,'3f000000'x) */
	cmpnless %xmm0, %xmm2		/* '00800000'x <= a, xmm2 1 where not */
        cmpeqss	.L4_387(%rip), %xmm0	/* Test for == +inf */
	movl	%edx, %eax		/* move ms */
	andl	$8388608, %edx		/* mx = IAND(ms,'00800000'x) */
	orl	%edx, %ecx		/* ig = IOR(ig,mx) */
	movl	%ecx, RZ_OFF(8)(%rsp)	/* move back over to fp sse */
	shrl	$23, %edx		/* ISHFT(mx,-23) */
        unpcklps %xmm2, %xmm0		/* Mask for nan, inf, neg and 0.0 */

	leaq	.L_STATICS1(%rip), %r8
	movl	RZ_OFF(4)(%rsp), %ecx	/* ia */
	andl	$8323072, %eax		/* ms = IAND(ms,'007f0000'x) */
	movss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	movmskps %xmm0, %r9d		/* move exception mask to gp reg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	movss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$126, %ecx		/* nx = ISHFT(ia,-23) - 126 */
	movss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
        shrl    $14, %eax		/* mt1 */
	and	$3, %r9d		/* mask with 3 */
	movss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
	jnz	LBL(.LB1_800)

LBL(.LB1_100):
	/* Do fp portion
         * xn = real(nx)
         * x0 = rg - 1.0
         * xsq = x0 * x0
         * xcu = xsq * x0
         * x1 = 0.5 * xsq
         * x3 = x1 * x1
         * x2 = thrd * xcu
         * rp = (COEFFS(mt) * x0 + COEFFS(mt+1)) * x0 + COEFFS(mt+2)
         * rz = rp - x3 + x2 - x1 + x0
         * rr = (xn * c1 + rz) + xn * c2 */

#ifdef GH_TARGET
	movd %ecx, %xmm0
	cvtdq2ps %xmm0, %xmm0
#else
	cvtsi2ss %ecx, %xmm0		/* xn */
#endif
	subss	.L4_386(%rip), %xmm1	/* x0 = rg - 1.0 */
	subss	.L4_386(%rip), %xmm6	/* x0 = rg - 1.0 */
	subss	.L4_386(%rip), %xmm4	/* x0 = rg - 1.0 */
	subss	.L4_386(%rip), %xmm2	/* x0 = rg - 1.0 */
	mulss	(%r8,%rax,4), %xmm1	/* COEFFS(mt) * x0 */
	mulss   %xmm6, %xmm6		/* xsq = x0 * x0 */
	addss	4(%r8,%rax,4), %xmm1	/* COEFFS(mt) * x0 + COEFFS(mt+1) */
	mulss   %xmm6, %xmm4		/* xcu = xsq * x0 */
	mulss   .L4_383(%rip), %xmm6	/* x1 = 0.5 * xsq */
	mulss   %xmm2, %xmm1		/* * x0 */
	mulss	12(%r8,%rax,4), %xmm4	/* x2 = thrd * xcu */
	movssRR	%xmm6, %xmm3		/* move x1 */
	mulss	%xmm6, %xmm6		/* x3 = x1 * x1 */
	addss	8(%r8,%rax,4), %xmm1	/* + COEFFS(mt+2) = rp */
	subss	%xmm6, %xmm1		/* rp - x3 */
	movss	.L4_388(%rip), %xmm5	/* Move c1 */
        movss   .L4_389(%rip), %xmm6	/* Move c2 */
	addss	%xmm1, %xmm4		/* rp - x3 + x2 */
	subss	%xmm3, %xmm4		/* rp - x3 + x2 - x1 */
	addss	%xmm2, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
	mulss   %xmm0, %xmm5		/* xn * c1 */
	addss   %xmm5, %xmm4		/* (xn * c1 + rz) */
        mulss   %xmm6, %xmm0		/* xn * c2 */
        addss   %xmm4, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */

LBL(.LB1_900):

#if defined(_WIN64)
	movdqa	RZ_OFF(24)(%rsp), %xmm6
#endif
	RZ_POP
	rep
	ret

	ALN_WORD
LBL(.LB1_800):
	/* ir = 'ff800000'x */
	xorq	%rax,%rax
	movss	RZ_OFF(4)(%rsp), %xmm0
	movd 	%rax, %xmm1
	comiss	%xmm1, %xmm0
	jp	LBL(.LB1_cvt_nan)
#ifdef FMATH_EXCEPTIONS
        movss  .L4_386(%rip), %xmm1
        divss  %xmm0, %xmm1     /* Generate div-by-zero op when x=0 */
#endif
	movss	.L4_391(%rip),%xmm0	/* Move -inf */
	je	LBL(.LB1_900)
#ifdef FMATH_EXCEPTIONS
        sqrtss %xmm0, %xmm0     /* Generate invalid op when x < 0 */
#endif
	movss	.L4_390(%rip),%xmm0	/* Move -nan */
	jb	LBL(.LB1_900)
	movss	.L4_387(%rip), %xmm0	/* Move +inf */
	movss	RZ_OFF(4)(%rsp), %xmm1
	comiss	%xmm1, %xmm0
	je	LBL(.LB1_900)

	/* Otherwise, we had a denormal as an input */
	mulss	.L4_392(%rip), %xmm1	/* a * scale factor */
	movss	%xmm1, RZ_OFF(4)(%rsp)
	movl	RZ_OFF(4)(%rsp), %ecx
	andl	$8388607, %ecx		/* ma = IAND(ia,'007fffff'x) */
	leaq	-3474675(%rcx), %rdx	/* ms = ma - '3504f3'x */
	orl	$1056964608, %ecx	/* ig = IOR(ma,'3f000000'x) */
	movl	%edx, %eax		/* move ms */
	andl	$8388608, %edx		/* mx = IAND(ms,'00800000'x) */
	orl	%edx, %ecx		/* ig = IOR(ig,mx) */
	movl	%ecx, RZ_OFF(8)(%rsp)	/* move back over to fp sse */
	shrl	$23, %edx		/* ISHFT(mx,-23) */
	movl	RZ_OFF(4)(%rsp), %ecx	/* ia */
	andl	$8323072, %eax		/* ms = IAND(ms,'007f0000'x) */
	movss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	movss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$149, %ecx		/* nx = ISHFT(ia,-23) - (126 + 23) */
	movss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
	movss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
        shrl    $14, %eax		/* mt1 */
	jmp	LBL(.LB1_100)

LBL(.LB1_cvt_nan):
	movss	.L4_394(%rip), %xmm1	/* nan bit */
	orps	%xmm1, %xmm0
	jmp	LBL(.LB1_900)

        ELF_FUNC(ENT_GH(__fmth_i_alog))
        ELF_SIZE(ENT_GH(__fmth_i_alog))
	IF_GH(ELF_FUNC(__fss_log))
	IF_GH(ELF_SIZE(__fss_log))


/*
#============================================================
#
# fastlog.s
#
# An implementation of the log libm function.
#
# Prototype:
#
#     double log(double x);
#
#   Computes the natural log of x.
#   Returns proper C99 values, but may not raise status flags properly.
#   Less than 1 ulp of error.  Runs in 96 cycles for worst case.
#
#
*/

	.text
	ALN_FUNC
        IF_GH(.globl  ENT(__fsd_log))
        .globl  ENT_GH(__fmth_i_dlog)
IF_GH(ENT(__fsd_log):)
ENT_GH(__fmth_i_dlog):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(24)(%rsp)
#endif
	/* Get input x into the range [0.5,1) */
	/* compute the index into the log tables */

	comisd	.L__real_mindp(%rip), %xmm0
	movdqa	%xmm0,%xmm3
	movsdRR	%xmm0,%xmm1
	jb	LBL(.L__z_or_n)

	psrlq	$52,%xmm3
	subsd	.L__real_one(%rip),%xmm1
	psubq	.L__mask_1023(%rip),%xmm3
	cvtdq2pd %xmm3,%xmm6	/* xexp */

LBL(.L__100):
	movdqa	%xmm0,%xmm3
	pand	.L__real_mant(%rip),%xmm3
	xorq	%r8,%r8
	movdqa	%xmm3,%xmm4
	movlpdMR	.L__real_half(%rip),%xmm5	/* .5 */
	/* Now  x = 2**xexp  * f,  1/2 <= f < 1. */
	psrlq	$45,%xmm3
	movdqa	%xmm3,%xmm2
	psrlq	$1,%xmm3
	paddq	.L__mask_040(%rip),%xmm3
	pand	.L__mask_001(%rip),%xmm2
	paddq	%xmm2,%xmm3

	andpd	.L__real_notsign(%rip),%xmm1
	comisd	.L__real_threshold(%rip),%xmm1
	cvtdq2pd %xmm3,%xmm1
	jb	LBL(.L__near_one)
	movd	%xmm3,%r8d

	/* reduce and get u */
	por	.L__real_half(%rip),%xmm4
	movdqa	%xmm4,%xmm2

	mulsd	.L__real_3f80000000000000(%rip),%xmm1	/* f1 = index/128 */
	leaq	.L__np_ln_lead_table(%rip),%r9
	subsd	%xmm1,%xmm2				/* f2 = f - f1 */

	mulsd	%xmm2,%xmm5
	addsd	%xmm5,%xmm1

	divsd	%xmm1,%xmm2				/* u */

	/* Check for +inf */
	comisd	.L__real_inf(%rip),%xmm0
	je	LBL(.L__finish)

	movlpdMR	-512(%r9,%r8,8),%xmm0 			/* z1 */
	/* solve for ln(1+u) */
	movsdRR	%xmm2,%xmm1				/* u */
	mulsd	%xmm2,%xmm2				/* u^2 */
	movsdRR	%xmm2,%xmm5
	movapd	.L__real_cb3(%rip),%xmm3
	mulsd	%xmm2,%xmm3				/* Cu2 */
	mulsd	%xmm1,%xmm5				/* u^3 */
	addsd	.L__real_cb2(%rip),%xmm3 		/* B+Cu2 */
	movapd	%xmm2,%xmm4
	mulsd	%xmm5,%xmm4				/* u^5 */
	movlpdMR	.L__real_log2_lead(%rip),%xmm2
	mulsd	.L__real_cb1(%rip),%xmm5 		/* Au3 */
	addsd	%xmm5,%xmm1				/* u+Au3 */
	mulsd	%xmm3,%xmm4				/* u5(B+Cu2) */
	addsd	%xmm4,%xmm1				/* poly */

	/* recombine */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	addsd	-512(%rdx,%r8,8),%xmm1 			/* z2	+=q */
	mulsd	%xmm6,%xmm2				/* npi2 * log2_lead */
	addsd	%xmm2,%xmm0				/* r1 */
	mulsd	.L__real_log2_tail(%rip),%xmm6
	addsd	%xmm6,%xmm1				/* r2 */
	addsd	%xmm1,%xmm0

LBL(.L__finish):
#if defined(_WIN64)
	movdqa	RZ_OFF(24)(%rsp), %xmm6
#endif

	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.L__near_one):
	/* saves 10 cycles */
	/* r = x - 1.0; */
	movlpdMR	.L__real_two(%rip),%xmm2
	subsd	.L__real_one(%rip),%xmm0

	/* u = r / (2.0 + r); */
	addsd	%xmm0,%xmm2
	movsdRR	%xmm0,%xmm1
	divsd	%xmm2,%xmm1
	movlpdMR	.L__real_ca4(%rip),%xmm4
	movlpdMR	.L__real_ca3(%rip),%xmm5
	/* correction = r * u; */
	movsdRR	%xmm0,%xmm6
	mulsd	%xmm1,%xmm6

	/* u = u + u; */
	addsd	%xmm1,%xmm1
	movsdRR	%xmm1,%xmm2
	mulsd	%xmm2,%xmm2
	/* r2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction); */
	mulsd	%xmm1,%xmm5
	movsdRR	%xmm1,%xmm3
	mulsd	%xmm2,%xmm3
	mulsd	.L__real_ca2(%rip),%xmm2
	mulsd	%xmm3,%xmm4

	addsd	.L__real_ca1(%rip),%xmm2
	movsdRR	%xmm3,%xmm1
	mulsd	%xmm1,%xmm1
	addsd	%xmm4,%xmm5

	mulsd	%xmm3,%xmm2
	mulsd	%xmm5,%xmm1
	addsd	%xmm1,%xmm2
	subsd	%xmm6,%xmm2

	/* return r + r2; */
	addsd	%xmm2,%xmm0
	jmp	LBL(.L__finish)

	/* Start here for all the conditional cases */
	/* we have a zero, a negative number, denorm, or nan. */
LBL(.L__z_or_n):
	jp	LBL(.L__lnan)
	xorpd	%xmm1, %xmm1
	comisd	%xmm1, %xmm0
	je	LBL(.L__zero)
	jbe	LBL(.L__negative_x)

	/* A Denormal input, scale appropriately */
	mulsd	.L__real_scale(%rip), %xmm0
	movdqa	%xmm0, %xmm3
	movsdRR	%xmm0, %xmm1

	psrlq	$52,%xmm3
	subsd	.L__real_one(%rip),%xmm1
	psubq	.L__mask_1075(%rip),%xmm3
	cvtdq2pd %xmm3,%xmm6
	jmp	LBL(.L__100)

	/* x == +/-0.0 */
LBL(.L__zero):
#ifdef FMATH_EXCEPTIONS
        movsd  .L__real_one(%rip), %xmm1
        divsd  %xmm0, %xmm1 /* Generate divide-by-zero op */
#endif

	movlpdMR	.L__real_ninf(%rip),%xmm0  /* C99 specs -inf for +-0 */
	jmp	LBL(.L__finish)

	/* x < 0.0 */
LBL(.L__negative_x):
#ifdef FMATH_EXCEPTIONS
        sqrtsd %xmm0, %xmm0
#endif

	movlpdMR	.L__real_nan(%rip),%xmm0
	jmp	LBL(.L__finish)

	/* NaN */
LBL(.L__lnan):
	xorpd	%xmm1, %xmm1
	movlpdMR	.L__real_qnanbit(%rip), %xmm1	/* convert to quiet */
	orpd	%xmm1, %xmm0
	jmp	LBL(.L__finish)


        ELF_FUNC(ENT_GH(__fmth_i_dlog))
        ELF_SIZE(ENT_GH(__fmth_i_dlog))
        IF_GH(ELF_FUNC(__fsd_log))
        IF_GH(ELF_SIZE(__fsd_log))


/*
 * This version uses SSE and one table lookup which pulls out 3 values,
 * for the polynomial approximation ax**2 + bx + c, where x has been reduced
 * to the range [ 1/sqrt(2), sqrt(2) ] and has had 1.0 subtracted from it.
 * The bulk of the answer comes from the taylor series
 *   log(x) = (x-1) - (x-1)**2/2 + (x-1)**3/3 - (x-1)**4/4
 * Method for argument reduction and result reconstruction is from
 * Cody & Waite.
 *
 * 5/15/04  B. Leback
 *
 */
	.text
	ALN_FUNC
	IF_GH(.globl ENT(__fvs_log))
	.globl ENT_GH(__fvslog)
IF_GH(ENT(__fvs_log):)
ENT_GH(__fvslog):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
	movdqa	%xmm7, RZ_OFF(72)(%rsp)
#endif

/* Fast vector natural logarithm code goes here... */
        /* First check for valid input:
         * if (a .gt. 0.0) then */
	movaps  .L4_384(%rip), %xmm4	/* Move min arg to xmm4 */
	xorps	%xmm7, %xmm7		/* Still need 0.0 */
	movaps	%xmm0, %xmm2		/* Move for nx */
	movaps	%xmm0, %xmm1		/* Move to xmm1 for later ma */

	/* Check exceptions and valid range */
	cmpleps	%xmm0, %xmm4		/* '00800000'x <= a, xmm4 1 where true */
	cmpltps	%xmm0, %xmm7		/* Test for 0.0 < a, xmm7 1 where true */
	cmpneqps .L4_387(%rip), %xmm0	/* Test for == +inf */
	xorps	%xmm7, %xmm4		/* xor to find just denormal inputs */
	movmskps %xmm4, %eax		/* Move denormal mask to gp ref */
	movaps	%xmm2, RZ_OFF(24)(%rsp)	/* Move for exception processing */
	movaps	.L4_382(%rip), %xmm3	/* Move 126 */
	cmp	$0, %eax		/* Test for denormals */
	jne	LBL(.LB_DENORMs)

        /* Get started:
         * ra = a
         * ma = IAND(ia,'007fffff'x)
         * ms = ma - '3504f3'x
         * ig = IOR(ma,'3f000000'x)
         * nx = ISHFT(ia,-23) - 126
         * mx = IAND(ms,'00800000'x)
         * ig = IOR(ig,mx)
         * nx = nx - ISHFT(mx,-23)
         * ms = IAND(ms,'007f0000'x)
         * mt = ISHFT(ms,-12) */

LBL(.LB_100):
	leaq	.L_STATICS1(%rip),%r8
	andps	.L4_380(%rip), %xmm1	/* ma = IAND(ia,'007fffff'x) */
	psrld	$23, %xmm2		/* nx = ISHFT(ia,-23) */
	andps	%xmm0, %xmm7		/* Mask for nan, inf, neg and 0.0 */
	movaps	%xmm1, %xmm6		/* move ma for ig */
	psubd	.L4_381(%rip), %xmm1	/* ms = ma - '3504f3'x */
	psubd	%xmm3, %xmm2		/* nx = ISHFT(ia,-23) - 126 */
	orps	.L4_383(%rip), %xmm6	/* ig = IOR(ma,'3f000000'x) */
	movaps	%xmm1, %xmm0		/* move ms for tbl ms */
	andps	.L4_384(%rip), %xmm1	/* mx = IAND(ms,'00800000'x) */
	andps	.L4_385(%rip), %xmm0	/* ms = IAND(ms,'007f0000'x) */
	orps	%xmm1, %xmm6		/* ig = IOR(ig, mx) */
	psrad	$23, %xmm1		/* ISHFT(mx,-23) */
	psrad	$12, %xmm0		/* ISHFT(ms,-12) for 128 bit reads */
	movmskps %xmm7, %eax		/* Move xmm7 mask to eax */
	psubd	%xmm1, %xmm2		/* nx = nx - ISHFT(mx,-23) */
	movaps	%xmm0, RZ_OFF(40)(%rsp)	/* Move to memory */
	cvtdq2ps  %xmm2, %xmm0		/* xn = real(nx) */

	movl	RZ_OFF(40)(%rsp), %ecx		/* Move to gp register */
	movaps	(%r8,%rcx,1), %xmm1		/* Read from 1st table location */
	movl	RZ_OFF(36)(%rsp), %edx		/* Move to gp register */
	movaps	(%r8,%rdx,1), %xmm2		/* Read from 2nd table location */
	movl	RZ_OFF(32)(%rsp), %ecx		/* Move to gp register */
	movaps	(%r8,%rcx,1), %xmm3		/* Read from 3rd table location */
	movl	RZ_OFF(28)(%rsp), %edx		/* Move to gp register */
	movaps	(%r8,%rdx,1), %xmm4		/* Read from 4th table location */

	/* So, we do 4 reads of a,b,c into registers xmm1, xmm2, xmm3, xmm4
	 * Assume we need to keep rg in xmm6, xn in xmm0
	 * The following shuffle gets them into SIMD mpy form:
	 */

	subps	.L4_386(%rip), %xmm6 	/* x0 = rg - 1.0 */

	movaps	%xmm1, %xmm5		/* Store 1/3, c0, b0, a0 */
	movaps	%xmm3, %xmm7		/* Store 1/3, c2, b2, a2 */

	unpcklps %xmm2, %xmm1		/* b1, b0, a1, a0 */
	unpcklps %xmm4, %xmm3		/* b3, b2, a3, a2 */
	unpckhps %xmm2, %xmm5		/* 1/3, 1/3, c1, c0 */
	unpckhps %xmm4, %xmm7		/* 1/3, 1/3, c3, c2 */

	movaps	%xmm6, %xmm4		/* move x0 */

	movaps	%xmm1, %xmm2		/* Store b1, b0, a1, a0 */
	movlhps	%xmm3, %xmm1		/* a3, a2, a1, a0 */
	movlhps	%xmm7, %xmm5		/* c3, c2, c1, c0 */
	movhlps	%xmm2, %xmm3		/* b3, b2, b1, b0 */

	mulps	%xmm6, %xmm1		/* COEFFS(mt) * x0 */
	mulps	%xmm6, %xmm6		/* xsq = x0 * x0 */
	movhlps	%xmm7, %xmm7		/* 1/3, 1/3, 1/3, 1/3 */

	movaps	%xmm4, %xmm2		/* move x0 */

        /* Do fp portion
         * xn = real(nx)
         * x0 = rg - 1.0
         * xsq = x0 * x0
         * xcu = xsq * x0
         * x1 = 0.5 * xsq
         * x3 = x1 * x1
         * x2 = thrd * xcu
         * rp = (COEFFS(mt) * x0 + COEFFS(mt+1)) * x0 + COEFFS(mt+2)
         * rz = rp - x3 + x2 - x1 + x0
         * rr = (xn * c1 + rz) + xn * c2 */

	/* Now do the packed coefficient multiply and adds */
	/* x4 has x0 */
	/* x6 has xsq */
	/* x7 has thrd * x0 */
	/* x1, x3, and x5 have a, b, c */
	/* x0 has xn */
	addps	%xmm3, %xmm1		/* COEFFS(mt) * g + COEFFS(mt+1) */
	mulps	%xmm6, %xmm4		/* xcu = xsq * x0 */
	mulps	.L4_383(%rip), %xmm6	/* x1 = 0.5 * xsq */
	mulps	%xmm2, %xmm1		/* * x0 */
	mulps	%xmm7, %xmm4		/* x2 = thrd * xcu */
	movaps	%xmm6, %xmm3		/* move x1 */
	mulps	%xmm6, %xmm6		/* x3 = x1 * x1 */
	addps	%xmm5, %xmm1		/* + COEFFS(mt+2) = rp */
	subps	%xmm6, %xmm1		/* rp - x3 */
	movaps	.L4_388(%rip), %xmm7	/* Move c1 */
        movaps  .L4_389(%rip), %xmm6	/* Move c2 */
	addps	%xmm1, %xmm4		/* rp - x3 + x2 */
	subps	%xmm3, %xmm4		/* rp - x3 + x2 - x1 */
	addps	%xmm2, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
	mulps   %xmm0, %xmm7		/* xn * c1 */
	addps   %xmm7, %xmm4		/* (xn * c1 + rz) */
        mulps   %xmm6, %xmm0		/* xn * c2 */
        addps   %xmm4, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */

	/* Compare exception mask now and jump if no exceptions */
	cmp	$15, %eax
	jne 	LBL(.LB_EXCEPTs)

LBL(.LB_900):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movdqa	RZ_OFF(72)(%rsp), %xmm7
#endif

	RZ_POP
	rep
	ret

LBL(.LB_EXCEPTs):
        /* Handle all exceptions by masking in xmm */
        movaps  RZ_OFF(24)(%rsp), %xmm1	/* original input */
        movaps  RZ_OFF(24)(%rsp), %xmm2	/* original input */
        movaps  RZ_OFF(24)(%rsp), %xmm3	/* original input */
        xorps   %xmm7, %xmm7            /* xmm7 = 0.0 */
        xorps   %xmm6, %xmm6            /* xmm6 = 0.0 */
	movaps	.L4_394(%rip), %xmm5	/* convert nan bit */
        xorps   %xmm4, %xmm4            /* xmm4 = 0.0 */
                                                                                
        cmpunordps %xmm1, %xmm7         /* Test if unordered */
        cmpltps %xmm6, %xmm2            /* Test if a < 0.0 */
        cmpordps %xmm1, %xmm6           /* Test if ordered */
                                                                                
        andps   %xmm7, %xmm5            /* And nan bit where unordered */
        orps    %xmm7, %xmm4            /* Or all masks together */
        andps   %xmm1, %xmm7            /* And input where unordered */
	orps	%xmm5, %xmm7		/* Convert unordered nans */
                                                                                
        xorps   %xmm5, %xmm5            /* xmm5 = 0.0 */
        andps   %xmm2, %xmm6            /* Must be ordered and < 0.0 */
        orps    %xmm6, %xmm4            /* Or all masks together */
        andps   .L4_390(%rip), %xmm6    /* And -nan if < 0.0 and ordered */
                                                                                
        cmpeqps .L4_387(%rip), %xmm3    /* Test if equal to infinity */
        cmpeqps %xmm5, %xmm1            /* Test if eq 0.0 */
        orps    %xmm6, %xmm7            /* or in < 0.0 */
                                                                                
        orps    %xmm3, %xmm4            /* Or all masks together */
        andps   .L4_387(%rip), %xmm3    /* inf and inf mask */
        movaps  %xmm0, %xmm2
        orps    %xmm3, %xmm7            /* or in infinity */
                                                                                
        orps    %xmm1, %xmm4            /* Or all masks together */
        andps   .L4_391(%rip), %xmm1    /* And -inf if == 0.0 */
        movaps  %xmm4, %xmm0
        orps    %xmm1, %xmm7            /* or in -infinity */
                                                                                
        andnps  %xmm2, %xmm0            /* Where mask not set, use result */
        orps    %xmm7, %xmm0            /* or in exceptional values */
	jmp	LBL(.LB_900)

LBL(.LB_DENORMs):
	/* Have the denorm mask in xmm4, so use it to scale a and the subtractor */
	movaps	%xmm4, %xmm5		/* Move mask */
	movaps	%xmm4, %xmm6		/* Move mask */
	andps	.L4_392(%rip), %xmm4	/* Have 2**23 where denorms are, 0 else */
	andnps	%xmm1, %xmm5		/* Have a where denormals aren't */
	mulps	%xmm4, %xmm1		/* denormals * 2**23 */
	andps	.L4_393(%rip), %xmm6	/* have 23 where denorms are, 0 else */
	orps	%xmm5, %xmm1		/* Or in the original a */
	paddd	%xmm6, %xmm3		/* Add 23 to 126 for offseting exponent */
	movaps	%xmm1, %xmm2		/* Move to the next location */
	jmp	LBL(.LB_100)

        ELF_FUNC(ENT_GH(__fvslog))
        ELF_SIZE(ENT_GH(__fvslog))
	IF_GH(ELF_FUNC(__fvs_log))
	IF_GH(ELF_SIZE(__fvs_log))


/* ======================================================================== */
    	.text
    	ALN_FUNC
	IF_GH(.globl ENT(__fvd_log))
	.globl ENT_GH(__fvdlog)
IF_GH(ENT(__fvd_log):)
ENT_GH(__fvdlog):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
#endif

	movdqa	%xmm0, RZ_OFF(40)(%rsp)	/* save the input values */
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm4
	pxor	%xmm1, %xmm1
	cmppd	$6, .L__real_maxfp(%rip), %xmm2
	cmppd 	$1, .L__real_mindp(%rip), %xmm4
	movdqa	%xmm0, %xmm3
	psrlq	$52, %xmm3
	orpd	%xmm2, %xmm4
	psubq	.L__mask_1023(%rip),%xmm3
	movmskpd %xmm4, %r8d
	packssdw %xmm1, %xmm3
	cvtdq2pd %xmm3, %xmm6		/* xexp */
	movdqa	%xmm0, %xmm2
	xorq	%rax, %rax
	subpd	.L__real_one(%rip), %xmm2
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdlog)

	movdqa	%xmm0,%xmm3
	andpd	.L__real_notsign(%rip),%xmm2
	pand	.L__real_mant(%rip),%xmm3
	movdqa	%xmm3,%xmm4
	movapd	.L__real_half(%rip),%xmm5	/* .5 */

	cmppd	$1,.L__real_threshold(%rip),%xmm2
	movmskpd %xmm2,%r10d
	cmp	$3,%r10d
	jz	LBL(.Lall_nearone)

	psrlq	$45,%xmm3
	movdqa	%xmm3,%xmm2
	psrlq	$1,%xmm3
	paddq	.L__mask_040(%rip),%xmm3
	pand	.L__mask_001(%rip),%xmm2
	paddq	%xmm2,%xmm3

	packssdw %xmm1,%xmm3
	cvtdq2pd %xmm3,%xmm1
	xorq	 %rcx,%rcx
	movq	 %xmm3,RZ_OFF(24)(%rsp)

	por	.L__real_half(%rip),%xmm4
	movdqa	%xmm4,%xmm2
	mulpd	.L__real_3f80000000000000(%rip),%xmm1	/* f1 = index/128 */

	leaq	.L__np_ln_lead_table(%rip),%rdx
	mov	RZ_OFF(24)(%rsp),%eax

	subpd	%xmm1,%xmm2				/* f2 = f - f1 */
	mulpd	%xmm2,%xmm5
	addpd	%xmm5,%xmm1

	divpd	%xmm1,%xmm2				/* u */

	movlpdMR	 -512(%rdx,%rax,8),%xmm0		/* z1 */
	mov	RZ_OFF(20)(%rsp),%ecx
	movhpd	 -512(%rdx,%rcx,8),%xmm0		/* z1 */
	movapd	%xmm2,%xmm1				/* u */
	mulpd	%xmm2,%xmm2				/* u^2 */
	movapd	%xmm2,%xmm5
	movapd	.L__real_cb3(%rip),%xmm3
	mulpd	%xmm2,%xmm3				/* Cu2 */
	mulpd	%xmm1,%xmm5				/* u^3 */
	addpd	.L__real_cb2(%rip),%xmm3 		/* B+Cu2 */

	mulpd	%xmm5,%xmm2				/* u^5 */
	movapd	.L__real_log2_lead(%rip),%xmm4

	mulpd	.L__real_cb1(%rip),%xmm5 		/* Au3 */
	addpd	%xmm5,%xmm1				/* u+Au3 */
	mulpd	%xmm3,%xmm2				/* u5(B+Cu2) */

	addpd	%xmm2,%xmm1				/* poly */
	mulpd	%xmm6,%xmm4				/* xexp * log2_lead */
	addpd	%xmm4,%xmm0				/* r1 */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	movlpdMR	 -512(%rdx,%rax,8),%xmm4		/* z2+=q */
	movhpd	 -512(%rdx,%rcx,8),%xmm4		/* z2+=q */

	addpd	%xmm4,%xmm1

	mulpd	.L__real_log2_tail(%rip),%xmm6

	addpd	%xmm6,%xmm1				/* r2 */

	addpd	%xmm1,%xmm0

LBL(.Lfinish):
	test		 $3,%r10d
	jnz		LBL(.Lnear_one)
LBL(.Lfinishn1):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
#endif
	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.Lall_nearone):
	movapd	.L__real_two(%rip),%xmm2
	subpd	.L__real_one(%rip),%xmm0	/* r */
	addpd	%xmm0,%xmm2
	movapd	%xmm0,%xmm1
	divpd	%xmm2,%xmm1			/* u */
	movapd	.L__real_ca4(%rip),%xmm4  	/* D */
	movapd	.L__real_ca3(%rip),%xmm5 	/* C */
	movapd	%xmm0,%xmm6
	mulpd	%xmm1,%xmm6			/* correction */
	addpd	%xmm1,%xmm1			/* u */
	movapd	%xmm1,%xmm2
	mulpd	%xmm2,%xmm2			/* v =u^2 */
	mulpd	%xmm1,%xmm5			/* Cu */
	movapd	%xmm1,%xmm3
	mulpd	%xmm2,%xmm3			/* u^3 */
	mulpd	.L__real_ca2(%rip),%xmm2	/* Bu^2 */
	mulpd	%xmm3,%xmm4			/* Du^3 */

	addpd	.L__real_ca1(%rip),%xmm2	/* +A */
	movapd	%xmm3,%xmm1
	mulpd	%xmm1,%xmm1			/* u^6 */
	addpd	%xmm4,%xmm5			/* Cu+Du3 */

	mulpd	%xmm3,%xmm2			/* u3(A+Bu2) */
	mulpd	%xmm5,%xmm1			/* u6(Cu+Du3) */
	addpd	%xmm1,%xmm2
	subpd	%xmm6,%xmm2			/*  -correction */
	
	addpd	%xmm2,%xmm0
	jmp	LBL(.Lfinishn1)

	ALN_QUAD
LBL(.Lnear_one):
	test	$1,%r10d
	jz	LBL(.Llnn12)

	movlpd	RZ_OFF(40)(%rsp),%xmm0          /* Don't mess with this one */
                                                /* Need the high half live */
	call	LBL(.Lln1)

LBL(.Llnn12):
	test	$2,%r10d			/* second number? */
	jz	LBL(.Llnn1e)
	movlpd	%xmm0,RZ_OFF(40)(%rsp)
	movlpdMR	RZ_OFF(32)(%rsp),%xmm0
	call	LBL(.Lln1)
	movlpd	%xmm0,RZ_OFF(32)(%rsp)
	movapd	RZ_OFF(40)(%rsp),%xmm0

LBL(.Llnn1e):
	jmp		LBL(.Lfinishn1)

LBL(.Lln1):
	movlpdMR	.L__real_two(%rip),%xmm2
	subsd	.L__real_one(%rip),%xmm0	/* r */
	addsd	%xmm0,%xmm2
	movsdRR	%xmm0,%xmm1
	divsd	%xmm2,%xmm1			/* u */
	movlpdMR	.L__real_ca4(%rip),%xmm4	/* D */
	movlpdMR	.L__real_ca3(%rip),%xmm5	/* C */
	movsdRR	%xmm0,%xmm6
	mulsd	%xmm1,%xmm6			/* correction */
	addsd	%xmm1,%xmm1			/* u */
	movsdRR	%xmm1,%xmm2
	mulsd	%xmm2,%xmm2			/* v =u^2 */
	mulsd	%xmm1,%xmm5			/* Cu */
	movsdRR	%xmm1,%xmm3
	mulsd	%xmm2,%xmm3			/* u^3 */
	mulsd	.L__real_ca2(%rip),%xmm2	/*Bu^2 */
	mulsd	%xmm3,%xmm4			/*Du^3 */

	addsd	.L__real_ca1(%rip),%xmm2	/* +A */
	movsdRR	%xmm3,%xmm1
	mulsd	%xmm1,%xmm1			/* u^6 */
	addsd	%xmm4,%xmm5			/* Cu+Du3 */

	mulsd	%xmm3,%xmm2			/* u3(A+Bu2) */
	mulsd	%xmm5,%xmm1			/* u6(Cu+Du3) */
	addsd	%xmm1,%xmm2
	subsd	%xmm6,%xmm2			/* -correction */
	
	addsd	%xmm2,%xmm0
	ret

#define _X0 0
#define _X1 8

#define _R0 32
#define _R1 40

LBL(.L__Scalar_fvdlog):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _X0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fsd_log))
#else
        CALL(ENT(__fmth_i_dlog))
#endif
        movsd   %xmm0, _R0(%rsp)

        movsd   _X1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fsd_log))
#else
        CALL(ENT(__fmth_i_dlog))
#endif
        movsd   %xmm0, _R1(%rsp)

        movapd  _R0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.Lfinishn1)

        ELF_FUNC(ENT_GH(__fvdlog))
        ELF_SIZE(ENT_GH(__fvdlog))
        IF_GH(ELF_FUNC(__fvd_log))
        IF_GH(ELF_SIZE(__fvd_log))


/* ============================================================ */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fss_sin))
        .globl ENT_GH(__fmth_i_sin)
IF_GH(ENT(__fss_sin):)
ENT_GH(__fmth_i_sin):
        movd    %xmm0, %eax
        mov     $0x03f490fdb,%edx   /* pi / 4 */
        movss   .L__sngl_sixteen_by_pi(%rip),%xmm4
        and     .L__sngl_mask_unsign(%rip), %eax
        cmp     %edx,%eax
        jle     LBL(.L__fmth_sin_shortcuts)
        shrl    $20,%eax
        cmpl    $0x498,%eax
        jge     GBLTXT(ENT(__mth_i_sin))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulss   %xmm0,%xmm4 
#ifdef GH_TARGET
        unpcklps %xmm0, %xmm0
        cvtps2pd %xmm0, %xmm0
#else
        cvtss2sd %xmm0,%xmm0
#endif

        /* Set n = nearest integer to r */
        cvtss2si %xmm4,%rcx    /* convert to integer */
        movsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        movsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movsd   .L__dble_pi_by_16_us(%rip), %xmm3
        cvtsi2sd %rcx,%xmm4    /* and back to double */

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulsd   %xmm4,%xmm1     /* n * p1 */
        mulsd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        subsd   %xmm1,%xmm0     /* x - n * p1 == rh */
	addsd   %xmm2,%xmm3

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subsd   %xmm3,%xmm0     /* c = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        movsdRR   %xmm0,%xmm1     /* r in xmm1 */
        movsdRR   %xmm0,%xmm2     /* r in xmm2 */
        movsdRR   %xmm0,%xmm4     /* r in xmm4 */
        mulsd   %xmm0,%xmm0     /* r^2 in xmm0 */
        mulsd   %xmm1,%xmm1     /* r^2 in xmm1 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm2,%xmm3     /* r in xmm3 */

        /* xmm0, xmm1, xmm4 have r^2, xmm2, xmm3 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        mulsd   .L__dble_pq3(%rip), %xmm0     /* p4 * r^2 */
        mulsd   .L__dble_pq3+16(%rip), %xmm1  /* q4 * r^2 */
        addsd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm1  /* + q2 */
        mulsd   %xmm4,%xmm0                   /* * r^2 */
        mulsd   %xmm4,%xmm1                   /* * r^2 */

        mulsd   %xmm4,%xmm3                   /* xmm3 = r^3 */
        addsd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm1  /* + q1 */
        mulsd   %xmm3,%xmm0                   /* * r^3 */
        mulsd   %xmm4,%xmm1                   /* * r^2 */

        addq    %rax,%rax
        addq    %rcx,%rcx
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */

        addsd   %xmm2,%xmm0                   /* + r */
        mulsd   (%rdx,%rcx,8),%xmm0           /* C * p(r) */
        mulsd   (%rdx,%rax,8),%xmm1           /* S * q(r) */
        addsd   (%rdx,%rax,8),%xmm1           /* S + S * q(r) */
        addsd   %xmm1,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

LBL(.L__fmth_sin_shortcuts):
#ifdef GH_TARGET
        unpcklps %xmm0, %xmm0
        cvtps2pd %xmm0, %xmm0
#else
        cvtss2sd %xmm0,%xmm0
#endif
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrl    $20,%eax
        cmpl    $0x0390,%eax
        jl      LBL(.L__fmth_sin_small)
        mulsd   %xmm0,%xmm0
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_dsin_c4(%rip),%xmm0    /* x2 * c4 */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + c3 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + c2 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c2 + ...) */
        mulsd   %xmm2,%xmm1                     /* x3 */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + c1 */
        mulsd   %xmm1,%xmm0                     /* x3 * (c1 + ...) */
        addsd   %xmm2,%xmm0                     /* x + x3 * (...) done */
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

LBL(.L__fmth_sin_small):
        cmpl    $0x0320,%eax
        jl      LBL(.L__fmth_sin_done1)
        /* return x - x * x * x * 1/3! */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1(%rip),%xmm2
        mulsd   %xmm2,%xmm1
        addsd   %xmm1,%xmm0

LBL(.L__fmth_sin_done1):
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

        ELF_FUNC(ENT_GH(__fmth_i_sin))
        ELF_SIZE(ENT_GH(__fmth_i_sin))
        IF_GH(ELF_FUNC(__fss_sin))
        IF_GH(ELF_SIZE(__fss_sin))


/* ============================================================ */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fsd_sin))
        .globl ENT_GH(__fmth_i_dsin)
IF_GH(ENT(__fsd_sin):)
ENT_GH(__fmth_i_dsin):

        movd    %xmm0, %rax
        mov     $0x03fe921fb54442d18,%rdx
        movapd  .L__dble_sixteen_by_pi(%rip),%xmm4
        andq    .L__real_mask_unsign(%rip), %rax
        cmpq    %rdx,%rax
        jle     LBL(.L__fmth_dsin_shortcuts)
        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     GBLTXT(ENT(__mth_i_dsin))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulsd   %xmm0,%xmm4 

        RZ_PUSH
#if defined(_WIN64)
        movdqa  %xmm6, RZ_OFF(24)(%rsp)
        movdqa  %xmm7, RZ_OFF(40)(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5    /* convert to integer */
        movsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        movsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movsd   .L__dble_pi_by_16_us(%rip), %xmm3
        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulsd   %xmm4,%xmm1     /* n * p1 */
        mulsd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        movsdRR   %xmm0,%xmm6     /* x in xmm6 */
        subsd   %xmm1,%xmm0     /* x - n * p1 == rh */
        subsd   %xmm1,%xmm6     /* x - n * p1 == rh == c */

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subsd   %xmm2,%xmm0     /* rh = rh - rt */

        subsd   %xmm0,%xmm6     /* (c - rh) */
        movsdRR   %xmm0,%xmm1     /* Move rh */
        movsdRR   %xmm0,%xmm4     /* Move rh */
        movsdRR   %xmm0,%xmm5     /* Move rh */
        subsd   %xmm2,%xmm6     /* ((c - rh) - rt) */
        subsd   %xmm6,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        movsdRR   %xmm1,%xmm2     /* Move rh */
        subsd   %xmm3,%xmm0     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm4     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm5     /* c = rh - rt aka r */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */
        
        subsd   %xmm0,%xmm1     /* (rh - c) */

        mulsd   %xmm0,%xmm0     /* r^2 in xmm0 */
        movsdRR   %xmm4,%xmm6     /* r in xmm6 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm5,%xmm7     /* r in xmm7 */
        mulsd   %xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulsd   .L__dble_pq4(%rip), %xmm0     /* p4 * r^2 */
        subsd   %xmm6,%xmm2                   /* (rh - c) */
        mulsd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        subsd   %xmm3,%xmm1                   /* (rh - c) - rt aka rr */

        addsd   .L__dble_pq3(%rip), %xmm0     /* + p3 */
        addsd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        subsd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */

        mulsd   %xmm5,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        mulsd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        mulsd   %xmm5,%xmm7                   /* xmm7 = r^3 */
        movsdRR   %xmm1,%xmm3                   /* Move rr */
        mulsd   %xmm5,%xmm1                   /* r * r * rr */

        addsd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        mulsd   .L__dble_pq1+16(%rip), %xmm1  /* r * r * rr * 0.5 */
        mulsd   %xmm6, %xmm3                  /* r * rr */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        mulsd   %xmm5,%xmm0                   /* * r^2 */
        mulsd   %xmm5,%xmm4                   /* * r^2 */
        addsd   %xmm1,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        movlpdMR  8(%rdx,%rax,8),%xmm1          /* ds2 in xmm1 */
        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

        addsd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulsd   %xmm7,%xmm0                   /* * r^3 */
        mulsd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addsd   %xmm2,%xmm0                   /* + cs  == dp aka p(r) */
        subsd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */
        movlpdMR  8(%rdx,%rcx,8),%xmm3          /* dc2 in xmm3 */
        movlpdMR   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        addsd   %xmm6,%xmm0                   /* + r   == dp aka p(r) */
        movsdRR   %xmm1,%xmm2                   /* ds2 in xmm2 */

        mulsd   %xmm4,%xmm1                   /* ds2 * dq */
        mulsd   %xmm0,%xmm3                   /* dc2 * dp */
        addsd   %xmm2,%xmm1                   /* ds2 + ds2*dq */
        mulsd   %xmm5,%xmm4                   /* ds1 * dq */
        addsd   %xmm3,%xmm1                   /* (ds2 + ds2*dq) + dc2*dp */
        mulsd   (%rdx,%rcx,8),%xmm0           /* dc1 * dp */
        addsd   %xmm4,%xmm1                   /* ((ds2...) + dc2*dp) + ds1*dq */
        addsd   %xmm5,%xmm1

#if defined(_WIN64)
        movdqa  RZ_OFF(24)(%rsp),%xmm6
        movdqa  RZ_OFF(40)(%rsp),%xmm7
#endif
        addsd   %xmm1,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        RZ_POP
        ret

LBL(.L__fmth_dsin_shortcuts):
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrq    $48,%rax
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fmth_dsin_small)
        mulsd   %xmm0,%xmm0
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_dsin_c6(%rip),%xmm0    /* x2 * c6 */
        addsd   .L__dble_dsin_c5(%rip),%xmm0    /* + c5 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dsin_c4(%rip),%xmm0    /* + c4 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + c3 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + c2 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c2 + ...) */
        mulsd   %xmm2,%xmm1                     /* x3 */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + c1 */
        mulsd   %xmm1,%xmm0                     /* x3 * (c1 + ...) */
        addsd   %xmm2,%xmm0                     /* x + x3 * (...) done */
        ret

LBL(.L__fmth_dsin_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fmth_dsin_done1)
        /* return x - x * x * x * 1/3! */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1(%rip),%xmm2
        mulsd   %xmm2,%xmm1
        addsd   %xmm1,%xmm0
        ret

LBL(.L__fmth_dsin_done1):
	rep
        ret
        
        ELF_FUNC(ENT_GH(__fmth_i_dsin))
        ELF_SIZE(ENT_GH(__fmth_i_dsin))
        IF_GH(ELF_FUNC(__fsd_sin))
        IF_GH(ELF_SIZE(__fsd_sin))


/* ------------------------------------------------------------------------- */
/* 
 *  vector sine
 * 
 *  An implementation of the sine libm function.
 * 
 *  Prototype:
 * 
 *      double __fvssin(float *x);
 * 
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fvs_sin))
        .globl ENT_GH(__fvssin)
IF_GH(ENT(__fvs_sin):)
ENT_GH(__fvssin):
	movaps	%xmm0, %xmm1		/* Move input vector */
        andps   .L__sngl_mask_unsign(%rip), %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

        movlps  .L__sngl_pi_over_fours(%rip),%xmm2
        movhps  .L__sngl_pi_over_fours(%rip),%xmm2
        movlps  .L__sngl_needs_argreds(%rip),%xmm3
        movhps  .L__sngl_needs_argreds(%rip),%xmm3
        movlps  .L__sngl_sixteen_by_pi(%rip),%xmm4
        movhps  .L__sngl_sixteen_by_pi(%rip),%xmm4

	cmpps   $5, %xmm0, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	cmpps   $2, %xmm0, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        movmskps %xmm2, %eax
        movmskps %xmm3, %ecx

	test	$15, %eax
        jnz	LBL(.L__Scalar_fvsin1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulps   %xmm1,%xmm4 

	test	$15, %ecx
        jnz	LBL(.L__Scalar_fvsin2)

        /* Set n = nearest integer to r */
	movhps	%xmm1,(%rsp)                     /* Store x4, x3 */

	xorq	%r10, %r10
	cvtps2pd %xmm1, %xmm1

        cvtps2dq %xmm4,%xmm5    /* convert to integer, n4,n3,n2,n1 */
LBL(.L__fvsin_do_twice):
#ifdef GH_TARGET
         movddup   .L__dble_pi_by_16_ms(%rip), %xmm0
         movddup   .L__dble_pi_by_16_ls(%rip), %xmm2
         movddup   .L__dble_pi_by_16_us(%rip), %xmm3
#else
        movlpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movhpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movlpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movhpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movlpd   .L__dble_pi_by_16_us(%rip), %xmm3
        movhpd   .L__dble_pi_by_16_us(%rip), %xmm3
#endif
        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - (n*p2 + n*p3) */
        mulpd   %xmm4,%xmm0     /* n * p1 */
        mulpd   %xmm4,%xmm2   /* n * p2 == rt */
        mulpd   %xmm4,%xmm3   /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        subpd   %xmm0,%xmm1   /* x - n * p1 == rh */
	addpd   %xmm2,%xmm3 

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subpd   %xmm3,%xmm1   /* c = rh - rt aka r */

        shrq    $32, %r9
        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

	movapd  %xmm1,%xmm0   /* r in xmm0 and xmm1 */
        movapd  %xmm1,%xmm2   /* r in xmm2 */
        movapd  %xmm1,%xmm4   /* r in xmm4 */
        mulpd   %xmm1,%xmm1   /* r^2 in xmm1 */
        mulpd   %xmm0,%xmm0   /* r^2 in xmm0 */
        mulpd   %xmm4,%xmm4   /* r^2 in xmm4 */
        movapd  %xmm2,%xmm3   /* r in xmm2 and xmm3 */

        leaq    24(%r9),%r8   /* Add 24 for sine */
        andq    $0x1f,%r8     /* And lower 5 bits */
        andq    $0x1f,%r9     /* And lower 5 bits */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulpd   .L__dble_pq3(%rip), %xmm0     /* p3 * r^2 */
        mulpd   .L__dble_pq3+16(%rip), %xmm1  /* q3 * r^2 */

        movq    %r8, %rdx     /* Duplicate it */
        sarq    $4,%r8        /* Sign bits moved down */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r8     /* Final tbl address */

        addpd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addpd   .L__dble_pq2+16(%rip), %xmm1  /* + q2 */

        movq    %r9, %rdx     /* Duplicate it */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r9     /* Final tbl address */

        mulpd   %xmm4,%xmm0                   /* * r^2 */
        mulpd   %xmm4,%xmm1                   /* * r^2 */

        mulpd   %xmm4,%xmm3                   /* xmm3 = r^3 */
        addpd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addpd   .L__dble_pq1+16(%rip), %xmm1  /* + q1 */

        addq    %rax,%rax
        addq    %r8,%r8
        addq    %rcx,%rcx
        addq    %r9,%r9

	mulpd   %xmm3,%xmm0                   /* * r^3 */
	mulpd   %xmm4,%xmm1                   /* * r^2  = q(r) */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        movlpdMR  (%rdx,%rax,8),%xmm4           /* S in xmm4 */
        movhpd  (%rdx,%r8,8),%xmm4            /* S in xmm4 */

        movlpdMR  (%rdx,%rcx,8),%xmm3           /* C in xmm3 */
        movhpd  (%rdx,%r9,8),%xmm3            /* C in xmm3 */

	addpd   %xmm2,%xmm0                   /* + r = p(r) */

	mulpd   %xmm4, %xmm1                  /* S * q(r) */
	mulpd   %xmm3, %xmm0                  /* C * p(r) */
	addpd   %xmm4, %xmm1                  /* S + S * q(r) */
	addpd   %xmm1, %xmm0                  /* sin(x) = Cp(r) + (S+Sq(r)) */

	cvtpd2ps %xmm0,%xmm0
	cmp	$0, %r10                      /* Compare loop count */
	shufps	$78, %xmm0, %xmm5             /* sin(x2), sin(x1), n4, n3 */
	jne 	LBL(.L__fvsin_done_twice)
	inc 	%r10
	cvtps2pd (%rsp),%xmm1
	jmp 	LBL(.L__fvsin_do_twice)

LBL(.L__fvsin_done_twice):
	movaps  %xmm5, %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsin1):
        /* Here when at least one argument is less than pi/4,
           or, at least one is a Nan.  What we will do for now, is
           if all are less than pi/4, do them all.  Otherwise, call
           fmth_i_sin or mth_i_sin one at a time.
        */
        movaps  %xmm0, (%rsp)                 /* Save xmm0, masked x */
	cmpps   $3, %xmm0, %xmm0              /* 3 is "unordered" */
        movaps  %xmm1, 16(%rsp)               /* Save xmm1, input x */
        movmskps %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

	cmp	$15, %eax
	jne	LBL(.L__Scalar_fvsin1a)

	cvtps2pd 16(%rsp),%xmm0               /* x(2), x(1) */
	cvtps2pd 24(%rsp),%xmm1               /* x(4), x(3) */

        movapd  %xmm0,16(%rsp)
        movapd  %xmm1,32(%rsp)
	mulpd   %xmm0,%xmm0                   /* x2 for x(2), x(1) */
	mulpd   %xmm1,%xmm1                   /* x2 for x(4), x(3) */

#ifdef GH_TARGET
         movddup  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
         movddup  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */
#else
        movlpd  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
        movhpd  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
        movlpd  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */
        movhpd  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */
#endif


        movapd  %xmm0,%xmm2
        movapd  %xmm1,%xmm3
        mulpd   %xmm4,%xmm0                   /* x2 * c4 */
        mulpd   %xmm4,%xmm1                   /* x2 * c4 */
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
#else
        movlpd  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
        movhpd  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
#endif

        addpd   %xmm5,%xmm0                   /* + c3 */
        addpd   %xmm5,%xmm1                   /* + c3 */
        movapd  .L__dble_pq1(%rip),%xmm5      /* c1 */
        mulpd   %xmm2,%xmm0                   /* x2 * (c3 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c3 + ...) */

        addpd   %xmm4,%xmm0                   /* + c2 */
        addpd   %xmm4,%xmm1                   /* + c2 */
        mulpd   %xmm2,%xmm0                   /* x2 * (c2 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c2 + ...) */
	mulpd   16(%rsp),%xmm2                /* x3 */
	mulpd   32(%rsp),%xmm3                /* x3 */
        addpd   %xmm5,%xmm0                   /* + c1 */
        addpd   %xmm5,%xmm1                   /* + c1 */
        mulpd   %xmm2,%xmm0                   /* x3 * (c1 + ...) */
        mulpd   %xmm3,%xmm1                   /* x3 * (c1 + ...) */

        addpd   16(%rsp),%xmm0                /* x + x3 * (...) done */
        addpd   32(%rsp),%xmm1                /* x + x3 * (...) done */
        cvtpd2ps %xmm0,%xmm0            /* sin(x2), sin(x1) */
        cvtpd2ps %xmm1,%xmm1            /* sin(x4), sin(x3) */
	shufps	$68, %xmm1, %xmm0       /* sin(x4),sin(x3),sin(x2),sin(x1) */

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsin1a):
	test    $1, %eax
	jz	LBL(.L__Scalar_fvsin3)
#ifdef GH_TARGET
	movss 16(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 16(%rsp),%xmm0               /* dble x(1) */
#endif
	movl	(%rsp),%edx
	call	LBL(.L__fmth_fvsin_local)
	jmp	LBL(.L__Scalar_fvsin5)

LBL(.L__Scalar_fvsin2):
        movaps  %xmm0, (%rsp)                 /* Save xmm0 */
        movaps  %xmm1, %xmm0                  /* Save xmm1 */
        movaps  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvsin3):
	movss   16(%rsp),%xmm0                /* x(1) */
	test    $1, %ecx
	jz	LBL(.L__Scalar_fvsin4)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_sin))                /* Here when big or a nan */
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvsin5)

LBL(.L__Scalar_fvsin4):
	mov     %eax, 32(%rsp)                /* Here when a scalar will do */
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fss_sin))
#else
	CALL(ENT(__fmth_i_sin))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsin5):
        movss   %xmm0, (%rsp)                 /* Move first result */

	test    $2, %eax
	jz	LBL(.L__Scalar_fvsin6)
#ifdef GH_TARGET
	movss 20(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 20(%rsp),%xmm0               /* dble x(2) */
#endif
	movl	4(%rsp),%edx
	call	LBL(.L__fmth_fvsin_local)
	jmp	LBL(.L__Scalar_fvsin8)

LBL(.L__Scalar_fvsin6):
	movss   20(%rsp),%xmm0                /* x(2) */
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvsin7)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_sin))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvsin8)

LBL(.L__Scalar_fvsin7):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fss_sin))
#else
	CALL(ENT(__fmth_i_sin))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsin8):
        movss   %xmm0, 4(%rsp)               /* Move 2nd result */

	test    $4, %eax
	jz	LBL(.L__Scalar_fvsin9)
#ifdef GH_TARGET
	movss 24(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 24(%rsp),%xmm0               /* dble x(3) */
#endif
	movl	8(%rsp),%edx
	call	LBL(.L__fmth_fvsin_local)
	jmp	LBL(.L__Scalar_fvsin11)

LBL(.L__Scalar_fvsin9):
	movss   24(%rsp),%xmm0                /* x(3) */
	test    $4, %ecx
	jz	LBL(.L__Scalar_fvsin10)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_sin))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvsin11)

LBL(.L__Scalar_fvsin10):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fss_sin))
#else
	CALL(ENT(__fmth_i_sin))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsin11):
        movss   %xmm0, 8(%rsp)               /* Move 3rd result */

	test    $8, %eax
	jz	LBL(.L__Scalar_fvsin12)
#ifdef GH_TARGET
	movss 28(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 28(%rsp),%xmm0               /* dble x(4) */
#endif
	movl	12(%rsp),%edx
	call	LBL(.L__fmth_fvsin_local)
	jmp	LBL(.L__Scalar_fvsin14)

LBL(.L__Scalar_fvsin12):
	movss   28(%rsp),%xmm0                /* x(4) */
	test    $8, %ecx
	jz	LBL(.L__Scalar_fvsin13)
	CALL(ENT(__mth_i_sin))
	jmp	LBL(.L__Scalar_fvsin14)

LBL(.L__Scalar_fvsin13):
#ifdef GH_TARGET
	CALL(ENT(__fss_sin))
#else
	CALL(ENT(__fmth_i_sin))
#endif

/* ---------------------------------- */
LBL(.L__Scalar_fvsin14):
        movss   %xmm0, 12(%rsp)               /* Move 4th result */
	movaps	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fmth_fvsin_local):
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrl    $20,%edx
        cmpl    $0x0390,%edx
        jl      LBL(.L__fmth_fvsin_small)
        mulsd   %xmm0,%xmm0
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_dsin_c4(%rip),%xmm0    /* x2 * c4 */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + c3 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + c2 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c2 + ...) */
        mulsd   %xmm2,%xmm1                     /* x3 */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + c1 */
        mulsd   %xmm1,%xmm0                     /* x3 * (c1 + ...) */
        addsd   %xmm2,%xmm0                     /* x + x3 * (...) done */
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

LBL(.L__fmth_fvsin_small):
        cmpl    $0x0320,%edx
        jl      LBL(.L__fmth_fvsin_done1)
        /* return x - x * x * x * 1/3! */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1(%rip),%xmm2
        mulsd   %xmm2,%xmm1
        addsd   %xmm1,%xmm0

LBL(.L__fmth_fvsin_done1):
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

        ELF_FUNC(ENT_GH(__fvssin))
        ELF_SIZE(ENT_GH(__fvssin))
        IF_GH(ELF_FUNC(__fvs_sin))
        IF_GH(ELF_SIZE(__fvs_sin))


/* ============================================================
 * 
 *  vector sine
 * 
 *  An implementation of the sine libm function.
 * 
 *  Prototype:
 * 
 *      double __fvdsin(double *x);
 * 
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fvd_sin))
        .globl ENT_GH(__fvdsin)
IF_GH(ENT(__fvd_sin):)
ENT_GH(__fvdsin):
	movapd	%xmm0, %xmm1		/* Move input vector */
        andpd   .L__real_mask_unsign(%rip), %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

#ifdef GH_TARGET
         movddup  .L__dble_pi_over_fours(%rip),%xmm2
         movddup  .L__dble_needs_argreds(%rip),%xmm3
         movddup  .L__dble_sixteen_by_pi(%rip),%xmm4
#else
        movlpd  .L__dble_pi_over_fours(%rip),%xmm2
        movhpd  .L__dble_pi_over_fours(%rip),%xmm2
        movlpd  .L__dble_needs_argreds(%rip),%xmm3
        movhpd  .L__dble_needs_argreds(%rip),%xmm3
        movlpd  .L__dble_sixteen_by_pi(%rip),%xmm4
        movhpd  .L__dble_sixteen_by_pi(%rip),%xmm4
#endif

	cmppd   $5, %xmm0, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	cmppd   $2, %xmm0, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        movmskpd %xmm2, %eax
        movmskpd %xmm3, %ecx

	test	$3, %eax
        jnz	LBL(.L__Scalar_fvdsin1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulpd   %xmm1,%xmm4 

	test	$3, %ecx
        jnz	LBL(.L__Scalar_fvdsin2)

#if defined(_WIN64)
        movdqa  %xmm6, (%rsp)
        movdqa  %xmm7, 16(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5    /* convert to integer */
#ifdef GH_TARGET
         movddup   .L__dble_pi_by_16_ms(%rip), %xmm0
         movddup   .L__dble_pi_by_16_ls(%rip), %xmm2
         movddup   .L__dble_pi_by_16_us(%rip), %xmm3
#else
        movlpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movhpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movlpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movhpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movlpd   .L__dble_pi_by_16_us(%rip), %xmm3
        movhpd   .L__dble_pi_by_16_us(%rip), %xmm3
#endif

        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulpd   %xmm4,%xmm0     /* n * p1 */
        mulpd   %xmm4,%xmm2   /* n * p2 == rt */
        mulpd   %xmm4,%xmm3   /* n * p3 */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */

        /* How to convert N into a table address */
        movapd  %xmm1,%xmm6   /* x in xmm6 */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        subpd   %xmm0,%xmm1   /* x - n * p1 == rh */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        subpd   %xmm0,%xmm6   /* x - n * p1 == rh == c */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        subpd   %xmm2,%xmm1   /* rh = rh - rt */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */
        subpd   %xmm1,%xmm6   /* (c - rh) */
        movq    %rax, %rdx    /* Duplicate it */
        movapd  %xmm1,%xmm0   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        movapd  %xmm1,%xmm4   /* Move rh */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        movapd  %xmm1,%xmm5   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        subpd   %xmm2,%xmm6   /* ((c - rh) - rt) */
        andq    $0xf,%rdx     /* And lower 5 bits */
        subpd   %xmm6,%xmm3   /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        addq    %rdx, %rax    /* Final tbl address */
        movapd  %xmm1,%xmm2   /* Move rh */
        shrq    $32, %r9
        subpd   %xmm3,%xmm0   /* c = rh - rt aka r */
        movq    %rcx, %rdx    /* Duplicate it */
        subpd   %xmm3,%xmm4   /* c = rh - rt aka r */
        sarq    $4,%rcx       /* Sign bits moved down */
        subpd   %xmm3,%xmm5   /* c = rh - rt aka r */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        subpd   %xmm0,%xmm1   /* (rh - c) */
        sarq    $4,%rcx       /* Sign bits moved down */
        mulpd   %xmm0,%xmm0   /* r^2 in xmm0 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        movapd  %xmm4,%xmm6   /* r in xmm6 */
        addq    %rdx, %rcx    /* Final tbl address */
        mulpd   %xmm4,%xmm4   /* r^2 in xmm4 */
        leaq    24(%r9),%r8   /* Add 24 for sine */
        movapd  %xmm5,%xmm7   /* r in xmm7 */
        andq    $0x1f,%r8     /* And lower 5 bits */
        mulpd   %xmm5,%xmm5   /* r^2 in xmm5 */
        andq    $0x1f,%r9     /* And lower 5 bits */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulpd   .L__dble_pq4(%rip), %xmm0     /* p4 * r^2 */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        subpd   %xmm6,%xmm2                   /* (rh - c) */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        mulpd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        subpd   %xmm3,%xmm1                   /* (rh - c) - rt aka rr */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */
        addpd   .L__dble_pq3(%rip), %xmm0     /* + p3 */
        movq    %r8, %rdx     /* Duplicate it */
        addpd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        sarq    $4,%r8        /* Sign bits moved down */
        subpd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        mulpd   %xmm5,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        sarq    $4,%r8        /* Sign bits moved down */
        mulpd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        mulpd   %xmm5,%xmm7                   /* xmm7 = r^3 */
        addq    %rdx, %r8     /* Final tbl address */
        movapd  %xmm1,%xmm3                   /* Move rr */
        movq    %r9, %rdx     /* Duplicate it */
        mulpd   %xmm5,%xmm1                   /* r * r * rr */
        sarq    $4,%r9        /* Sign bits moved down */
        addpd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        addpd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        sarq    $4,%r9        /* Sign bits moved down */
        mulpd   .L__dble_pq1+16(%rip), %xmm1  /* r * r * rr * 0.5 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        mulpd   %xmm6, %xmm3                  /* r * rr */
        addq    %rdx, %r9     /* Final tbl address */
        mulpd   %xmm5,%xmm0                   /* * r^2 */
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        mulpd   %xmm5,%xmm4                   /* * r^2 */
        addq    %rax,%rax
        addpd   %xmm1,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        addq    %r8,%r8
        movlpdMR  8(%rdx,%rax,8),%xmm1          /* ds2 in xmm1 */
        movhpd  8(%rdx,%r8,8),%xmm1           /* ds2 in xmm1 */


        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

        addpd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addpd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulpd   %xmm7,%xmm0                   /* * r^3 */
        mulpd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addpd   %xmm2,%xmm0                   /* + cs  == dp aka p(r) */
        addq    %rcx,%rcx
        subpd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */
        addq    %r9,%r9
        movlpdMR  8(%rdx,%rcx,8),%xmm3          /* dc2 in xmm3 */
        movhpd  8(%rdx,%r9,8),%xmm3           /* dc2 in xmm3 */

        movlpdMR   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        movhpd   (%rdx,%r8,8),%xmm5           /* ds1 in xmm5 */

        addpd   %xmm6,%xmm0                   /* + r   == dp aka p(r) */
        movapd  %xmm1,%xmm2                   /* ds2 in xmm2 */
        movlpdMR  (%rdx,%rcx,8),%xmm6           /* dc1 in xmm6 */
        movhpd  (%rdx,%r9,8),%xmm6            /* dc1 in xmm6 */


        mulpd   %xmm4,%xmm1                   /* ds2 * dq */
        mulpd   %xmm0,%xmm3                   /* dc2 * dp */
        addpd   %xmm2,%xmm1                   /* ds2 + ds2*dq */
        mulpd   %xmm5,%xmm4                   /* ds1 * dq */
        addpd   %xmm3,%xmm1                   /* (ds2 + ds2*dq) + dc2*dp */
        mulpd   %xmm6,%xmm0                   /* dc1 * dp */
        addpd   %xmm4,%xmm1                   /* ((ds2...) + dc2*dp) + ds1*dq */

#if defined(_WIN64)
        movdqa  (%rsp),%xmm6
        movdqa  16(%rsp),%xmm7
#endif
        addpd   %xmm5,%xmm1
        addpd   %xmm1,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsin1):
        movapd  %xmm0, (%rsp)                 /* Save xmm0 */
	cmppd   $3, %xmm0, %xmm0              /* 3 is "unordered" */
        movapd  %xmm1, 16(%rsp)               /* Save xmm1 */
        movmskpd %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

        movapd  16(%rsp), %xmm0
	test    $1, %eax
	jz	LBL(.L__Scalar_fvdsin3)
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsin1a)

        movapd  %xmm0,%xmm1
        movapd  %xmm0,%xmm2
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c6(%rip),%xmm3    /* c6 */
#else
        movlpd  .L__dble_dsin_c6(%rip),%xmm3    /* c6 */
        movhpd  .L__dble_dsin_c6(%rip),%xmm3    /* c6 */
#endif

        mulpd   %xmm0,%xmm0
        mulpd   %xmm1,%xmm1
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c5(%rip),%xmm4    /* c5 */
#else
        movlpd  .L__dble_dsin_c5(%rip),%xmm4    /* c5 */
        movhpd  .L__dble_dsin_c5(%rip),%xmm4    /* c5 */
#endif

        mulpd   %xmm3,%xmm0                     /* x2 * c6 */
        addpd   %xmm4,%xmm0                     /* + c5 */
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c4(%rip),%xmm3    /* c4 */
#else
        movlpd  .L__dble_dsin_c4(%rip),%xmm3    /* c4 */
        movhpd  .L__dble_dsin_c4(%rip),%xmm3    /* c4 */
#endif

        mulpd   %xmm1,%xmm0                     /* x2 * (c5 + ...) */
        addpd   %xmm3,%xmm0                     /* + c4 */
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c3(%rip),%xmm4    /* c3 */
#else
        movlpd  .L__dble_dsin_c3(%rip),%xmm4    /* c3 */
        movhpd  .L__dble_dsin_c3(%rip),%xmm4    /* c3 */
#endif

        mulpd   %xmm1,%xmm0                     /* x2 * (c4 + ...) */
        addpd   %xmm4,%xmm0                     /* + c3 */
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c2(%rip),%xmm3    /* c2 */
#else
        movlpd  .L__dble_dsin_c2(%rip),%xmm3    /* c2 */
        movhpd  .L__dble_dsin_c2(%rip),%xmm3    /* c2 */
#endif

        mulpd   %xmm1,%xmm0                     /* x2 * (c3 + ...) */
        addpd   %xmm3,%xmm0                     /* + c2 */
        mulpd   %xmm1,%xmm0                     /* x2 * (c2 + ...) */
        mulpd   %xmm2,%xmm1                     /* x3 */
        addpd   .L__dble_pq1(%rip),%xmm0        /* + c1 */
        mulpd   %xmm1,%xmm0                     /* x3 * (c1 + ...) */
        addpd   %xmm2,%xmm0                     /* x + x3 * (...) done */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsin1a):
	movq	(%rsp),%rdx
	call	LBL(.L__fmth_fvdsin_local)
	jmp	LBL(.L__Scalar_fvdsin5)

LBL(.L__Scalar_fvdsin2):
        movapd  %xmm0, (%rsp)                 /* Save xmm0 */
        movapd  %xmm1, %xmm0                  /* Save xmm1 */
        movapd  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvdsin3):
	test    $1, %ecx
	jz	LBL(.L__Scalar_fvdsin4)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_dsin))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvdsin5)

LBL(.L__Scalar_fvdsin4):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fsd_sin))
#else
	CALL(ENT(__fmth_i_dsin))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

LBL(.L__Scalar_fvdsin5):
        movlpd  %xmm0, (%rsp)
        movlpdMR  24(%rsp), %xmm0
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsin6)
	movq	8(%rsp),%rdx
	call	LBL(.L__fmth_fvdsin_local)
	jmp	LBL(.L__Scalar_fvdsin8)

LBL(.L__Scalar_fvdsin6):
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvdsin7)
	CALL(ENT(__mth_i_dsin))
	jmp	LBL(.L__Scalar_fvdsin8)

LBL(.L__Scalar_fvdsin7):
#ifdef GH_TARGET
	CALL(ENT(__fsd_sin))
#else
	CALL(ENT(__fmth_i_dsin))
#endif

LBL(.L__Scalar_fvdsin8):
        movlpd  %xmm0, 8(%rsp)
	movapd	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fmth_fvdsin_local):
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrq    $48,%rdx
        cmpl    $0x03f20,%edx
        jl      LBL(.L__fmth_fvdsin_small)
        mulsd   %xmm0,%xmm0
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_dsin_c6(%rip),%xmm0    /* x2 * c6 */
        addsd   .L__dble_dsin_c5(%rip),%xmm0    /* + c5 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dsin_c4(%rip),%xmm0    /* + c4 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + c3 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + c2 */
        mulsd   %xmm1,%xmm0                     /* x2 * (c2 + ...) */
        mulsd   %xmm2,%xmm1                     /* x3 */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + c1 */
        mulsd   %xmm1,%xmm0                     /* x3 * (c1 + ...) */
        addsd   %xmm2,%xmm0                     /* x + x3 * (...) done */
        ret

LBL(.L__fmth_fvdsin_small):
        cmpl    $0x03e40,%edx
        jl      LBL(.L__fmth_fvdsin_done1)
        /* return x - x * x * x * 1/3! */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1(%rip),%xmm2
        mulsd   %xmm2,%xmm1
        addsd   %xmm1,%xmm0
        ret

LBL(.L__fmth_fvdsin_done1):
	rep
        ret
        
        ELF_FUNC(ENT_GH(__fvdsin))
        ELF_SIZE(ENT_GH(__fvdsin))
        IF_GH(ELF_FUNC(__fvd_sin))
        IF_GH(ELF_SIZE(__fvd_sin))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fss_cos))
        .globl ENT_GH(__fmth_i_cos)
IF_GH(ENT(__fss_cos):)
ENT_GH(__fmth_i_cos):
        movd    %xmm0, %eax
        mov     $0x03f490fdb,%edx   /* pi / 4 */
        movss   .L__sngl_sixteen_by_pi(%rip),%xmm4
        and     .L__sngl_mask_unsign(%rip), %eax
        cmp     %edx,%eax
        jle     LBL(.L__fmth_cos_shortcuts)
        shrl    $20,%eax
        cmpl    $0x498,%eax
        jge     GBLTXT(ENT(__mth_i_cos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulss   %xmm0,%xmm4 
#ifdef GH_TARGET
        unpcklps %xmm0, %xmm0
        cvtps2pd %xmm0, %xmm0
#else
        cvtss2sd %xmm0,%xmm0
#endif

        /* Set n = nearest integer to r */
        cvtss2si %xmm4,%rcx    /* convert to integer */
        movsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        movsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movsd   .L__dble_pi_by_16_us(%rip), %xmm3
        cvtsi2sd %rcx,%xmm4    /* and back to double */

        /* r = (x - n*p1) - (n*p2 + n*p3)  */
        mulsd   %xmm4,%xmm1     /* n * p1 */
        mulsd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        subsd   %xmm1,%xmm0     /* x - n * p1 == rh */
	addsd   %xmm2,%xmm3

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subsd   %xmm3,%xmm0     /* c = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        movsdRR   %xmm0,%xmm1     /* r in xmm1 */
        movsdRR   %xmm0,%xmm2     /* r in xmm2 */
        movsdRR   %xmm0,%xmm4     /* r in xmm4 */
        mulsd   %xmm0,%xmm0     /* r^2 in xmm0 */
        mulsd   %xmm1,%xmm1     /* r^2 in xmm1 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm2,%xmm3     /* r in xmm3 */

        /* xmm0, xmm1, xmm4 have r^2, xmm2, xmm3 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        mulsd   .L__dble_pq3(%rip), %xmm0     /* p4 * r^2 */
        mulsd   .L__dble_pq3+16(%rip), %xmm1  /* q4 * r^2 */
        addsd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm1  /* + q2 */
        mulsd   %xmm4,%xmm0                   /* * r^2 */
        mulsd   %xmm4,%xmm1                   /* * r^2 */

        mulsd   %xmm4,%xmm3                   /* xmm3 = r^3 */
        addsd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm1  /* + q1 */
        mulsd   %xmm3,%xmm0                   /* * r^3 */
        mulsd   %xmm4,%xmm1                   /* * r^2 */

        addq    %rax,%rax
        addq    %rcx,%rcx
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */

        addsd   %xmm2,%xmm0                   /* + r  = p(r) */
        mulsd   (%rdx,%rcx,8),%xmm1           /* C * q(r) */
        mulsd   (%rdx,%rax,8),%xmm0           /* S * p(r) */
        addsd   (%rdx,%rcx,8),%xmm1           /* C + C * q(r) */
        subsd   %xmm0,%xmm1                   /* cos(x) = (C + Cq(r)) - Sp(r) */
#ifdef GH_TARGET
	unpcklpd %xmm1, %xmm1
	cvtpd2ps %xmm1, %xmm0
#else
	cvtsd2ss %xmm1,%xmm0
#endif
        ret

LBL(.L__fmth_cos_shortcuts):
#ifdef GH_TARGET
        unpcklps %xmm0, %xmm0
        cvtps2pd %xmm0, %xmm0
#else
        cvtss2sd %xmm0,%xmm0
#endif
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrl    $20,%eax
	movlpdMR  .L__dble_sincostbl(%rip), %xmm0  /* 1.0 */
        cmpl    $0x0390,%eax
        jl      LBL(.L__fmth_cos_small)
        mulsd   %xmm1,%xmm1
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_dcos_c4(%rip),%xmm1    /* x2 * c4 */
        addsd   .L__dble_dcos_c3(%rip),%xmm1    /* + c3 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dcos_c2(%rip),%xmm1    /* + c2 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c2 + ...) */
        addsd   .L__dble_dcos_c1(%rip),%xmm1    /* + c1 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c1 + ...) */
        addsd   .L__dble_pq1+16(%rip),%xmm1     /* - 0.5 */
        mulsd   %xmm2,%xmm1                     /* x2 * (0.5 + ...) */
        addsd   %xmm1,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

LBL(.L__fmth_cos_small):
        cmpl    $0x0320,%eax
        jl      LBL(.L__fmth_cos_done1)
        /* return 1.0 - x * x * 0.5 */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1+16(%rip),%xmm1
        addsd   %xmm1,%xmm0

LBL(.L__fmth_cos_done1):
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

        ELF_FUNC(ENT_GH(__fmth_i_cos))
        ELF_SIZE(ENT_GH(__fmth_i_cos))
        IF_GH(ELF_FUNC(__fss_cos))
        IF_GH(ELF_SIZE(__fss_cos))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fsd_cos))
        .globl ENT_GH(__fmth_i_dcos)
IF_GH(ENT(__fsd_cos):)
ENT_GH(__fmth_i_dcos):
        movd    %xmm0, %rax
        mov     $0x03fe921fb54442d18,%rdx
        movapd  .L__dble_sixteen_by_pi(%rip),%xmm4
        andq    .L__real_mask_unsign(%rip), %rax
        cmpq    %rdx,%rax
        jle     LBL(.L__fmth_dcos_shortcuts)
        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     GBLTXT(ENT(__mth_i_dcos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulsd   %xmm0,%xmm4 

        RZ_PUSH
#if defined(_WIN64)
        movdqa  %xmm6, RZ_OFF(24)(%rsp)
        movdqa  %xmm7, RZ_OFF(40)(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5    /* convert to integer */
        movsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        movsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movsd   .L__dble_pi_by_16_us(%rip), %xmm3
        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulsd   %xmm4,%xmm1     /* n * p1 */
        mulsd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        movsdRR   %xmm0,%xmm6     /* x in xmm6 */
        subsd   %xmm1,%xmm0     /* x - n * p1 == rh */
        subsd   %xmm1,%xmm6     /* x - n * p1 == rh == c */

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subsd   %xmm2,%xmm0     /* rh = rh - rt */

        subsd   %xmm0,%xmm6     /* (c - rh) */
        movsdRR   %xmm0,%xmm1     /* Move rh */
        movsdRR   %xmm0,%xmm4     /* Move rh */
        movsdRR   %xmm0,%xmm5     /* Move rh */
        subsd   %xmm2,%xmm6     /* ((c - rh) - rt) */
        subsd   %xmm6,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        movsdRR   %xmm1,%xmm2     /* Move rh */
        subsd   %xmm3,%xmm0     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm4     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm5     /* c = rh - rt aka r */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */
        
        subsd   %xmm0,%xmm1     /* (rh - c) */

        mulsd   %xmm0,%xmm0     /* r^2 in xmm0 */
        movsdRR   %xmm4,%xmm6     /* r in xmm6 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm5,%xmm7     /* r in xmm7 */
        mulsd   %xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulsd   .L__dble_pq4(%rip), %xmm0     /* p4 * r^2 */
        subsd   %xmm6,%xmm2                   /* (rh - c) */
        mulsd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        subsd   %xmm3,%xmm1                   /* (rh - c) - rt aka rr */

        addsd   .L__dble_pq3(%rip), %xmm0     /* + p3 */
        addsd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        subsd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */

        mulsd   %xmm5,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        mulsd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        mulsd   %xmm5,%xmm7                   /* xmm7 = r^3 */
        movsdRR   %xmm1,%xmm3                   /* Move rr */
        mulsd   %xmm5,%xmm1                   /* r * r * rr */

        addsd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        mulsd   .L__dble_pq1+16(%rip), %xmm1   /* r * r * rr * 0.5 */
        mulsd   %xmm6, %xmm3                  /* r * rr */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        mulsd   %xmm5,%xmm0                   /* * r^2 */
        mulsd   %xmm5,%xmm4                   /* * r^2 */
        addsd   %xmm1,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        movlpdMR  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

        addsd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulsd   %xmm7,%xmm0                   /* * r^3 */
        mulsd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addsd   %xmm2,%xmm0                   /* + cs  == dp aka p(r) */
        subsd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */
        movlpdMR  8(%rdx,%rax,8),%xmm3          /* ds2 in xmm3 */
        movlpdMR   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        addsd   %xmm0,%xmm6                   /* + r   == dp aka p(r) */
        movsdRR   %xmm1,%xmm2                   /* dc2 in xmm2 */
        movlpdMR  (%rdx,%rcx,8),%xmm0           /* dc1 */
        mulsd   %xmm4,%xmm1                   /* dc2 * dq */
        mulsd   %xmm6,%xmm3                   /* ds2 * dp */
        addsd   %xmm2,%xmm1                   /* dc2 + dc2*dq */
        mulsd   %xmm5,%xmm6                   /* ds1 * dp */
        subsd   %xmm3,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */

        mulsd   %xmm0,%xmm4                   /* dc1 * dq */
        subsd   %xmm6,%xmm1                   /* (() - ds2*dp) - ds1*dp */
        addsd   %xmm4,%xmm1
        addsd   %xmm1,%xmm0                   /* cos(x) = (C + Cq(r)) + Sq(r) */

#if defined(_WIN64)
        movdqa  RZ_OFF(24)(%rsp),%xmm6
        movdqa  RZ_OFF(40)(%rsp),%xmm7
#endif
        RZ_POP
        ret

LBL(.L__fmth_dcos_shortcuts):
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrq    $48,%rax
        movlpdMR  .L__dble_sincostbl(%rip), %xmm0  /* 1.0 */
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fmth_dcos_small)
        mulsd   %xmm1,%xmm1
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_dcos_c6(%rip),%xmm1    /* x2 * c6 */
        addsd   .L__dble_dcos_c5(%rip),%xmm1    /* + c5 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dcos_c4(%rip),%xmm1    /* + c4 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dcos_c3(%rip),%xmm1    /* + c3 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dcos_c2(%rip),%xmm1    /* + c2 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c2 + ...) */
        addsd   .L__dble_dcos_c1(%rip),%xmm1    /* + c1 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c1 + ...) */
        addsd   .L__dble_pq1+16(%rip),%xmm1     /* - 0.5 */
        mulsd   %xmm2,%xmm1                     /* x2 * (0.5 + ...) */
        addsd   %xmm1,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
        ret

LBL(.L__fmth_dcos_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fmth_dcos_done1)
        /* return 1.0 - x * x * 0.5 */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1+16(%rip),%xmm1
        addsd   %xmm1,%xmm0
        ret

LBL(.L__fmth_dcos_done1):
	rep
        ret

        ELF_FUNC(ENT_GH(__fmth_i_dcos))
        ELF_SIZE(ENT_GH(__fmth_i_dcos))
        IF_GH(ELF_FUNC(__fsd_cos))
        IF_GH(ELF_SIZE(__fsd_cos))


/* ------------------------------------------------------------------------- */
/* 
 *  vector cosine
 * 
 *  An implementation of the cosine libm function.
 * 
 *  Prototype:
 * 
 *      double __fvscos(float *x);
 * 
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fvs_cos))
        .globl ENT_GH(__fvscos)
IF_GH(ENT(__fvs_cos):)
ENT_GH(__fvscos):
	movaps	%xmm0, %xmm1		/* Move input vector */
        andps   .L__sngl_mask_unsign(%rip), %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

        movlps  .L__sngl_pi_over_fours(%rip),%xmm2
        movhps  .L__sngl_pi_over_fours(%rip),%xmm2
        movlps  .L__sngl_needs_argreds(%rip),%xmm3
        movhps  .L__sngl_needs_argreds(%rip),%xmm3
        movlps  .L__sngl_sixteen_by_pi(%rip),%xmm4
        movhps  .L__sngl_sixteen_by_pi(%rip),%xmm4

	cmpps   $5, %xmm0, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	cmpps   $2, %xmm0, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        movmskps %xmm2, %eax
        movmskps %xmm3, %ecx

	test	$15, %eax
        jnz	LBL(.L__Scalar_fvcos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulps   %xmm1,%xmm4 

	test	$15, %ecx
        jnz	LBL(.L__Scalar_fvcos2)

        /* Set n = nearest integer to r */
	movhps	%xmm1,(%rsp)                     /* Store x4, x3 */

	xorq	%r10, %r10
	cvtps2pd %xmm1, %xmm1

        cvtps2dq %xmm4,%xmm5    /* convert to integer, n4,n3,n2,n1 */
LBL(.L__fvcos_do_twice):
#ifdef GH_TARGET
         movddup   .L__dble_pi_by_16_ms(%rip), %xmm0
         movddup   .L__dble_pi_by_16_ls(%rip), %xmm2
         movddup   .L__dble_pi_by_16_us(%rip), %xmm3
#else
        movlpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movhpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movlpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movhpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movlpd   .L__dble_pi_by_16_us(%rip), %xmm3
        movhpd   .L__dble_pi_by_16_us(%rip), %xmm3
#endif

        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - (n*p2 + n*p3) */
        mulpd   %xmm4,%xmm0     /* n * p1 */
        mulpd   %xmm4,%xmm2   /* n * p2 == rt */
        mulpd   %xmm4,%xmm3   /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        subpd   %xmm0,%xmm1   /* x - n * p1 == rh */
	addpd   %xmm2,%xmm3 

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subpd   %xmm3,%xmm1   /* c = rh - rt aka r */

        shrq    $32, %r9
        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

	movapd  %xmm1,%xmm0   /* r in xmm0 and xmm1 */
        movapd  %xmm1,%xmm2   /* r in xmm2 */
        movapd  %xmm1,%xmm4   /* r in xmm4 */
        mulpd   %xmm1,%xmm1   /* r^2 in xmm1 */
        mulpd   %xmm0,%xmm0   /* r^2 in xmm0 */
        mulpd   %xmm4,%xmm4   /* r^2 in xmm4 */
        movapd  %xmm2,%xmm3   /* r in xmm2 and xmm3 */

        leaq    24(%r9),%r8   /* Add 24 for sine */
        andq    $0x1f,%r8     /* And lower 5 bits */
        andq    $0x1f,%r9     /* And lower 5 bits */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulpd   .L__dble_pq3(%rip), %xmm0     /* p3 * r^2 */
        mulpd   .L__dble_pq3+16(%rip), %xmm1  /* q3 * r^2 */

        movq    %r8, %rdx     /* Duplicate it */
        sarq    $4,%r8        /* Sign bits moved down */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r8     /* Final tbl address */

        addpd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addpd   .L__dble_pq2+16(%rip), %xmm1  /* + q2 */

        movq    %r9, %rdx     /* Duplicate it */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r9     /* Final tbl address */

        mulpd   %xmm4,%xmm0                   /* * r^2 */
        mulpd   %xmm4,%xmm1                   /* * r^2 */

        mulpd   %xmm4,%xmm3                   /* xmm3 = r^3 */
        addpd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addpd   .L__dble_pq1+16(%rip), %xmm1  /* + q1 */

        addq    %rax,%rax
        addq    %r8,%r8
        addq    %rcx,%rcx
        addq    %r9,%r9

	mulpd   %xmm3,%xmm0                   /* * r^3 */
	mulpd   %xmm4,%xmm1                   /* * r^2  = q(r) */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        movlpdMR  (%rdx,%rcx,8),%xmm3           /* C in xmm3 */
        movhpd  (%rdx,%r9,8),%xmm3            /* C in xmm3 */

        movlpdMR  (%rdx,%rax,8),%xmm4           /* S in xmm4 */
        movhpd  (%rdx,%r8,8),%xmm4            /* S in xmm4 */

	addpd   %xmm2,%xmm0                   /* + r = p(r) */

	mulpd   %xmm3, %xmm1                  /* C * q(r) */
	mulpd   %xmm4, %xmm0                  /* S * p(r) */
	addpd   %xmm3, %xmm1                  /* C + C * q(r) */
	subpd   %xmm0, %xmm1                  /* cos(x) = (C+Cq(r)) - Sp(r) */

	cvtpd2ps %xmm1,%xmm0
	cmp	$0, %r10                      /* Compare loop count */
	shufps	$78, %xmm0, %xmm5             /* sin(x2), sin(x1), n4, n3 */
	jne 	LBL(.L__fvcos_done_twice)
	inc 	%r10
	cvtps2pd (%rsp),%xmm1
	jmp 	LBL(.L__fvcos_do_twice)

LBL(.L__fvcos_done_twice):
	movaps  %xmm5, %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvcos1):
        /* Here when at least one argument is less than pi/4,
           or, at least one is a Nan.  What we will do for now, is
           if all are less than pi/4, do them all.  Otherwise, call
           fmth_i_cos or mth_i_cos one at a time.
        */
        movaps  %xmm0, (%rsp)                 /* Save xmm0, masked x */
	cmpps   $3, %xmm0, %xmm0              /* 3 is "unordered" */
        movaps  %xmm1, 16(%rsp)               /* Save xmm1, input x */
        movmskps %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

	cmp	$15, %eax
	jne	LBL(.L__Scalar_fvcos1a)

	cvtps2pd 16(%rsp),%xmm0               /* x(2), x(1) */
	cvtps2pd 24(%rsp),%xmm1               /* x(4), x(3) */

        movapd  %xmm0,16(%rsp)
        movapd  %xmm1,32(%rsp)
	mulpd   %xmm0,%xmm0                   /* x2 for x(2), x(1) */
	mulpd   %xmm1,%xmm1                   /* x2 for x(4), x(3) */

#ifdef GH_TARGET
         movddup  .L__dble_dcos_c4(%rip),%xmm4  /* c4 */
         movddup  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */
#else
        movlpd  .L__dble_dcos_c4(%rip),%xmm4  /* c4 */
        movhpd  .L__dble_dcos_c4(%rip),%xmm4  /* c4 */
        movlpd  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */
        movhpd  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */
#endif

        movapd  %xmm0,%xmm2
        movapd  %xmm1,%xmm3
        mulpd   %xmm4,%xmm0                   /* x2 * c4 */
        mulpd   %xmm4,%xmm1                   /* x2 * c4 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c2(%rip),%xmm4  /* c2 */
#else
        movlpd  .L__dble_dcos_c2(%rip),%xmm4  /* c2 */
        movhpd  .L__dble_dcos_c2(%rip),%xmm4  /* c2 */
#endif

        addpd   %xmm5,%xmm0                   /* + c3 */
        addpd   %xmm5,%xmm1                   /* + c3 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */
#else
        movlpd  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */
        movhpd  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */
#endif

        mulpd   %xmm2,%xmm0                   /* x2 * (c3 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c3 + ...) */

        addpd   %xmm4,%xmm0                   /* + c2 */
        addpd   %xmm4,%xmm1                   /* + c2 */
        movapd  .L__dble_pq1+16(%rip),%xmm4   /* -0.5 */
        mulpd   %xmm2,%xmm0                   /* x2 * (c2 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c2 + ...) */
        addpd   %xmm5,%xmm0                   /* + c1 */
        addpd   %xmm5,%xmm1                   /* + c1 */
	movapd  .L__real_one(%rip), %xmm5     /* 1.0 */
        mulpd   %xmm2,%xmm0                   /* x2 * (c1 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c1 + ...) */
        addpd   %xmm4,%xmm0                   /* -0.5 */
        addpd   %xmm4,%xmm1                   /* -0.5 */
        mulpd   %xmm2,%xmm0                   /* - x2 * (0.5 + ...) */
        mulpd   %xmm3,%xmm1                   /* - x2 * (0.5 + ...) */
	addpd   %xmm5,%xmm0                   /* 1.0 - 0.5x2 + (...) done */
	addpd   %xmm5,%xmm1                   /* 1.0 - 0.5x2 + (...) done */
        cvtpd2ps %xmm0,%xmm0            /* cos(x2), cos(x1) */
        cvtpd2ps %xmm1,%xmm1            /* cos(x4), cos(x3) */
	shufps	$68, %xmm1, %xmm0       /* cos(x4),cos(x3),cos(x2),cos(x1) */

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvcos1a):
	test    $1, %eax
	jz	LBL(.L__Scalar_fvcos3)
#ifdef GH_TARGET
	movss 16(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 16(%rsp),%xmm0               /* dble x(1) */
#endif
	movl	(%rsp),%edx
	call	LBL(.L__fmth_fvcos_local)
	jmp	LBL(.L__Scalar_fvcos5)

LBL(.L__Scalar_fvcos2):
        movaps  %xmm0, (%rsp)                 /* Save xmm0 */
        movaps  %xmm1, %xmm0                  /* Save xmm1 */
        movaps  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvcos3):
	movss   16(%rsp),%xmm0                /* x(1) */
	test    $1, %ecx
	jz	LBL(.L__Scalar_fvcos4)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_cos))                /* Here when big or a nan */
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvcos5)

LBL(.L__Scalar_fvcos4):
	mov     %eax, 32(%rsp)                /* Here when a scalar will do */
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fmth_i_cos_gh))
#else
	CALL(ENT(__fmth_i_cos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvcos5):
        movss   %xmm0, (%rsp)                 /* Move first result */

	test    $2, %eax
	jz	LBL(.L__Scalar_fvcos6)
#ifdef GH_TARGET
	movss 20(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 20(%rsp),%xmm0               /* dble x(2) */
#endif
	movl	4(%rsp),%edx
	call	LBL(.L__fmth_fvcos_local)
	jmp	LBL(.L__Scalar_fvcos8)

LBL(.L__Scalar_fvcos6):
	movss   20(%rsp),%xmm0                /* x(2) */
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvcos7)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_cos))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvcos8)

LBL(.L__Scalar_fvcos7):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fmth_i_cos_gh))
#else
	CALL(ENT(__fmth_i_cos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvcos8):
        movss   %xmm0, 4(%rsp)               /* Move 2nd result */

	test    $4, %eax
	jz	LBL(.L__Scalar_fvcos9)
#ifdef GH_TARGET
	movss 24(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 24(%rsp),%xmm0               /* dble x(3) */
#endif
	movl	8(%rsp),%edx
	call	LBL(.L__fmth_fvcos_local)
	jmp	LBL(.L__Scalar_fvcos11)

LBL(.L__Scalar_fvcos9):
	movss   24(%rsp),%xmm0                /* x(3) */
	test    $4, %ecx
	jz	LBL(.L__Scalar_fvcos10)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_cos))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvcos11)

LBL(.L__Scalar_fvcos10):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fmth_i_cos_gh))
#else
	CALL(ENT(__fmth_i_cos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvcos11):
        movss   %xmm0, 8(%rsp)               /* Move 3rd result */

	test    $8, %eax
	jz	LBL(.L__Scalar_fvcos12)
#ifdef GH_TARGET
	movss 28(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 28(%rsp),%xmm0               /* dble x(4) */
#endif
	movl	12(%rsp),%edx
	call	LBL(.L__fmth_fvcos_local)
	jmp	LBL(.L__Scalar_fvcos14)

LBL(.L__Scalar_fvcos12):
	movss   28(%rsp),%xmm0                /* x(4) */
	test    $8, %ecx
	jz	LBL(.L__Scalar_fvcos13)
	CALL(ENT(__mth_i_cos))
	jmp	LBL(.L__Scalar_fvcos14)

LBL(.L__Scalar_fvcos13):
#ifdef GH_TARGET
	CALL(ENT(__fmth_i_cos_gh))
#else
	CALL(ENT(__fmth_i_cos))
#endif

/* ---------------------------------- */
LBL(.L__Scalar_fvcos14):
        movss   %xmm0, 12(%rsp)               /* Move 4th result */
	movaps	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fmth_fvcos_local):
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrl    $20,%edx
	movlpdMR  .L__dble_sincostbl(%rip), %xmm0 /* 1.0 */
        cmpl    $0x0390,%edx
        jl      LBL(.L__fmth_fvcos_small)
        mulsd   %xmm1,%xmm1
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_dcos_c4(%rip),%xmm1    /* x2 * c4 */
        addsd   .L__dble_dcos_c3(%rip),%xmm1    /* + c3 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dcos_c2(%rip),%xmm1    /* + c2 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c2 + ...) */
        addsd   .L__dble_dcos_c1(%rip),%xmm1    /* + c1 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c1 + ...) */
	addsd	.L__dble_pq1+16(%rip),%xmm1     /* -0.5 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c1 + ...) */
        addsd   %xmm1,%xmm0                     /* x + x3 * (...) done */
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

LBL(.L__fmth_fvcos_small):
        cmpl    $0x0320,%edx
        jl      LBL(.L__fmth_fvcos_done1)
        /* return 1.0 - x * x * 0.5 */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1+16(%rip),%xmm1
        addsd   %xmm1,%xmm0

LBL(.L__fmth_fvcos_done1):
#ifdef GH_TARGET
	unpcklpd %xmm0, %xmm0
	cvtpd2ps %xmm0, %xmm0
#else
	cvtsd2ss %xmm0,%xmm0
#endif
        ret

        ELF_FUNC(ENT_GH(__fvscos))
        ELF_SIZE(ENT_GH(__fvscos))
        IF_GH(ELF_FUNC(__fvs_cos))
        IF_GH(ELF_SIZE(__fvs_cos))


/* ------------------------------------------------------------------------- */
/* 
 *  vector cosine
 * 
 *  An implementation of the cosine libm function.
 * 
 *  Prototype:
 * 
 *      double __fvdcos(double *x);
 * 
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 * 
 */
        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fvd_cos))
        .globl ENT_GH(__fvdcos)
IF_GH(ENT(__fvd_cos):)
ENT_GH(__fvdcos):
	movapd	%xmm0, %xmm1		/* Move input vector */
        andpd   .L__real_mask_unsign(%rip), %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

#ifdef GH_TARGET
         movddup  .L__dble_pi_over_fours(%rip),%xmm2
         movddup  .L__dble_needs_argreds(%rip),%xmm3
         movddup  .L__dble_sixteen_by_pi(%rip),%xmm4
#else
        movlpd  .L__dble_pi_over_fours(%rip),%xmm2
        movhpd  .L__dble_pi_over_fours(%rip),%xmm2
        movlpd  .L__dble_needs_argreds(%rip),%xmm3
        movhpd  .L__dble_needs_argreds(%rip),%xmm3
        movlpd  .L__dble_sixteen_by_pi(%rip),%xmm4
        movhpd  .L__dble_sixteen_by_pi(%rip),%xmm4
#endif

	cmppd   $5, %xmm0, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	cmppd   $2, %xmm0, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        movmskpd %xmm2, %eax
        movmskpd %xmm3, %ecx

	test	$3, %eax
        jnz	LBL(.L__Scalar_fvdcos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulpd   %xmm1,%xmm4 

	test	$3, %ecx
        jnz	LBL(.L__Scalar_fvdcos2)

#if defined(_WIN64)
        movdqa  %xmm6, (%rsp)
        movdqa  %xmm7, 16(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5    /* convert to integer */
#ifdef GH_TARGET
         movddup   .L__dble_pi_by_16_ms(%rip), %xmm0
         movddup   .L__dble_pi_by_16_ls(%rip), %xmm2
         movddup   .L__dble_pi_by_16_us(%rip), %xmm3
#else
        movlpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movhpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movlpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movhpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movlpd   .L__dble_pi_by_16_us(%rip), %xmm3
        movhpd   .L__dble_pi_by_16_us(%rip), %xmm3
#endif

        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulpd   %xmm4,%xmm0     /* n * p1 */
        mulpd   %xmm4,%xmm2   /* n * p2 == rt */
        mulpd   %xmm4,%xmm3   /* n * p3 */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */

        /* How to convert N into a table address */
        movapd  %xmm1,%xmm6   /* x in xmm6 */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        subpd   %xmm0,%xmm1   /* x - n * p1 == rh */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        subpd   %xmm0,%xmm6   /* x - n * p1 == rh == c */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        subpd   %xmm2,%xmm1   /* rh = rh - rt */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */
        subpd   %xmm1,%xmm6   /* (c - rh) */
        movq    %rax, %rdx    /* Duplicate it */
        movapd  %xmm1,%xmm0   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        movapd  %xmm1,%xmm4   /* Move rh */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        movapd  %xmm1,%xmm5   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        subpd   %xmm2,%xmm6   /* ((c - rh) - rt) */
        andq    $0xf,%rdx     /* And lower 5 bits */
        subpd   %xmm6,%xmm3   /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        addq    %rdx, %rax    /* Final tbl address */
        movapd  %xmm1,%xmm2   /* Move rh */
        shrq    $32, %r9
        subpd   %xmm3,%xmm0   /* c = rh - rt aka r */
        movq    %rcx, %rdx    /* Duplicate it */
        subpd   %xmm3,%xmm4   /* c = rh - rt aka r */
        sarq    $4,%rcx       /* Sign bits moved down */
        subpd   %xmm3,%xmm5   /* c = rh - rt aka r */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        subpd   %xmm0,%xmm1   /* (rh - c) */
        sarq    $4,%rcx       /* Sign bits moved down */
        mulpd   %xmm0,%xmm0   /* r^2 in xmm0 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        movapd  %xmm4,%xmm6   /* r in xmm6 */
        addq    %rdx, %rcx    /* Final tbl address */
        mulpd   %xmm4,%xmm4   /* r^2 in xmm4 */
        leaq    24(%r9),%r8   /* Add 24 for sine */
        movapd  %xmm5,%xmm7   /* r in xmm7 */
        andq    $0x1f,%r8     /* And lower 5 bits */
        mulpd   %xmm5,%xmm5   /* r^2 in xmm5 */
        andq    $0x1f,%r9     /* And lower 5 bits */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulpd   .L__dble_pq4(%rip), %xmm0     /* p4 * r^2 */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        subpd   %xmm6,%xmm2                   /* (rh - c) */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        mulpd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        subpd   %xmm3,%xmm1                   /* (rh - c) - rt aka rr */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */
        addpd   .L__dble_pq3(%rip), %xmm0     /* + p3 */
        movq    %r8, %rdx     /* Duplicate it */
        addpd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        sarq    $4,%r8        /* Sign bits moved down */
        subpd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        mulpd   %xmm5,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        sarq    $4,%r8        /* Sign bits moved down */
        mulpd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        mulpd   %xmm5,%xmm7                   /* xmm7 = r^3 */
        addq    %rdx, %r8     /* Final tbl address */
        movapd  %xmm1,%xmm3                   /* Move rr */
        movq    %r9, %rdx     /* Duplicate it */
        mulpd   %xmm5,%xmm1                   /* r * r * rr */
        sarq    $4,%r9        /* Sign bits moved down */
        addpd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        addpd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        sarq    $4,%r9        /* Sign bits moved down */
        mulpd   .L__dble_pq1+16(%rip), %xmm1  /* r * r * rr * 0.5 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        mulpd   %xmm6, %xmm3                  /* r * rr */
        addq    %rdx, %r9     /* Final tbl address */
        mulpd   %xmm5,%xmm0                   /* * r^2 */
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        mulpd   %xmm5,%xmm4                   /* * r^2 */
        addq    %rcx,%rcx
        addq    %r9,%r9
        addpd   %xmm1,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        addq    %rax,%rax
        addq    %r8,%r8
        movlpdMR  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        movhpd  8(%rdx,%r9,8),%xmm1           /* dc2 in xmm1 */


        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

        addpd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addpd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulpd   %xmm7,%xmm0                   /* * r^3 */
        mulpd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addpd   %xmm2,%xmm0                   /* + cs  == dp aka p(r) */
        subpd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */
        movlpdMR  8(%rdx,%rax,8),%xmm3          /* ds2 in xmm3 */
        movhpd  8(%rdx,%r8,8),%xmm3           /* ds2 in xmm3 */

        movlpdMR   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        movhpd   (%rdx,%r8,8),%xmm5           /* ds1 in xmm5 */

        addpd   %xmm0,%xmm6                   /* + r   == dp aka p(r) */
        movapd  %xmm1,%xmm2                   /* ds2 in xmm2 */

        movlpdMR  (%rdx,%rcx,8),%xmm0           /* dc1 in xmm6 */
        movhpd  (%rdx,%r9,8),%xmm0            /* dc1 in xmm6 */


        mulpd   %xmm4,%xmm1                   /* dc2 * dq */
        mulpd   %xmm6,%xmm3                   /* ds2 * dp */
        addpd   %xmm2,%xmm1                   /* dc2 + dc2*dq */
        mulpd   %xmm5,%xmm6                   /* ds1 * dp */
        subpd   %xmm3,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */
        mulpd   %xmm0,%xmm4                   /* dc1 * dq */
        subpd   %xmm6,%xmm1                   /* ((dc2...) - ds2*dp) - ds1*dp */

#if defined(_WIN64)
        movdqa  (%rsp),%xmm6
        movdqa  16(%rsp),%xmm7
#endif
        addpd   %xmm4,%xmm1
        addpd   %xmm1,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdcos1):
        movapd  %xmm0, (%rsp)                 /* Save xmm0 */
	cmppd   $3, %xmm0, %xmm0              /* 3 is "unordered" */
        movapd  %xmm1, 16(%rsp)               /* Save xmm1 */
        movmskpd %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

        movapd  16(%rsp), %xmm0
	test    $1, %eax
	jz	LBL(.L__Scalar_fvdcos3)
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdcos1a)

        movapd  %xmm0,%xmm1
        movapd  %xmm0,%xmm2
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c6(%rip),%xmm3    /* c6 */
#else
        movlpd  .L__dble_dcos_c6(%rip),%xmm3    /* c6 */
        movhpd  .L__dble_dcos_c6(%rip),%xmm3    /* c6 */
#endif

        mulpd   %xmm1,%xmm1
        mulpd   %xmm2,%xmm2
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c5(%rip),%xmm4    /* c5 */
#else
        movlpd  .L__dble_dcos_c5(%rip),%xmm4    /* c5 */
        movhpd  .L__dble_dcos_c5(%rip),%xmm4    /* c5 */
#endif

        mulpd   %xmm3,%xmm1                     /* x2 * c6 */
	movapd  .L__real_one(%rip), %xmm0       /* 1.0 */
        addpd   %xmm4,%xmm1                     /* + c5 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c4(%rip),%xmm3    /* c4 */
#else
        movlpd  .L__dble_dcos_c4(%rip),%xmm3    /* c4 */
        movhpd  .L__dble_dcos_c4(%rip),%xmm3    /* c4 */
#endif

        mulpd   %xmm2,%xmm1                     /* x2 * (c5 + ...) */
        addpd   %xmm3,%xmm1                     /* + c4 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c3(%rip),%xmm4    /* c3 */
#else
        movlpd  .L__dble_dcos_c3(%rip),%xmm4    /* c3 */
        movhpd  .L__dble_dcos_c3(%rip),%xmm4    /* c3 */
#endif

        mulpd   %xmm2,%xmm1                     /* x2 * (c4 + ...) */
        addpd   %xmm4,%xmm1                     /* + c3 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c2(%rip),%xmm3    /* c2 */
#else
        movlpd  .L__dble_dcos_c2(%rip),%xmm3    /* c2 */
        movhpd  .L__dble_dcos_c2(%rip),%xmm3    /* c2 */
#endif

        mulpd   %xmm2,%xmm1                     /* x2 * (c3 + ...) */
        addpd   %xmm3,%xmm1                     /* + c2 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c1(%rip),%xmm4    /* c1 */
#else
        movlpd  .L__dble_dcos_c1(%rip),%xmm4    /* c1 */
        movhpd  .L__dble_dcos_c1(%rip),%xmm4    /* c1 */
#endif

        mulpd   %xmm2,%xmm1                     /* x2 * (c2 + ...) */
        addpd   %xmm4,%xmm1                     /* + c1 */
        mulpd   %xmm2,%xmm1                     /* x2 */
        addpd   .L__dble_pq1+16(%rip),%xmm1     /* - 0.5 */
        mulpd   %xmm2,%xmm1                     /* x2 * (c1 + ...) */
        addpd   %xmm1,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdcos1a):
	movq	(%rsp),%rdx
	call	LBL(.L__fmth_fvdcos_local)
	jmp	LBL(.L__Scalar_fvdcos5)

LBL(.L__Scalar_fvdcos2):
        movapd  %xmm0, (%rsp)                 /* Save xmm0 */
        movapd  %xmm1, %xmm0                  /* Save xmm1 */
        movapd  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvdcos3):
	test    $1, %ecx
	jz	LBL(.L__Scalar_fvdcos4)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_dcos))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvdcos5)

LBL(.L__Scalar_fvdcos4):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fsd_cos))
#else
	CALL(ENT(__fmth_i_dcos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

LBL(.L__Scalar_fvdcos5):
        movlpd  %xmm0, (%rsp)
        movlpdMR  24(%rsp), %xmm0
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdcos6)
	movq	8(%rsp),%rdx
	call	LBL(.L__fmth_fvdcos_local)
	jmp	LBL(.L__Scalar_fvdcos8)

LBL(.L__Scalar_fvdcos6):
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvdcos7)
	CALL(ENT(__mth_i_dcos))
	jmp	LBL(.L__Scalar_fvdcos8)

LBL(.L__Scalar_fvdcos7):
#ifdef GH_TARGET
	CALL(ENT(__fsd_cos))
#else
	CALL(ENT(__fmth_i_dcos))
#endif

LBL(.L__Scalar_fvdcos8):
        movlpd  %xmm0, 8(%rsp)
	movapd	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fmth_fvdcos_local):
        movsdRR   %xmm0,%xmm1
        movsdRR   %xmm0,%xmm2
        shrq    $48,%rdx
	movlpdMR  .L__dble_sincostbl(%rip), %xmm0 /* 1.0 */
        cmpl    $0x03f20,%edx
        jl      LBL(.L__fmth_fvdcos_small)
        mulsd   %xmm1,%xmm1
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_dcos_c6(%rip),%xmm1    /* x2 * c6 */
        addsd   .L__dble_dcos_c5(%rip),%xmm1    /* + c5 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dcos_c4(%rip),%xmm1    /* + c4 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dcos_c3(%rip),%xmm1    /* + c3 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dcos_c2(%rip),%xmm1    /* + c2 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c2 + ...) */
        addsd   .L__dble_dcos_c1(%rip),%xmm1    /* + c1 */
        mulsd   %xmm2,%xmm1                     /* x2 * (c1 + ...) */
        addsd   .L__dble_pq1+16(%rip),%xmm1     /* - 0.5 */
        mulsd   %xmm2,%xmm1                     /* x2 * (0.5 + ...) */
        addsd   %xmm1,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
        ret

LBL(.L__fmth_fvdcos_small):
        cmpl    $0x03e40,%edx
        jl      LBL(.L__fmth_fvdcos_done1)
        /* return 1.0 - x * x * 0.5 */
        mulsd   %xmm1,%xmm1
        mulsd   .L__dble_pq1+16(%rip),%xmm1
        addsd   %xmm1,%xmm0
        ret

LBL(.L__fmth_fvdcos_done1):
	rep
        ret

        ELF_FUNC(ENT_GH(__fvdcos))
        ELF_SIZE(ENT_GH(__fvdcos))
        IF_GH(ELF_FUNC(__fvd_cos))
        IF_GH(ELF_SIZE(__fvd_cos))


/* ============================================================
 *  fastsinh.s
 * 
 *  An implementation of the sinh libm function.
 * 
 *  Prototype:
 * 
 *      float fastsinh(float x);
 * 
 *    Computes the hyperbolic sine.
 * 
 */
	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fss_sinh))
	.globl ENT_GH(__fmth_i_sinh)
IF_GH(ENT(__fss_sinh):)
ENT_GH(__fmth_i_sinh):
	RZ_PUSH

        movd    %xmm0, %eax
        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
#ifdef GH_TARGET
	unpcklps %xmm0, %xmm0
	cvtps2pd %xmm0, %xmm2
#else
	cvtss2sd %xmm0, %xmm2
#endif

	shrl	$23, %eax
	andl	$0xff, %eax
	cmpl	$122, %eax
	jb	LBL(.L__fss_sinh_shortcuts)

	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm2,%xmm3 

	/* Set n = nearest integer to r */
	comiss	.L_sp_sinh_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_inf)
	comiss	.L_sp_sinh_min_singleval(%rip), %xmm0
	jb	LBL(.L_sp_sinh_ninf)

	cvtpd2dq %xmm3,%xmm4	/* convert to integer */
	cvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
	mulsd	.L__real_log2_by_32(%rip),%xmm1
	movd	%xmm4,%ecx
	subsd	%xmm1,%xmm2	/* r1 in xmm2, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* j = n & 0x0000001f; */
	movq	%r8,%rax
	and	%ecx,%eax
	subl	%ecx,%r9d
	and	%r9d,%r8d

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%eax,%ecx
	sar	$5,%ecx

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm1
	movlpdMR	.L__real_3fe0000000000000(%rip),%xmm5
	sub	%r8d,%r9d
	sar	$5,%r9d

	movsdRR	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	mulsd	%xmm2,%xmm2
	subsd	%xmm1,%xmm5                             /* exp(-x) */
	addsd	.L__real_3fe0000000000000(%rip),%xmm1   /* exp(x) */
	mulsd	%xmm2,%xmm5        /* exp(-x) */
	mulsd	%xmm1,%xmm2        /* exp(x) */

	movlpdMR	(%rdx,%r8,8),%xmm3   /* exp(-x) */
	movlpdMR	(%rdx,%rax,8),%xmm4   /* exp(x) */
	subsd	%xmm0,%xmm5        /* exp(-x) */
	addsd	%xmm0,%xmm2        /* exp(x) */

	/* *z2 = f2 + ((f1 + f2) * q); */
        add	$1022, %ecx	/* add bias */
        add	$1022, %r9d	/* add bias */

	mulsd	%xmm3,%xmm5
	mulsd	%xmm4,%xmm2
        shlq	$52,%rcx        /* build 2^n */
        shlq	$52,%r9         /* build 2^n */
	addsd	%xmm3,%xmm5  /* z = z1 + z2   done with 1,2,3,4,5 */
	addsd	%xmm4,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(16)(%rsp),%xmm5	/* result *= 2^n */
	mulsd	RZ_OFF(24)(%rsp),%xmm2	/* result *= 2^n */
	subsd	%xmm5,%xmm2             /* result = exp(x) - exp(-x) */

LBL(.L__fss_sinh_done):
#ifdef GH_TARGET
	unpcklpd %xmm2, %xmm2
	cvtpd2ps %xmm2, %xmm0
#else
	cvtsd2ss %xmm2,%xmm0
#endif

	RZ_POP
	rep
	ret

LBL(.L__fss_sinh_shortcuts):
	movapd	%xmm2, %xmm1
	movapd	%xmm2, %xmm0
	mulsd	%xmm2, %xmm2
	mulsd 	%xmm1, %xmm1
	mulsd	.L__dsinh_shortval_y4(%rip), %xmm2
	addsd	.L__dsinh_shortval_y3(%rip), %xmm2
	mulsd	%xmm1, %xmm2
	addsd	.L__dsinh_shortval_y2(%rip), %xmm2
	mulsd	%xmm1, %xmm2
	mulsd	%xmm0, %xmm1
	addsd	.L__dsinh_shortval_y1(%rip), %xmm2
	mulsd	%xmm1, %xmm2
	addsd	%xmm0, %xmm2
	jmp	LBL(.L__fss_sinh_done)

        ELF_FUNC(ENT_GH(__fmth_i_sinh))
        ELF_SIZE(ENT_GH(__fmth_i_sinh))
	IF_GH(ELF_FUNC(__fss_sinh))
	IF_GH(ELF_SIZE(__fss_sinh))


/* ============================================================
 *  fastsinh.s
 *
 *  An implementation of the dsinh libm function.
 * 
 *  Prototype:
 * 
 *      double fastsinh(double x);
 * 
 *    Computes the hyperbolic sine.
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fsd_sinh))
	.globl ENT_GH(__fmth_i_dsinh)
IF_GH(ENT(__fsd_sinh):)
ENT_GH(__fmth_i_dsinh):
	RZ_PUSH
#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(40)(%rsp)
#endif

	movd 	%xmm0, %rax
        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm0,%xmm3 
	andq	.L__real_mask_unsign(%rip), %rax
	shrq	$47, %rax

	/* Set n = nearest integer to r */
	comisd	.L_sinh_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	comisd	.L_sinh_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_sinh_ninf)
	cmpq	$0x7fd4, %rax
	jbe	LBL(.L__fsd_sinh_shortcuts)

	cvtpd2dq %xmm3,%xmm4	/* convert to integer */
	cvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
	movlpdMR	.L__real_log2_by_32_lead(%rip),%xmm2
	mulsd	%xmm1,%xmm2
	movd	%xmm4,%ecx
	subsd	%xmm2,%xmm0	/* r1 in xmm0, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
	mulsd	.L__real_log2_by_32_tail(%rip),%xmm1

	/* j = n & 0x0000001f; */
	movq	%r8,%rax
	andl	%ecx,%eax
	subl	%ecx,%r9d
	andl	%r9d,%r8d

	movsdRR	%xmm0,%xmm2

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%eax,%ecx
	sarl	$5,%ecx
	addsd	%xmm1,%xmm2    /* r = r1 + r2 */
	subl	%r8d,%r9d
	sarl	$5,%r9d

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */

	movsdRR	%xmm2,%xmm1
	movlpdMR	.L__real_3f56c1728d739765(%rip),%xmm3
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm0
	movlpdMR	.L__real_3F811115B7AA905E(%rip),%xmm5
	movlpdMR	.L__real_3fe0000000000000(%rip),%xmm6
	mulsd	%xmm2,%xmm3
	mulsd	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	movsdRR	%xmm1,%xmm4
	subsd	%xmm3,%xmm5
	addsd	.L__real_3F811115B7AA905E(%rip),%xmm3
	subsd	%xmm0,%xmm6
	addsd	.L__real_3fe0000000000000(%rip),%xmm0
	mulsd	%xmm1,%xmm4
	mulsd	%xmm2,%xmm5
	mulsd	%xmm2,%xmm3
	mulsd	%xmm1,%xmm6
	mulsd	%xmm1,%xmm0
	subsd	.L__real_3FA5555555545D4E(%rip),%xmm5
	addsd	.L__real_3FA5555555545D4E(%rip),%xmm3
	subsd	%xmm2,%xmm6
	addsd	%xmm2,%xmm0
	mulsd	%xmm4,%xmm5
	mulsd	%xmm4,%xmm3
	subsd	%xmm5,%xmm6
	addsd	%xmm3,%xmm0

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%rdx,%r8,8),%xmm4   /* exp(-x) */
	movlpdMR	(%rdx,%rax,8),%xmm5   /* exp(x) */
	/* deal with infinite results */
        movslq	%ecx,%rcx
	mulsd	%xmm5,%xmm0
	addsd	%xmm5,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* deal with denormal results */
	movq	$1, %rdx
	movq	$1, %rax
        addq	$1022, %rcx	/* add bias */
	cmovleq	%rcx, %rdx
	cmovleq	%rax, %rcx
        shlq	$52,%rcx        /* build 2^n */
        addq	$1022, %rdx	/* add bias */
        shlq	$52,%rdx        /* build 2^n */
	movq	%rdx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

	/* deal with infinite results */
        movslq	%r9d,%rcx
	mulsd	%xmm4,%xmm6
	addsd	%xmm4,%xmm6  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* deal with denormal results */
	movq	$1, %rdx
	movq	$1, %rax
        addq	$1022, %rcx	/* add bias */
	cmovleq	%rcx, %rdx
	cmovleq	%rax, %rcx
        shlq	$52,%rcx        /* build 2^n */
        addq	$1022, %rdx	/* add bias */
        shlq	$52,%rdx        /* build 2^n */
	movq	%rdx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm6	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm6	/* result *= 2^n */
	subsd	%xmm6, %xmm0

LBL(.L__fsd_sinh_done):

#if defined(_WIN64)
	movdqa	RZ_OFF(40)(%rsp), %xmm6
#endif
	RZ_POP
	rep
	ret

LBL(.L__fsd_sinh_shortcuts):
	/*   x2 = x*x
	     x3 = x2*x
  	     y1 = 1.0d0/(3.0d0*2.0d0)
  	     y2 = 1.0d0/(5.0d0*4.0d0) * y1
  	     y3 = 1.0d0/(7.0d0*6.0d0) * y2
  	     y4 = 1.0d0/(9.0d0*8.0d0) * y3
  	     dsinh = (((y4*x2 + y3)*x2 + y2)*x2 + y1)*x3 + x
	*/
	movapd	%xmm0, %xmm1
	movapd	%xmm0, %xmm2
	mulsd	%xmm0, %xmm0
	mulsd 	%xmm1, %xmm1
	mulsd	.L__dsinh_shortval_y7(%rip), %xmm0
	addsd	.L__dsinh_shortval_y6(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	.L__dsinh_shortval_y5(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	.L__dsinh_shortval_y4(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	.L__dsinh_shortval_y3(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	.L__dsinh_shortval_y2(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	mulsd	%xmm2, %xmm1
	addsd	.L__dsinh_shortval_y1(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	%xmm2, %xmm0
	jmp	LBL(.L__fsd_sinh_done)


        ELF_FUNC(ENT_GH(__fmth_i_dsinh))
        ELF_SIZE(ENT_GH(__fmth_i_dsinh))
	IF_GH(ELF_FUNC(__fsd_sinh))
	IF_GH(ELF_SIZE(__fsd_sinh))

/* ============================================================
 *  vector fastsinhf.s
 * 
 *  An implementation of the sinh libm function.
 * 
 *  Prototype:
 * 
 *      float fastsinhf(float x);
 * 
 *    Computes hyperbolic sine of x
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fvs_sinh))
	.globl ENT_GH(__fvssinh)
IF_GH(ENT(__fvs_sinh):)
ENT_GH(__fvssinh):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
	movdqa	%xmm7, RZ_OFF(88)(%rsp)  /* SINH needs xmm7 */
#endif

	/* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movhlps  %xmm0, %xmm1
	movaps	 %xmm0, %xmm5
	movaps	.L__ps_vssinh_too_small(%rip), %xmm3
	andps	 .L__ps_mask_unsign(%rip), %xmm5
	cmpps	$5, %xmm5, %xmm3
	cmpps	$6, .L__sp_ln_max_singleval(%rip), %xmm5
	movmskps %xmm3, %r9d
	test	 $15, %r9d
	jnz	LBL(.L__Scalar_fvssinh)
	movmskps %xmm5, %r8d
	test	 $15, %r8d
	jnz	LBL(.L__Scalar_fvssinh)

	cvtps2pd %xmm0, %xmm2		/* xmm2 = dble(a(2)), dble(a(1)) */
	cvtps2pd %xmm1, %xmm1		/* xmm1 = dble(a(4)), dble(a(3)) */
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm4
	mulpd	%xmm2, %xmm3 
	mulpd	%xmm1, %xmm4 

	/* Set n = nearest integer to r */
	cvtpd2dq %xmm3,%xmm5	/* convert to integer */
	cvtpd2dq %xmm4,%xmm6	/* convert to integer */
	cvtdq2pd %xmm5,%xmm3	/* and back to float. */
	cvtdq2pd %xmm6,%xmm4	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
	mulpd	.L__real_log2_by_32(%rip),%xmm3
	mulpd	.L__real_log2_by_32(%rip),%xmm4
	movq	%xmm5,RZ_OFF(96)(%rsp)
	movq	%xmm6,RZ_OFF(104)(%rsp)
	subpd	%xmm3,%xmm2	/* r1 in xmm2, */
	subpd	%xmm4,%xmm1	/* r1 in xmm1, */
	leaq	.L__two_to_jby32_table(%rip),%rax

	/* j = n & 0x0000001f; */
	mov	RZ_OFF(92)(%rsp),%r8d
	mov	RZ_OFF(96)(%rsp),%r9d
	mov	RZ_OFF(100)(%rsp),%r10d
	mov	RZ_OFF(104)(%rsp),%r11d
	movq	$0x1f, %rcx
	and 	%r8d, %ecx
	movq	$0x1f, %rdx
	and 	%r9d, %edx
	movapd	%xmm2,%xmm0
	movapd	%xmm1,%xmm3
	movapd	%xmm2,%xmm4
	movapd	%xmm1,%xmm5

	xorps	%xmm6, %xmm6		/* SINH zero out this register */
	psubd	RZ_OFF(104)(%rsp), %xmm6
	movdqa	%xmm6, RZ_OFF(104)(%rsp) /* Now contains -n */

	movq	$0x1f, %rsi
	and 	%r10d, %esi
	movq	$0x1f, %rdi
	and 	%r11d, %edi

	movapd	.L__real_3fe0000000000000(%rip),%xmm6  /* SINH needs */
	movapd	.L__real_3fe0000000000000(%rip),%xmm7

	sub 	%ecx,%r8d
	sar 	$5,%r8d
	sub 	%edx,%r9d
	sar 	$5,%r9d

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	mulpd	.L__real_3FC5555555548F7C(%rip),%xmm0
	mulpd	.L__real_3FC5555555548F7C(%rip),%xmm1

	sub 	%esi,%r10d
	sar 	$5,%r10d
	sub 	%edi,%r11d
	sar 	$5,%r11d

	mulpd	%xmm2,%xmm2
	mulpd	%xmm3,%xmm3
	subpd	%xmm0,%xmm6   /* SINH exp(-x) */
	subpd	%xmm1,%xmm7   /* SINH exp(-x) */
	addpd	.L__real_3fe0000000000000(%rip),%xmm0
	addpd	.L__real_3fe0000000000000(%rip),%xmm1
	mulpd	%xmm2,%xmm6   /* SINH exp(-x) */
	mulpd	%xmm3,%xmm7   /* SINH exp(-x) */
	mulpd	%xmm0,%xmm2
	mulpd	%xmm1,%xmm3

	movlpdMR	(%rax,%rdx,8),%xmm0
	movhpd	(%rax,%rcx,8),%xmm0
	movlpdMR	(%rax,%rdi,8),%xmm1
	movhpd	(%rax,%rsi,8),%xmm1
	movq	$0x1f, %rcx
	andl	RZ_OFF(92)(%rsp), %ecx
	movq	$0x1f, %rdx
	andl	RZ_OFF(96)(%rsp), %edx
	movq	$0x1f, %rsi
	andl	RZ_OFF(100)(%rsp), %esi
	movq	$0x1f, %rdi
	andl	RZ_OFF(104)(%rsp), %edi

	subpd	%xmm4,%xmm6   /* SINH exp(-x) */
	subpd	%xmm5,%xmm7   /* SINH exp(-x) */
	addpd	%xmm4,%xmm2
	addpd	%xmm5,%xmm3

	movlpdMR	(%rax,%rdx,8),%xmm4
	movhpd	(%rax,%rcx,8),%xmm4
	movlpdMR	(%rax,%rdi,8),%xmm5
	movhpd	(%rax,%rsi,8),%xmm5

	/* *z2 = f2 + ((f1 + f2) * q); */
        add 	$1022, %r8d	/* add bias */
        add 	$1022, %r9d	/* add bias */
        add 	$1022, %r10d	/* add bias */
        add 	$1022, %r11d	/* add bias */

	/* deal with infinite and denormal results */
	mulpd	%xmm0,%xmm2
	mulpd	%xmm1,%xmm3
	mulpd	%xmm4,%xmm6   /* SINH exp(-x) */
	mulpd	%xmm5,%xmm7   /* SINH exp(-x) */
        shlq	$52,%r8
        shlq	$52,%r9
        shlq	$52,%r10
        shlq	$52,%r11
	addpd	%xmm0,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
	addpd	%xmm1,%xmm3  /* z = z1 + z2   done with 1,2,3,4,5 */
	addpd	%xmm4,%xmm6   /* SINH exp(-x) */
	addpd	%xmm5,%xmm7   /* SINH exp(-x) */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%r8,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */

	movq	%r11,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r10,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */

	mov	RZ_OFF(92)(%rsp),%r8d
	mov	RZ_OFF(96)(%rsp),%r9d
	mov	RZ_OFF(100)(%rsp),%r10d
	mov	RZ_OFF(104)(%rsp),%r11d

	mulpd	RZ_OFF(24)(%rsp),%xmm2	/* result *= 2^n */
	mulpd	RZ_OFF(40)(%rsp),%xmm3	/* result *= 2^n */

	sub 	%ecx,%r8d
	sar 	$5,%r8d
	sub 	%edx,%r9d
	sar 	$5,%r9d
	sub 	%esi,%r10d
	sar 	$5,%r10d
	sub 	%edi,%r11d
	sar 	$5,%r11d

        add 	$1022, %r8d	/* add bias */
        add 	$1022, %r9d	/* add bias */
        add 	$1022, %r10d	/* add bias */
        add 	$1022, %r11d	/* add bias */

        shlq	$52,%r8
        shlq	$52,%r9
        shlq	$52,%r10
        shlq	$52,%r11

	movq	%r9,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%r8,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */

	movq	%r11,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r10,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */

	mulpd	RZ_OFF(24)(%rsp),%xmm6	/* result *= 2^n */
	mulpd	RZ_OFF(40)(%rsp),%xmm7	/* result *= 2^n */

	subpd	%xmm6,%xmm2	/* SINH result = exp(x) - exp(-x) */
	subpd	%xmm7,%xmm3	/* SINH result = exp(x) - exp(-x) */

	cvtpd2ps %xmm2,%xmm0
	cvtpd2ps %xmm3,%xmm1
	shufps	$68,%xmm1,%xmm0

LBL(.L_fvsinh_final_check):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
	movdqa	RZ_OFF(88)(%rsp), %xmm7
#endif

	RZ_POP
	rep
	ret

LBL(.L__Scalar_fvssinh):
        /* Need to restore callee-saved regs can do here for this path
         * because entry was only thru fvs_sinh/fvs_sinh
         */
#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
	movdqa	RZ_OFF(88)(%rsp), %xmm7
#endif
        pushq   %rbp		/* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        movaps  %xmm0, _SX0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fss_sinh))
#else
        CALL(ENT(__fmth_i_sinh))
#endif
        movss   %xmm0, _SR0(%rsp)

        movss   _SX1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_sinh))
#else
        CALL(ENT(__fmth_i_sinh))
#endif
        movss   %xmm0, _SR1(%rsp)

        movss   _SX2(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_sinh))
#else
        CALL(ENT(__fmth_i_sinh))
#endif
        movss   %xmm0, _SR2(%rsp)

        movss   _SX3(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_sinh))
#else
        CALL(ENT(__fmth_i_sinh))
#endif
        movss   %xmm0, _SR3(%rsp)

        movaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

        ELF_FUNC(ENT_GH(__fvssinh))
        ELF_SIZE(ENT_GH(__fvssinh))
        IF_GH(ELF_FUNC(__fvs_sinh))
        IF_GH(ELF_SIZE(__fvs_sinh))


/* ============================================================
 * 
 *  A vector implementation of the sinh libm function.
 * 
 *  Prototype:
 * 
 *      __m128d __fvdsinh(__m128d x);
 * 
 *  Computes the hyperbolic sine of x
 * 
 */

        .text
        ALN_FUNC
	IF_GH(.globl ENT(__fvd_sinh))
	.globl ENT_GH(__fvdsinh)
IF_GH(ENT(__fvd_sinh):)
ENT_GH(__fvdsinh):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	%xmm0, %xmm2
#ifdef GH_TARGET
	movddup	.L__dsinh_too_small(%rip),%xmm5
#else
	movlpd	.L__dsinh_too_small(%rip),%xmm5
	movhpd	.L__dsinh_too_small(%rip),%xmm5
#endif
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulpd	%xmm0,%xmm3 

	/* save x for later. */
	andpd	.L__real_mask_unsign(%rip), %xmm2

        /* Set n = nearest integer to r */
	cmppd	$5, %xmm2, %xmm5
	cmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2
	movmskpd %xmm5, %r9d
	movmskpd %xmm2, %r8d

	testl	$3, %r9d
	jnz	LBL(.L__Scalar_fvdsinh)

	testl	$3, %r8d
	jnz	LBL(.L__Scalar_fvdsinh)

#if defined(_WIN64)
        movdqa  %xmm6, RZ_OFF(72)(%rsp)
#endif

	cvtpd2dq %xmm3,%xmm4
	cvtdq2pd %xmm4,%xmm1

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	movapd	.L__real_log2_by_32_lead(%rip),%xmm2
	mulpd	%xmm1,%xmm2
	movq	 %xmm4,RZ_OFF(24)(%rsp)
	/* r2 =   - n * logbaseof2_by_32_trail; */
	subpd	%xmm2,%xmm0	/* r1 in xmm0, */
	mulpd	.L__real_log2_by_32_tail(%rip),%xmm1 	/* r2 in xmm1 */

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	movl	RZ_OFF(24)(%rsp),%ecx
	andl	%ecx,%r9d

	movl	RZ_OFF(20)(%rsp),%edx
	andl	%edx,%r8d
	movapd	%xmm0,%xmm2

	xorl	%r10d,%r10d
	xorl	%r11d,%r11d
	subl	%ecx,%r10d
	subl	%edx,%r11d
	movl	%r10d,RZ_OFF(24)(%rsp)
	movl	%r11d,RZ_OFF(20)(%rsp)

	/* f1 = two_to_jby32_lead_table[j]; */
	/* f2 = two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%r9d,%ecx
	sarl	$5,%ecx
	subl	%r8d,%edx
	sarl	$5,%edx
	addpd	%xmm1,%xmm2    /* r = r1 + r2 */

	andl	$0x01f,%r10d
	andl	$0x01f,%r11d
	movl	%r10d,RZ_OFF(16)(%rsp)
	movl	%r11d,RZ_OFF(12)(%rsp)

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	 * r*r*( 5.00000000000000008883e-01 +
	 * r*( 1.66666666665260878863e-01 +
	 * r*( 4.16666666662260795726e-02 +
	 * r*( 8.33336798434219616221e-03 +
	 * r*( 1.38889490863777199667e-03 ))))));
	 * q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 *
	 * q = r + r^2*c1 + r^3*c2 + r^4*c3 + r^5*c4 + r^6*c5 *
	 * q = -r + r^2*c1 - r^3*c2 + r^4*c3 - r^5*c4 + r^6*c5 */
	movapd	%xmm2,%xmm1

	movapd	.L__real_3f56c1728d739765(%rip),%xmm3
	movapd	.L__real_3FC5555555548F7C(%rip),%xmm0
	movapd	.L__real_3F811115B7AA905E(%rip),%xmm5
	movapd	.L__real_3fe0000000000000(%rip),%xmm6

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	leaq	.L__two_to_jby32_table(%rip),%r11

	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

	mulpd	%xmm2,%xmm3	/* r*c5 */
	mulpd	%xmm2,%xmm0	/* r*c2 */
	mulpd	%xmm2,%xmm1	/* r*r */
	movapd	%xmm1,%xmm4

	subpd	%xmm3,%xmm5	/* c4 - r*c5 */
	addpd	 .L__real_3F811115B7AA905E(%rip),%xmm3  /* c4 + r*c5 */
	subpd	%xmm0,%xmm6	                        /* c1 - r*c2 */
	addpd	 .L__real_3fe0000000000000(%rip),%xmm0  /* c1 + r*c2 */
	mulpd	%xmm1,%xmm4	/* r^4 */
	mulpd	%xmm2,%xmm5	/* r*c4 - r^2*c5 */
	mulpd	%xmm2,%xmm3	/* r*c4 + r^2*c5 */

	mulpd	%xmm1,%xmm6	/* r^2*c1 - r^3*c2 */
	mulpd	%xmm1,%xmm0	/* r^2*c1 + r^3*c2 */
	subpd	.L__real_3FA5555555545D4E(%rip),%xmm5 /* -c3 + r*c4 - r^2*c5 */
	addpd	.L__real_3FA5555555545D4E(%rip),%xmm3 /* c3 + r*c4 + r^2*c5 */
	subpd	%xmm2,%xmm6	/* -r + r^2*c1 - r^3*c2 */
	addpd	%xmm2,%xmm0	/* r + r^2*c1 + r^3*c2 */
	mulpd	%xmm4,%xmm5	/* -r^4*c3 + r^5*c4 - r^6*c5 */
	mulpd	%xmm4,%xmm3	/* r^4*c3 + r^5*c4 + r^6*c5 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1022,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	subpd	%xmm5,%xmm6	/* q = final sum */
	addpd	%xmm3,%xmm0	/* q = final sum */

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	movhpd	(%r11,%r8,8),%xmm5 	/* f1 + f2 */
	movl	RZ_OFF(16)(%rsp),%r9d
	movl	RZ_OFF(12)(%rsp),%r8d

	movlpdMR	(%r11,%r9,8),%xmm4 	/* f1 + f2 */
	movhpd	(%r11,%r8,8),%xmm4 	/* f1 + f2 */

	mulpd	%xmm5,%xmm0
	addpd	%xmm5,%xmm0		/* z = z1 + z2 */

	mulpd	%xmm4,%xmm6
	addpd	%xmm4,%xmm6		/* z = z1 + z2 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r11		/* 1 */
	addq	$1022,%rdx		/* add bias */
	cmovleq	%rdx, %r11
	cmovleq	%rax, %rdx
	addq	$1022,%r11		/* add bias */
	shlq	$52, %r11		/* build 2^n */

	/* Step 3. Reconstitute. */
	movq	%r10,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r11,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(56)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(48)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(56)(%rsp),%xmm0  /* result*= 2^n */

	movl	RZ_OFF(24)(%rsp),%ecx
	movl	RZ_OFF(20)(%rsp),%edx
	subl	%r9d,%ecx
	sarl	$5,%ecx
	subl	%r8d,%edx
	sarl	$5,%edx

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1022,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	/* deal with denormal and close to infinity */
	movq	%rax, %r11		/* 1 */
	addq	$1022,%rdx		/* add bias */
	cmovleq	%rdx, %r11
	cmovleq	%rax, %rdx
	addq	$1022,%r11		/* add bias */
	shlq	$52, %r11		/* build 2^n */

	/* Step 3. Reconstitute. */
	movq	%r10,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r11,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm6  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(24)(%rsp),%xmm6  /* result*= 2^n */

	subpd	%xmm6,%xmm0		/* done with sinh */

#if defined(_WIN64)
	movdqa  RZ_OFF(72)(%rsp),%xmm6
 
#endif

	RZ_POP
	rep
	ret

#define _DX0 0
#define _DX1 8
#define _DX2 16
#define _DX3 24

#define _DR0 32
#define _DR1 40

LBL(.L__Scalar_fvdsinh):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _DX0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fsd_sinh))
#else
        CALL(ENT(__fmth_i_dsinh))
#endif
        movsd   %xmm0, _DR0(%rsp)

        movsd   _DX1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fsd_sinh))
#else
        CALL(ENT(__fmth_i_dsinh))
#endif
        movsd   %xmm0, _DR1(%rsp)

        movapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

        ELF_FUNC(ENT_GH(__fvdsinh))
        ELF_SIZE(ENT_GH(__fvdsinh))
	IF_GH(ELF_FUNC(__fvd_sinh))
	IF_GH(ELF_SIZE(__fvd_sinh))


/* ============================================================
 *  fastcosh.s
 * 
 *  An implementation of the cosh libm function.
 * 
 *  Prototype:
 * 
 *      float fastcosh(float x);
 * 
 *    Computes the hyperbolic cosine.
 * 
 */
	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fss_cosh))
	.globl ENT_GH(__fmth_i_cosh)
IF_GH(ENT(__fss_cosh):)
ENT_GH(__fmth_i_cosh):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
#ifdef GH_TARGET
	unpcklps %xmm0, %xmm0
	cvtps2pd %xmm0, %xmm2
#else
	cvtss2sd %xmm0, %xmm2
#endif
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm2,%xmm3 

	/* Set n = nearest integer to r */
	comiss	.L_sp_sinh_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_inf)
	comiss	.L_sp_sinh_min_singleval(%rip), %xmm0
	jb	LBL(.L_sp_cosh_ninf)

	cvtpd2dq %xmm3,%xmm4	/* convert to integer */
	cvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
	mulsd	.L__real_log2_by_32(%rip),%xmm1
	movd	%xmm4,%ecx
	subsd	%xmm1,%xmm2	/* r1 in xmm2, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* j = n & 0x0000001f; */
	movq	%r8,%rax
	and	%ecx,%eax
	subl	%ecx,%r9d
	and	%r9d,%r8d

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%eax,%ecx
	sar	$5,%ecx

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm1
	movlpdMR	.L__real_3fe0000000000000(%rip),%xmm5
	sub	%r8d,%r9d
	sar	$5,%r9d

	movsdRR	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	mulsd	%xmm2,%xmm2
	subsd	%xmm1,%xmm5                             /* exp(-x) */
	addsd	.L__real_3fe0000000000000(%rip),%xmm1   /* exp(x) */
	mulsd	%xmm2,%xmm5        /* exp(-x) */
	mulsd	%xmm1,%xmm2        /* exp(x) */

	movlpdMR	(%rdx,%r8,8),%xmm3   /* exp(-x) */
	movlpdMR	(%rdx,%rax,8),%xmm4   /* exp(x) */
	subsd	%xmm0,%xmm5        /* exp(-x) */
	addsd	%xmm0,%xmm2        /* exp(x) */

	/* *z2 = f2 + ((f1 + f2) * q); */
        add	$1022, %ecx	/* add bias */
        add	$1022, %r9d	/* add bias */

	mulsd	%xmm3,%xmm5
	mulsd	%xmm4,%xmm2
        shlq	$52,%rcx        /* build 2^n */
        shlq	$52,%r9         /* build 2^n */
	addsd	%xmm3,%xmm5  /* z = z1 + z2   done with 1,2,3,4,5 */
	addsd	%xmm4,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(16)(%rsp),%xmm5	/* result *= 2^n */
	mulsd	RZ_OFF(24)(%rsp),%xmm2	/* result *= 2^n */
	addsd	%xmm5,%xmm2		/* result = exp(x) + exp(-x) */
#ifdef GH_TARGET
	unpcklpd %xmm2, %xmm2
	cvtpd2ps %xmm2, %xmm0
#else
	cvtsd2ss %xmm2,%xmm0
#endif

	RZ_POP
	rep
	ret

        ELF_FUNC(ENT_GH(__fmth_i_cosh))
        ELF_SIZE(ENT_GH(__fmth_i_cosh))
	IF_GH(ELF_FUNC(__fss_cosh))
	IF_GH(ELF_SIZE(__fss_cosh))


/* ============================================================
 *  fastcosh.s
 * 
 *  An implementation of the dcosh libm function.
 * 
 *  Prototype:
 * 
 *      double fastcosh(double x);
 * 
 *    Computes the hyperbolic cosine.
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fsd_cosh))
	.globl ENT_GH(__fmth_i_dcosh)
IF_GH(ENT(__fsd_cosh):)
ENT_GH(__fmth_i_dcosh):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(40)(%rsp)
#endif

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulsd	%xmm0,%xmm3 

	/* Set n = nearest integer to r */
	comisd	.L_sinh_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	comisd	.L_sinh_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_cosh_ninf)
	cvtpd2dq %xmm3,%xmm4	/* convert to integer */
	cvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
	movlpdMR	.L__real_log2_by_32_lead(%rip),%xmm2
	mulsd	%xmm1,%xmm2
	movd	%xmm4,%ecx
	subsd	%xmm2,%xmm0	/* r1 in xmm0, */
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
	mulsd	.L__real_log2_by_32_tail(%rip),%xmm1

	/* j = n & 0x0000001f; */
	movq	%r8,%rax
	andl	%ecx,%eax
	subl	%ecx,%r9d
	andl	%r9d,%r8d

	movsdRR	%xmm0,%xmm2

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%eax,%ecx
	sarl	$5,%ecx
	addsd	%xmm1,%xmm2    /* r = r1 + r2 */
	subl	%r8d,%r9d
	sarl	$5,%r9d

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	movsdRR	%xmm2,%xmm1
	movlpdMR	.L__real_3f56c1728d739765(%rip),%xmm3
	movlpdMR	.L__real_3FC5555555548F7C(%rip),%xmm0
	movlpdMR	.L__real_3F811115B7AA905E(%rip),%xmm5
	movlpdMR	.L__real_3fe0000000000000(%rip),%xmm6
	mulsd	%xmm2,%xmm3
	mulsd	%xmm2,%xmm0
	mulsd	%xmm2,%xmm1
	movsdRR	%xmm1,%xmm4
	subsd	%xmm3,%xmm5
	addsd	.L__real_3F811115B7AA905E(%rip),%xmm3
	subsd	%xmm0,%xmm6
	addsd	.L__real_3fe0000000000000(%rip),%xmm0
	mulsd	%xmm1,%xmm4
	mulsd	%xmm2,%xmm5
	mulsd	%xmm2,%xmm3
	mulsd	%xmm1,%xmm6
	mulsd	%xmm1,%xmm0
	subsd	.L__real_3FA5555555545D4E(%rip),%xmm5
	addsd	.L__real_3FA5555555545D4E(%rip),%xmm3
	subsd	%xmm2,%xmm6
	addsd	%xmm2,%xmm0
	mulsd	%xmm4,%xmm5
	mulsd	%xmm4,%xmm3
	subsd	%xmm5,%xmm6
	addsd	%xmm3,%xmm0

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%rdx,%r8,8),%xmm4   /* exp(-x) */
	movlpdMR	(%rdx,%rax,8),%xmm5   /* exp(x) */
	/* deal with infinite results */
        movslq	%ecx,%rcx
	mulsd	%xmm5,%xmm0
	addsd	%xmm5,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* deal with denormal results */
	movq	$1, %rdx
	movq	$1, %rax
        addq	$1022, %rcx	/* add bias */
	cmovleq	%rcx, %rdx
	cmovleq	%rax, %rcx
        shlq	$52,%rcx        /* build 2^n */
        addq	$1022, %rdx	/* add bias */
        shlq	$52,%rdx        /* build 2^n */
	movq	%rdx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm0	/* result *= 2^n */

	/* deal with infinite results */
        movslq	%r9d,%rcx
	mulsd	%xmm4,%xmm6
	addsd	%xmm4,%xmm6  /* z = z1 + z2   done with 1,2,3,4,5 */

	/* deal with denormal results */
	movq	$1, %rdx
	movq	$1, %rax
        addq	$1022, %rcx	/* add bias */
	cmovleq	%rcx, %rdx
	cmovleq	%rax, %rcx
        shlq	$52,%rcx        /* build 2^n */
        addq	$1022, %rdx	/* add bias */
        shlq	$52,%rdx        /* build 2^n */
	movq	%rdx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm6	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	mulsd	RZ_OFF(24)(%rsp),%xmm6	/* result *= 2^n */
	addsd	%xmm6, %xmm0

#if defined(_WIN64)
	movdqa	RZ_OFF(40)(%rsp), %xmm6
#endif

	RZ_POP
	rep
	ret

        ELF_FUNC(ENT_GH(__fmth_i_dcosh))
        ELF_SIZE(ENT_GH(__fmth_i_dcosh))
	IF_GH(ELF_FUNC(__fsd_cosh))
	IF_GH(ELF_SIZE(__fsd_cosh))


/* ============================================================
 *  vector fastcoshf.s
 * 
 *  An implementation of the cosh libm function.
 * 
 *  Prototype:
 * 
 *      float fastcoshf(float x);
 * 
 *    Computes hyperbolic cosine of x
 * 
 */

	.text
        ALN_FUNC
	IF_GH(.globl ENT(__fvs_cosh))
	.globl ENT_GH(__fvscosh)
IF_GH(ENT(__fvs_cosh):)
ENT_GH(__fvscosh):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
	movdqa	%xmm7, RZ_OFF(88)(%rsp)  /* COSH needs xmm7 */
#endif

	/* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movhlps  %xmm0, %xmm1
	movaps	 %xmm0, %xmm5
	cvtps2pd %xmm0, %xmm2		/* xmm2 = dble(a(2)), dble(a(1)) */
	cvtps2pd %xmm1, %xmm1		/* xmm1 = dble(a(4)), dble(a(3)) */
	andps	 .L__ps_mask_unsign(%rip), %xmm5
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm4
	cmpps	$6, .L__sp_ln_max_singleval(%rip), %xmm5
	mulpd	%xmm2, %xmm3 
	mulpd	%xmm1, %xmm4 
	movmskps %xmm5, %r8d

	/* Set n = nearest integer to r */
	cvtpd2dq %xmm3,%xmm5	/* convert to integer */
	cvtpd2dq %xmm4,%xmm6	/* convert to integer */
	test	 $15, %r8d
	cvtdq2pd %xmm5,%xmm3	/* and back to float. */
	cvtdq2pd %xmm6,%xmm4	/* and back to float. */
	jnz	LBL(.L__Scalar_fvscosh)

	/* r1 = x - n * logbaseof2_by_32_lead; */
	mulpd	.L__real_log2_by_32(%rip),%xmm3
	mulpd	.L__real_log2_by_32(%rip),%xmm4
	movq	%xmm5,RZ_OFF(96)(%rsp)
	movq	%xmm6,RZ_OFF(104)(%rsp)
	subpd	%xmm3,%xmm2	/* r1 in xmm2, */
	subpd	%xmm4,%xmm1	/* r1 in xmm1, */
	leaq	.L__two_to_jby32_table(%rip),%rax

	/* j = n & 0x0000001f; */
	mov	RZ_OFF(92)(%rsp),%r8d
	mov	RZ_OFF(96)(%rsp),%r9d
	mov	RZ_OFF(100)(%rsp),%r10d
	mov	RZ_OFF(104)(%rsp),%r11d
	movq	$0x1f, %rcx
	and 	%r8d, %ecx
	movq	$0x1f, %rdx
	and 	%r9d, %edx
	movapd	%xmm2,%xmm0
	movapd	%xmm1,%xmm3
	movapd	%xmm2,%xmm4
	movapd	%xmm1,%xmm5

	xorps	%xmm6, %xmm6		/* COSH zero out this register */
	psubd	RZ_OFF(104)(%rsp), %xmm6
	movdqa	%xmm6, RZ_OFF(104)(%rsp) /* Now contains -n */

	movq	$0x1f, %rsi
	and 	%r10d, %esi
	movq	$0x1f, %rdi
	and 	%r11d, %edi

	movapd	.L__real_3fe0000000000000(%rip),%xmm6  /* COSH needs */
	movapd	.L__real_3fe0000000000000(%rip),%xmm7

	sub 	%ecx,%r8d
	sar 	$5,%r8d
	sub 	%edx,%r9d
	sar 	$5,%r9d

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	mulpd	.L__real_3FC5555555548F7C(%rip),%xmm0
	mulpd	.L__real_3FC5555555548F7C(%rip),%xmm1

	sub 	%esi,%r10d
	sar 	$5,%r10d
	sub 	%edi,%r11d
	sar 	$5,%r11d

	mulpd	%xmm2,%xmm2
	mulpd	%xmm3,%xmm3
	subpd	%xmm0,%xmm6   /* COSH exp(-x) */
	subpd	%xmm1,%xmm7   /* COSH exp(-x) */
	addpd	.L__real_3fe0000000000000(%rip),%xmm0
	addpd	.L__real_3fe0000000000000(%rip),%xmm1
	mulpd	%xmm2,%xmm6   /* COSH exp(-x) */
	mulpd	%xmm3,%xmm7   /* COSH exp(-x) */
	mulpd	%xmm0,%xmm2
	mulpd	%xmm1,%xmm3

	movlpdMR	(%rax,%rdx,8),%xmm0
	movhpd	(%rax,%rcx,8),%xmm0
	movlpdMR	(%rax,%rdi,8),%xmm1
	movhpd	(%rax,%rsi,8),%xmm1
	movq	$0x1f, %rcx
	andl	RZ_OFF(92)(%rsp), %ecx
	movq	$0x1f, %rdx
	andl	RZ_OFF(96)(%rsp), %edx
	movq	$0x1f, %rsi
	andl	RZ_OFF(100)(%rsp), %esi
	movq	$0x1f, %rdi
	andl	RZ_OFF(104)(%rsp), %edi

	subpd	%xmm4,%xmm6   /* COSH exp(-x) */
	subpd	%xmm5,%xmm7   /* COSH exp(-x) */
	addpd	%xmm4,%xmm2
	addpd	%xmm5,%xmm3

	movlpdMR	(%rax,%rdx,8),%xmm4
	movhpd	(%rax,%rcx,8),%xmm4
	movlpdMR	(%rax,%rdi,8),%xmm5
	movhpd	(%rax,%rsi,8),%xmm5

	/* *z2 = f2 + ((f1 + f2) * q); */
        add 	$1022, %r8d	/* add bias */
        add 	$1022, %r9d	/* add bias */
        add 	$1022, %r10d	/* add bias */
        add 	$1022, %r11d	/* add bias */

	/* deal with infinite and denormal results */
	mulpd	%xmm0,%xmm2
	mulpd	%xmm1,%xmm3
	mulpd	%xmm4,%xmm6   /* COSH exp(-x) */
	mulpd	%xmm5,%xmm7   /* COSH exp(-x) */
        shlq	$52,%r8
        shlq	$52,%r9
        shlq	$52,%r10
        shlq	$52,%r11
	addpd	%xmm0,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
	addpd	%xmm1,%xmm3  /* z = z1 + z2   done with 1,2,3,4,5 */
	addpd	%xmm4,%xmm6   /* COSH exp(-x) */
	addpd	%xmm5,%xmm7   /* COSH exp(-x) */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%r8,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */

	movq	%r11,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r10,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */

	mov	RZ_OFF(92)(%rsp),%r8d
	mov	RZ_OFF(96)(%rsp),%r9d
	mov	RZ_OFF(100)(%rsp),%r10d
	mov	RZ_OFF(104)(%rsp),%r11d

	mulpd	RZ_OFF(24)(%rsp),%xmm2	/* result *= 2^n */
	mulpd	RZ_OFF(40)(%rsp),%xmm3	/* result *= 2^n */

	sub 	%ecx,%r8d
	sar 	$5,%r8d
	sub 	%edx,%r9d
	sar 	$5,%r9d
	sub 	%esi,%r10d
	sar 	$5,%r10d
	sub 	%edi,%r11d
	sar 	$5,%r11d

        add 	$1022, %r8d	/* add bias */
        add 	$1022, %r9d	/* add bias */
        add 	$1022, %r10d	/* add bias */
        add 	$1022, %r11d	/* add bias */

        shlq	$52,%r8
        shlq	$52,%r9
        shlq	$52,%r10
        shlq	$52,%r11

	movq	%r9,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%r8,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */

	movq	%r11,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r10,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */

	mulpd	RZ_OFF(24)(%rsp),%xmm6	/* result *= 2^n */
	mulpd	RZ_OFF(40)(%rsp),%xmm7	/* result *= 2^n */

	addpd	%xmm6,%xmm2	/* COSH result = exp(x) + exp(-x) */
	addpd	%xmm7,%xmm3	/* COSH result = exp(x) + exp(-x) */

	cvtpd2ps %xmm2,%xmm0
	cvtpd2ps %xmm3,%xmm1
	shufps	$68,%xmm1,%xmm0

LBL(.L_fvcosh_final_check):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
	movdqa	RZ_OFF(88)(%rsp), %xmm7
#endif

	RZ_POP
	rep
	ret

LBL(.L__Scalar_fvscosh):
        /* Need to restore callee-saved regs can do here for this path
         * because entry was only thru fvs_cosh_fma4/fvs_cosh_vex
         */
#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
	movdqa	RZ_OFF(88)(%rsp), %xmm7
#endif
        pushq   %rbp		/* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        movaps  %xmm0, _SX0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fss_cosh))
#else
        CALL(ENT(__fmth_i_cosh))
#endif
        movss   %xmm0, _SR0(%rsp)

        movss   _SX1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_cosh))
#else
        CALL(ENT(__fmth_i_cosh))
#endif
        movss   %xmm0, _SR1(%rsp)

        movss   _SX2(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_cosh))
#else
        CALL(ENT(__fmth_i_cosh))
#endif
        movss   %xmm0, _SR2(%rsp)

        movss   _SX3(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fss_cosh))
#else
        CALL(ENT(__fmth_i_cosh))
#endif
        movss   %xmm0, _SR3(%rsp)

        movaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

        ELF_FUNC(ENT_GH(__fvscosh))
        ELF_SIZE(ENT_GH(__fvscosh))
        IF_GH(ELF_FUNC(__fvs_cosh))
        IF_GH(ELF_SIZE(__fvs_cosh))


/* ============================================================
 * 
 *  A vector implementation of the cosh libm function.
 * 
 *  Prototype:
 * 
 *      __m128d __fvdcosh(__m128d x);
 * 
 *  Computes the hyperbolic cosine of x
 * 
 */

        .text
        ALN_FUNC
	IF_GH(.globl ENT(__fvd_cosh))
	.globl ENT_GH(__fvdcosh)
IF_GH(ENT(__fvd_cosh):)
ENT_GH(__fvdcosh):
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	movapd	%xmm0, %xmm2
	movapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	mulpd	%xmm0,%xmm3 

	/* save x for later. */
	andpd	.L__real_mask_unsign(%rip), %xmm2

        /* Set n = nearest integer to r */
	cvtpd2dq %xmm3,%xmm4
	cmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2
	cvtdq2pd %xmm4,%xmm1
	movmskpd %xmm2, %r8d

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	movapd	.L__real_log2_by_32_lead(%rip),%xmm2
	mulpd	%xmm1,%xmm2
	movq	 %xmm4,RZ_OFF(24)(%rsp)
	testl	$3, %r8d
	jnz	LBL(.L__Scalar_fvdcosh)

#if defined(_WIN64)
        movdqa  %xmm6, RZ_OFF(72)(%rsp)
#endif

	/* r2 =   - n * logbaseof2_by_32_trail; */
	subpd	%xmm2,%xmm0	/* r1 in xmm0, */
	mulpd	.L__real_log2_by_32_tail(%rip),%xmm1 	/* r2 in xmm1 */

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	movl	RZ_OFF(24)(%rsp),%ecx
	andl	%ecx,%r9d

	movl	RZ_OFF(20)(%rsp),%edx
	andl	%edx,%r8d
	movapd	%xmm0,%xmm2

	xorl	%r10d,%r10d
	xorl	%r11d,%r11d
	subl	%ecx,%r10d
	subl	%edx,%r11d
	movl	%r10d,RZ_OFF(24)(%rsp)
	movl	%r11d,RZ_OFF(20)(%rsp)

	/* f1 = two_to_jby32_lead_table[j]; */
	/* f2 = two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%r9d,%ecx
	sarl	$5,%ecx
	subl	%r8d,%edx
	sarl	$5,%edx
	addpd	%xmm1,%xmm2    /* r = r1 + r2 */

	andl	$0x01f,%r10d
	andl	$0x01f,%r11d
	movl	%r10d,RZ_OFF(16)(%rsp)
	movl	%r11d,RZ_OFF(12)(%rsp)

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +	
	 * r*r*( 5.00000000000000008883e-01 +
	 * r*( 1.66666666665260878863e-01 +
	 * r*( 4.16666666662260795726e-02 +
	 * r*( 8.33336798434219616221e-03 +
	 * r*( 1.38889490863777199667e-03 ))))));
	 * q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 *
	 * q = r + r^2*c1 + r^3*c2 + r^4*c3 + r^5*c4 + r^6*c5 *
	 * q = -r + r^2*c1 - r^3*c2 + r^4*c3 - r^5*c4 + r^6*c5 */
	movapd	%xmm2,%xmm1

	movapd	.L__real_3f56c1728d739765(%rip),%xmm3
	movapd	.L__real_3FC5555555548F7C(%rip),%xmm0
	movapd	.L__real_3F811115B7AA905E(%rip),%xmm5
	movapd	.L__real_3fe0000000000000(%rip),%xmm6

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	leaq	.L__two_to_jby32_table(%rip),%r11

	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

	mulpd	%xmm2,%xmm3	/* r*c5 */
	mulpd	%xmm2,%xmm0	/* r*c2 */
	mulpd	%xmm2,%xmm1	/* r*r */
	movapd	%xmm1,%xmm4

	subpd	%xmm3,%xmm5	/* c4 - r*c5 */
	addpd	 .L__real_3F811115B7AA905E(%rip),%xmm3  /* c4 + r*c5 */
	subpd	%xmm0,%xmm6	                        /* c1 - r*c2 */
	addpd	 .L__real_3fe0000000000000(%rip),%xmm0  /* c1 + r*c2 */
	mulpd	%xmm1,%xmm4	/* r^4 */
	mulpd	%xmm2,%xmm5	/* r*c4 - r^2*c5 */
	mulpd	%xmm2,%xmm3	/* r*c4 + r^2*c5 */

	mulpd	%xmm1,%xmm6	/* r^2*c1 - r^3*c2 */
	mulpd	%xmm1,%xmm0	/* r^2*c1 + r^3*c2 */
	subpd	.L__real_3FA5555555545D4E(%rip),%xmm5 /* -c3 + r*c4 - r^2*c5 */
	addpd	.L__real_3FA5555555545D4E(%rip),%xmm3 /* c3 + r*c4 + r^2*c5 */
	subpd	%xmm2,%xmm6	/* -r + r^2*c1 - r^3*c2 */
	addpd	%xmm2,%xmm0	/* r + r^2*c1 + r^3*c2 */
	mulpd	%xmm4,%xmm5	/* -r^4*c3 + r^5*c4 - r^6*c5 */
	mulpd	%xmm4,%xmm3	/* r^4*c3 + r^5*c4 + r^6*c5 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1022,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	subpd	%xmm5,%xmm6	/* q = final sum */
	addpd	%xmm3,%xmm0	/* q = final sum */

	/* *z2 = f2 + ((f1 + f2) * q); */
	movlpdMR	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	movhpd	(%r11,%r8,8),%xmm5 	/* f1 + f2 */
	movl	RZ_OFF(16)(%rsp),%r9d
	movl	RZ_OFF(12)(%rsp),%r8d

	movlpdMR	(%r11,%r9,8),%xmm4 	/* f1 + f2 */
	movhpd	(%r11,%r8,8),%xmm4 	/* f1 + f2 */

	mulpd	%xmm5,%xmm0
	addpd	%xmm5,%xmm0		/* z = z1 + z2 */

	mulpd	%xmm4,%xmm6
	addpd	%xmm4,%xmm6		/* z = z1 + z2 */

	/* deal with denormal and close to infinity */
	movq	%rax, %r11		/* 1 */
	addq	$1022,%rdx		/* add bias */
	cmovleq	%rdx, %r11
	cmovleq	%rax, %rdx
	addq	$1022,%r11		/* add bias */
	shlq	$52, %r11		/* build 2^n */

	/* Step 3. Reconstitute. */
	movq	%r10,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r11,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(56)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(48)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(56)(%rsp),%xmm0  /* result*= 2^n */

	movl	RZ_OFF(24)(%rsp),%ecx
	movl	RZ_OFF(20)(%rsp),%edx
	subl	%r9d,%ecx
	sarl	$5,%ecx
	subl	%r8d,%edx
	sarl	$5,%edx

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1022,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	/* deal with denormal and close to infinity */
	movq	%rax, %r11		/* 1 */
	addq	$1022,%rdx		/* add bias */
	cmovleq	%rdx, %r11
	cmovleq	%rax, %rdx
	addq	$1022,%r11		/* add bias */
	shlq	$52, %r11		/* build 2^n */

	/* Step 3. Reconstitute. */
	movq	%r10,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r11,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(40)(%rsp),%xmm6  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	mulpd	RZ_OFF(24)(%rsp),%xmm6  /* result*= 2^n */

	addpd	%xmm6,%xmm0		/* done with cosh */

#if defined(_WIN64)
        movdqa  RZ_OFF(72)(%rsp),%xmm6
#endif

	RZ_POP
	rep
	ret

#define _DX0 0
#define _DX1 8
#define _DX2 16
#define _DX3 24

#define _DR0 32
#define _DR1 40

LBL(.L__Scalar_fvdcosh):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _DX0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fsd_cosh))
#else
        CALL(ENT(__fmth_i_dcosh))
#endif
        movsd   %xmm0, _DR0(%rsp)

        movsd   _DX1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fsd_cosh))
#else
        CALL(ENT(__fmth_i_dcosh))
#endif
        movsd   %xmm0, _DR1(%rsp)

        movapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

        ELF_FUNC(ENT_GH(__fvdcosh))
        ELF_SIZE(ENT_GH(__fvdcosh))
	IF_GH(ELF_FUNC(__fvd_cosh))
	IF_GH(ELF_SIZE(__fvd_cosh))


/* ------------------------------------------------------------------------- */

/* ============================================================
 * 
 *  A scalar implementation of the single precision SINCOS() function.
 *
 *  __fss_sincos(float)
 * 
 *  Entry:
 *	(%xmm0-ss)	Angle
 *
 *  Exit:
 *	(%xmm0-ss)	SIN(angle)
 *	(%xmm1-ss)	COS(angle)
 * 
 */
        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fss_sincos))
        .globl ENT_GH(__fmth_i_sincos)
IF_GH(ENT(__fss_sincos):)
ENT_GH(__fmth_i_sincos):
        movd    %xmm0, %eax
        mov     $0x03f490fdb,%edx   /* pi / 4 */
        movss   .L__sngl_sixteen_by_pi(%rip),%xmm4
        and     .L__sngl_mask_unsign(%rip), %eax
        cmp     %edx,%eax
        jle     LBL(.L__fss_sincos_shortcuts)
        shrl    $20,%eax
        cmpl    $0x498,%eax
        jge     GBLTXT(ENT(__mth_i_sincos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulss   %xmm0,%xmm4 
#ifdef GH_TARGET
        unpcklps %xmm0, %xmm0
        cvtps2pd %xmm0, %xmm0
#else
        cvtss2sd %xmm0,%xmm0
#endif

        /* Set n = nearest integer to r */
        cvtss2si %xmm4,%rcx    /* convert to integer */
        movsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        movsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movsd   .L__dble_pi_by_16_us(%rip), %xmm3
        cvtsi2sd %rcx,%xmm4    /* and back to double */

        /* r = (x - n*p1) - (n*p2 + n*p3)  */
        mulsd   %xmm4,%xmm1     /* n * p1 */
        mulsd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        subsd   %xmm1,%xmm0     /* x - n * p1 == rh */
	addsd   %xmm2,%xmm3

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subsd   %xmm3,%xmm0     /* c = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        movsdRR   %xmm0,%xmm1     /* r in xmm1 */
        movsdRR   %xmm0,%xmm2     /* r in xmm2 */
        movsdRR   %xmm0,%xmm4     /* r in xmm4 */
        mulsd   %xmm0,%xmm0     /* r^2 in xmm0 */
        mulsd   %xmm1,%xmm1     /* r^2 in xmm1 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm2,%xmm3     /* r in xmm3 */

        /* xmm0, xmm1, xmm4 have r^2, xmm2, xmm3 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        mulsd   .L__dble_pq3(%rip), %xmm0     /* p3 * r^2 */
        mulsd   .L__dble_pq3+16(%rip), %xmm1  /* q3 * r^2 */
        addsd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm1  /* + q2 */
        mulsd   %xmm4,%xmm0                   /* * r^2 */
        mulsd   %xmm4,%xmm1                   /* * r^2 */

        mulsd   %xmm4,%xmm3                   /* xmm3 = r^3 */
        addsd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm1  /* + q1 */
        mulsd   %xmm3,%xmm0                   /* * r^3 */
        mulsd   %xmm4,%xmm1                   /* * r^2 = q(r) */

        addq    %rax,%rax
        addq    %rcx,%rcx
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */

        addsd   %xmm2,%xmm0                   /* + r  = p(r) */

	movsd   (%rdx,%rcx,8), %xmm5          /* Move C */
	movsd   (%rdx,%rax,8), %xmm2          /* Move S */
	movsdRR	%xmm1,%xmm3                   /* Move for cosine */
	movsdRR	%xmm0,%xmm4                   /* Move for sine */

        mulsd   %xmm5,%xmm0                   /* C * p(r) */
        mulsd   %xmm2,%xmm1                   /* S * q(r) */
        mulsd   %xmm5,%xmm3                   /* C * q(r) */
        mulsd   %xmm2,%xmm4                   /* S * p(r) */

        addsd   %xmm2,%xmm1                   /* S + S * q(r) */
        addsd   %xmm5,%xmm3                   /* C + C * q(r) */
        addsd   %xmm1,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        subsd   %xmm4,%xmm3                   /* cos(x) = (C + Cq(r)) - Sp(r) */

        shufpd  $0, %xmm3, %xmm0              /* Shuffle it in */

LBL(.L__fss_sincos_done1):
	cvtpd2ps %xmm0,%xmm0
	movaps   %xmm0, %xmm1
	shufps  $1, %xmm1, %xmm1             /* xmm1 now has cos(x) */
        ret

LBL(.L__fss_sincos_shortcuts):
#ifdef GH_TARGET
        unpcklps %xmm0, %xmm0
        cvtps2pd %xmm0, %xmm0
#else
        cvtss2sd %xmm0,%xmm0
#endif
	movlpdMR  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        movsdRR   %xmm0,%xmm2
        movsdRR   %xmm0,%xmm3
        shrl    $20,%eax
        cmpl    $0x0390,%eax
        jl      LBL(.L__fss_sincos_small)
        movsdRR   %xmm0,%xmm4
        mulsd   %xmm0,%xmm0
        mulsd   %xmm2,%xmm2
        mulsd   %xmm4,%xmm4

        mulsd   .L__dble_dsin_c4(%rip),%xmm0    /* x2 * s4 */
        mulsd   .L__dble_dcos_c4(%rip),%xmm2    /* x2 * c4 */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + s3 */
        addsd   .L__dble_dcos_c3(%rip),%xmm2    /* + c3 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s3 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + 22 */
        addsd   .L__dble_dcos_c2(%rip),%xmm2    /* + c2 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s2 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c2 + ...) */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + s1 */
        addsd   .L__dble_dcos_c1(%rip),%xmm2    /* + c1 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s1 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c1 + ...) */
        mulsd   %xmm3,%xmm0                     /* x3 * (s1 + ...) */
        addsd   .L__dble_pq1+16(%rip),%xmm2     /* - 0.5 */
        mulsd   %xmm4,%xmm2                     /* x2 * (0.5 + ...) */
        addsd   %xmm3,%xmm0                     /* x + x3 * (...) done */
        addsd   %xmm2,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
	shufpd  $0, %xmm1, %xmm0
	jmp 	LBL(.L__fss_sincos_done1)

LBL(.L__fss_sincos_small):
        cmpl    $0x0320,%eax
	shufpd  $0, %xmm1, %xmm0
        jl      LBL(.L__fss_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_pq1(%rip),%xmm3
        mulsd   %xmm2,%xmm3
        mulsd   .L__dble_pq1+16(%rip),%xmm2
        addsd   %xmm3,%xmm0
        addsd   %xmm2,%xmm1
	shufpd  $0, %xmm1, %xmm0
	jmp 	LBL(.L__fss_sincos_done1)

        ELF_FUNC(ENT_GH(__fmth_i_sincos))
        ELF_SIZE(ENT_GH(__fmth_i_sincos))
        IF_GH(ELF_FUNC(__fss_sincos))
        IF_GH(ELF_SIZE(__fss_sincos))


/* ------------------------------------------------------------------------- */

/* ============================================================
 * 
 *  A scalar implementation of the double precision SINCOS() function.
 *
 *  __fsd_sincos(double)
 * 
 *  Entry:
 *	(%xmm0-sd)	Angle
 *
 *  Exit:
 *	(%xmm0-sd)	SIN(angle)
 *	(%xmm1-sd)	COS(angle)
 * 
 */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fsd_sincos))
        .globl ENT_GH(__fmth_i_dsincos)
IF_GH(ENT(__fsd_sincos):)
ENT_GH(__fmth_i_dsincos):
        movd    %xmm0, %rax
        mov     $0x03fe921fb54442d18,%rdx
        movapd  .L__dble_sixteen_by_pi(%rip),%xmm4
        andq    .L__real_mask_unsign(%rip), %rax
        cmpq    %rdx,%rax
        jle     LBL(.L__fsd_sincos_shortcuts)
        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     GBLTXT(ENT(__mth_i_dsincos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulsd   %xmm0,%xmm4 

        RZ_PUSH
#if defined(_WIN64)
        movdqa  %xmm6, RZ_OFF(24)(%rsp)
        movdqa  %xmm7, RZ_OFF(40)(%rsp)
        movdqa  %xmm8, RZ_OFF(56)(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5    /* convert to integer */
        movsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        movsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movsd   .L__dble_pi_by_16_us(%rip), %xmm3
        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulsd   %xmm4,%xmm1     /* n * p1 */
        mulsd   %xmm4,%xmm2     /* n * p2 == rt */
        mulsd   %xmm4,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        movsdRR   %xmm0,%xmm6     /* x in xmm6 */
        subsd   %xmm1,%xmm0     /* x - n * p1 == rh */
        subsd   %xmm1,%xmm6     /* x - n * p1 == rh == c */

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subsd   %xmm2,%xmm0     /* rh = rh - rt */

        subsd   %xmm0,%xmm6     /* (c - rh) */
        movsdRR   %xmm0,%xmm1     /* Move rh */
        movsdRR   %xmm0,%xmm4     /* Move rh */
        movsdRR   %xmm0,%xmm5     /* Move rh */
        subsd   %xmm2,%xmm6     /* ((c - rh) - rt) */
        subsd   %xmm6,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        movsdRR   %xmm1,%xmm2     /* Move rh */
        subsd   %xmm3,%xmm0     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm4     /* c = rh - rt aka r */
        subsd   %xmm3,%xmm5     /* c = rh - rt aka r */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */
        
        subsd   %xmm0,%xmm1     /* (rh - c) */

        mulsd   %xmm0,%xmm0     /* r^2 in xmm0 */
        movsdRR   %xmm4,%xmm6     /* r in xmm6 */
        mulsd   %xmm4,%xmm4     /* r^2 in xmm4 */
        movsdRR   %xmm5,%xmm7     /* r in xmm7 */
        mulsd   %xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulsd   .L__dble_pq4(%rip), %xmm0     /* p4 * r^2 */
        subsd   %xmm6,%xmm2                   /* (rh - c) */
        mulsd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        subsd   %xmm3,%xmm1                   /* (rh - c) - rt aka rr */

        addsd   .L__dble_pq3(%rip), %xmm0     /* + p3 */
        addsd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        subsd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */

        mulsd   %xmm5,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        mulsd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        mulsd   %xmm5,%xmm7                   /* xmm7 = r^3 */
        movsdRR   %xmm1,%xmm3                   /* Move rr */
        mulsd   %xmm5,%xmm1                   /* r * r * rr */

        addsd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addsd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        mulsd   .L__dble_pq1+16(%rip), %xmm1  /* r * r * rr * 0.5 */
        mulsd   %xmm6, %xmm3                  /* r * rr */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        mulsd   %xmm5,%xmm0                   /* * r^2 */
        mulsd   %xmm5,%xmm4                   /* * r^2 */
        addsd   %xmm1,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        movlpdMR  8(%rdx,%rax,8),%xmm8          /* ds2 in xmm8 */
        movlpdMR  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3
           xmm8 is ds2 */

        addsd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addsd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulsd   %xmm7,%xmm0                   /* * r^3 */
        mulsd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addsd   %xmm2,%xmm0                   /* + cs  == dp aka p(r) */
        subsd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */
        movsdRR   %xmm1,%xmm3                   /* dc2 in xmm3 */
        movlpdMR   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        movlpdMR   (%rdx,%rcx,8),%xmm7          /* dc1 in xmm7 */
        addsd   %xmm6,%xmm0                   /* + r   == dp aka p(r) */
        movsdRR   %xmm8,%xmm2                   /* ds2 in xmm2 */

        mulsd   %xmm4,%xmm8                   /* ds2 * dq */
        mulsd   %xmm4,%xmm1                   /* dc2 * dq */

        addsd   %xmm2,%xmm8                   /* ds2 + ds2*dq */
        addsd   %xmm3,%xmm1                   /* dc2 + dc2*dq */

        mulsd   %xmm0,%xmm3                   /* dc2 * dp */
        mulsd   %xmm0,%xmm2                   /* ds2 * dp */
	movsdRR	%xmm4,%xmm6                   /* xmm6 = dq */

        addsd   %xmm3,%xmm8                   /* (ds2 + ds2*dq) + dc2*dp */
        subsd   %xmm2,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */

	movsdRR	%xmm5,%xmm3                   /* xmm3 = ds1 */
        mulsd   %xmm5,%xmm4                   /* ds1 * dq */
        mulsd   %xmm0,%xmm5                   /* ds1 * dp */
	mulsd   %xmm7,%xmm6                   /* dc1 * dq */

        mulsd   %xmm7,%xmm0                   /* dc1 * dp */
        addsd   %xmm4,%xmm8                   /* ((ds2...) + dc2*dp) + ds1*dq */
        subsd   %xmm5,%xmm1                   /* (() - ds2*dp) - ds1*dp */

        addsd   %xmm3,%xmm8                   /* + ds1 */
        addsd   %xmm6,%xmm1                   /* + dc1*dq */

        addsd   %xmm8,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        addsd   %xmm7,%xmm1                   /* cos(x) = (C + Cq(r)) + Sq(r) */

#if defined(_WIN64)
        movdqa  RZ_OFF(24)(%rsp),%xmm6
        movdqa  RZ_OFF(40)(%rsp),%xmm7
        movdqa  RZ_OFF(56)(%rsp),%xmm8
#endif
        RZ_POP
        ret

LBL(.L__fsd_sincos_shortcuts):
        movlpdMR  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        movsdRR   %xmm0,%xmm2
        movsdRR   %xmm0,%xmm3
        shrq    $48,%rax
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fsd_sincos_small)
        movsdRR   %xmm0,%xmm4
        mulsd   %xmm0,%xmm0
        mulsd   %xmm2,%xmm2
        mulsd   %xmm4,%xmm4

        mulsd   .L__dble_dsin_c6(%rip),%xmm0    /* x2 * s6 */
        mulsd   .L__dble_dcos_c6(%rip),%xmm2    /* x2 * c6 */
        addsd   .L__dble_dsin_c5(%rip),%xmm0    /* + s5 */
        addsd   .L__dble_dcos_c5(%rip),%xmm2    /* + c5 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s5 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dsin_c4(%rip),%xmm0    /* + s4 */
        addsd   .L__dble_dcos_c4(%rip),%xmm2    /* + c4 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s4 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + s3 */
        addsd   .L__dble_dcos_c3(%rip),%xmm2    /* + c3 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s3 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + s2 */
        addsd   .L__dble_dcos_c2(%rip),%xmm2    /* + c2 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s2 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c2 + ...) */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + s1 */
        addsd   .L__dble_dcos_c1(%rip),%xmm2    /* + c1 */
        mulsd   %xmm4,%xmm0                     /* x3 * (s1 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c1 + ...) */
        mulsd   %xmm3,%xmm0                     /* x3 */
        addsd   .L__dble_pq1+16(%rip),%xmm2     /* - 0.5 */
        mulsd   %xmm4,%xmm2                     /* x2 * (0.5 + ...) */
        addsd   %xmm3,%xmm0                     /* x + x3 * (...) done */
        addsd   %xmm2,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
        ret

LBL(.L__fsd_sincos_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fsd_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_pq1(%rip),%xmm3
        mulsd   %xmm2,%xmm3
        mulsd   .L__dble_pq1+16(%rip),%xmm2
        addsd   %xmm3,%xmm0
        addsd   %xmm2,%xmm1
        ret

LBL(.L__fsd_sincos_done1):
	rep
        ret

        ELF_FUNC(ENT_GH(__fmth_i_dsincos))
        ELF_SIZE(ENT_GH(__fmth_i_dsincos))
        IF_GH(ELF_FUNC(__fsd_sincos))
        IF_GH(ELF_SIZE(__fsd_sincos))


/* ------------------------------------------------------------------------- */

/* ============================================================
 * 
 *  A vector implementation of the single precision SINCOS() function.
 *
 *  __fvs_sincos(float)
 * 
 *  Entry:
 *	(%xmm0-ps)	Angle
 *
 *  Exit:
 *	(%xmm0-ps)	SIN(angle)
 *	(%xmm1-ps)	COS(angle)
 * 
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fvs_sincos))
        .globl ENT_GH(__fvssincos)
IF_GH(ENT(__fvs_sincos):)
ENT_GH(__fvssincos):
	movaps	%xmm0, %xmm1		/* Move input vector */
        andps   .L__sngl_mask_unsign(%rip), %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

        movlps  .L__sngl_pi_over_fours(%rip),%xmm2
        movhps  .L__sngl_pi_over_fours(%rip),%xmm2
        movlps  .L__sngl_needs_argreds(%rip),%xmm3
        movhps  .L__sngl_needs_argreds(%rip),%xmm3
        movlps  .L__sngl_sixteen_by_pi(%rip),%xmm4
        movhps  .L__sngl_sixteen_by_pi(%rip),%xmm4

	cmpps   $5, %xmm0, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	cmpps   $2, %xmm0, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        movmskps %xmm2, %eax
        movmskps %xmm3, %ecx

	test	$15, %eax
        jnz	LBL(.L__Scalar_fvsincos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulps   %xmm1,%xmm4 

	test	$15, %ecx
        jnz	LBL(.L__Scalar_fvsincos2)

        /* Set n = nearest integer to r */
	movhps	%xmm1,(%rsp)                     /* Store x4, x3 */

#if defined(_WIN64)
        movdqa  %xmm6, 16(%rsp)
        movdqa  %xmm7, 32(%rsp)
#endif
	xorq	%r10, %r10
	cvtps2pd %xmm1, %xmm6
	xorps	%xmm7, %xmm7

        cvtps2dq %xmm4,%xmm5    /* convert to integer, n4,n3,n2,n1 */

LBL(.L__fvsincos_do_twice):
#ifdef GH_TARGET
         movddup   .L__dble_pi_by_16_ms(%rip), %xmm0
         movddup   .L__dble_pi_by_16_ls(%rip), %xmm2
         movddup   .L__dble_pi_by_16_us(%rip), %xmm3
#else
        movlpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movhpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movlpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movhpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movlpd   .L__dble_pi_by_16_us(%rip), %xmm3
        movhpd   .L__dble_pi_by_16_us(%rip), %xmm3
#endif

        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - (n*p2 + n*p3) */
        mulpd   %xmm4,%xmm0     /* n * p1 */
        mulpd   %xmm4,%xmm2   /* n * p2 == rt */
        mulpd   %xmm4,%xmm3   /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        subpd   %xmm0,%xmm6   /* x - n * p1 == rh */
	addpd   %xmm2,%xmm3 

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        subpd   %xmm3,%xmm6   /* c = rh - rt aka r */

        shrq    $32, %r9
        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

	movapd  %xmm6,%xmm0   /* r in xmm0 and xmm6 */
        movapd  %xmm6,%xmm2   /* r in xmm2 */
        movapd  %xmm6,%xmm4   /* r in xmm4 */
        mulpd   %xmm6,%xmm6   /* r^2 in xmm6 */
        mulpd   %xmm0,%xmm0   /* r^2 in xmm0 */
        mulpd   %xmm4,%xmm4   /* r^2 in xmm4 */
        movapd  %xmm2,%xmm3   /* r in xmm2 and xmm3 */

        leaq    24(%r9),%r8   /* Add 24 for sine */
        andq    $0x1f,%r8     /* And lower 5 bits */
        andq    $0x1f,%r9     /* And lower 5 bits */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */

        /* xmm0, xmm4, xmm5 have r^2, xmm6, xmm2 has rr  */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulpd   .L__dble_pq3(%rip), %xmm0     /* p3 * r^2 */
        mulpd   .L__dble_pq3+16(%rip), %xmm6  /* q3 * r^2 */

        movq    %r8, %rdx     /* Duplicate it */
        sarq    $4,%r8        /* Sign bits moved down */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r8     /* Final tbl address */

        addpd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        addpd   .L__dble_pq2+16(%rip), %xmm6  /* + q2 */

        movq    %r9, %rdx     /* Duplicate it */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r9     /* Final tbl address */

        mulpd   %xmm4,%xmm0                   /* * r^2 */
        mulpd   %xmm4,%xmm6                   /* * r^2 */

        mulpd   %xmm4,%xmm3                   /* xmm3 = r^3 */
        addpd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addpd   .L__dble_pq1+16(%rip), %xmm6  /* + q1 */

        addq    %rax,%rax
        addq    %r8,%r8
        addq    %rcx,%rcx
        addq    %r9,%r9

	mulpd   %xmm3,%xmm0                   /* * r^3 */
	mulpd   %xmm4,%xmm6                   /* * r^2  = q(r) */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        movlpdMR  (%rdx,%rax,8),%xmm4           /* S in xmm4 */
        movhpd  (%rdx,%r8,8),%xmm4            /* S in xmm4 */

        movlpdMR  (%rdx,%rcx,8),%xmm3           /* C in xmm3 */
        movhpd  (%rdx,%r9,8),%xmm3            /* C in xmm3 */

	addpd   %xmm2,%xmm0                   /* + r = p(r) */
	movapd  %xmm6,%xmm1                   /* Move for cosine */
	movapd  %xmm0,%xmm2                   /* Move for sine */

	mulpd   %xmm3, %xmm0                  /* C * p(r) */
	mulpd   %xmm4, %xmm6                  /* S * q(r) */
	mulpd   %xmm3, %xmm1                  /* C * q(r) */
	mulpd   %xmm4, %xmm2                  /* S * p(r) */
	addpd   %xmm4, %xmm6                  /* S + S * q(r) */
	addpd   %xmm3, %xmm1                  /* C + C * q(r) */
	addpd   %xmm6, %xmm0                  /* sin(x) = Cp(r) + (S+Sq(r)) */
	subpd   %xmm2, %xmm1                  /* cos(x) = (C+Cq(r)) - Sp(r) */

	cvtpd2ps %xmm0,%xmm0
	cvtpd2ps %xmm1,%xmm1
	cmp	$0, %r10                      /* Compare loop count */
	shufps	$78, %xmm0, %xmm5             /* sin(x2), sin(x1), n4, n3 */
	shufps	$78, %xmm1, %xmm7             /* cos(x2), cos(x1), 0, 0 */
	jne 	LBL(.L__fvsincos_done_twice)
	inc 	%r10
	cvtps2pd (%rsp),%xmm6
	jmp 	LBL(.L__fvsincos_do_twice)

LBL(.L__fvsincos_done_twice):
	movaps  %xmm5, %xmm0
	movaps  %xmm7, %xmm1

#if defined(_WIN64)
        movdqa  16(%rsp), %xmm6
        movdqa  32(%rsp), %xmm7
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsincos1):
        /* Here when at least one argument is less than pi/4,
           or, at least one is a Nan.  What we will do for now, is
           if all are less than pi/4, do them all.  Otherwise, call
           fmth_i_sincos or mth_i_sincos one at a time.
        */
        movaps  %xmm0, (%rsp)                 /* Save xmm0, masked x */
	cmpps   $3, %xmm0, %xmm0              /* 3 is "unordered" */
        movaps  %xmm1, 16(%rsp)               /* Save xmm1, input x */
        movmskps %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

	cmp	$15, %eax
	jne	LBL(.L__Scalar_fvsincos1a)

	cvtps2pd 16(%rsp),%xmm0               /* x(2), x(1) */
	cvtps2pd 24(%rsp),%xmm1               /* x(4), x(3) */

#if defined(_WIN64)
        movdqa  %xmm6, 0(%rsp)
#endif
        movapd  %xmm0,16(%rsp)
        movapd  %xmm1,32(%rsp)
	mulpd   %xmm0,%xmm0                   /* x2 for x(2), x(1) */
	mulpd   %xmm1,%xmm1                   /* x2 for x(4), x(3) */

#ifdef GH_TARGET
         movddup  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
         movddup  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */
#else
        movlpd  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
        movhpd  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
        movlpd  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */
        movhpd  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */
#endif

        movapd  %xmm0,%xmm2
        movapd  %xmm1,%xmm3
        mulpd   %xmm4,%xmm0                   /* x2 * c4 */
        mulpd   %xmm4,%xmm1                   /* x2 * c4 */
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
#else
        movlpd  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
        movhpd  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
#endif

        addpd   %xmm5,%xmm0                   /* + c3 */
        addpd   %xmm5,%xmm1                   /* + c3 */
        movapd  .L__dble_pq1(%rip),%xmm5      /* c1 */
        mulpd   %xmm2,%xmm0                   /* x2 * (c3 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c3 + ...) */

        addpd   %xmm4,%xmm0                   /* + c2 */
        addpd   %xmm4,%xmm1                   /* + c2 */
	movapd  16(%rsp), %xmm6               /* x */
	movapd  32(%rsp), %xmm4               /* x */

        mulpd   %xmm2,%xmm0                   /* x2 * (c2 + ...) */
        mulpd   %xmm3,%xmm1                   /* x2 * (c2 + ...) */
	mulpd   %xmm6,%xmm2                   /* x3 */
	mulpd   %xmm4,%xmm3                   /* x3 */
        addpd   %xmm5,%xmm0                   /* + c1 */
        addpd   %xmm5,%xmm1                   /* + c1 */
        mulpd   %xmm2,%xmm0                   /* x3 * (c1 + ...) */
        mulpd   %xmm3,%xmm1                   /* x3 * (c1 + ...) */

        addpd   %xmm6,%xmm0             /* x + x3 * (...) done */
        addpd   %xmm4,%xmm1             /* x + x3 * (...) done */
	mulpd   %xmm6,%xmm6                   /* x2 for x(2), x(1) */
	mulpd   %xmm4,%xmm4                   /* x2 for x(4), x(3) */

        cvtpd2ps %xmm0,%xmm0            /* sin(x2), sin(x1) */
        cvtpd2ps %xmm1,%xmm1            /* sin(x4), sin(x3) */
	shufps	$68, %xmm1, %xmm0       /* sin(x4),sin(x3),sin(x2),sin(x1) */

#ifdef GH_TARGET
         movddup  .L__dble_dcos_c4(%rip),%xmm1  /* c4 */
         movddup  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */
#else
        movlpd  .L__dble_dcos_c4(%rip),%xmm1  /* c4 */
        movhpd  .L__dble_dcos_c4(%rip),%xmm1  /* c4 */
        movlpd  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */
        movhpd  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */
#endif

        movapd  %xmm6,%xmm2
        movapd  %xmm4,%xmm3
        mulpd   %xmm1,%xmm6                   /* x2 * c4 */
        mulpd   %xmm1,%xmm4                   /* x2 * c4 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c2(%rip),%xmm1  /* c2 */
#else
        movlpd  .L__dble_dcos_c2(%rip),%xmm1  /* c2 */
        movhpd  .L__dble_dcos_c2(%rip),%xmm1  /* c2 */
#endif

        addpd   %xmm5,%xmm6                   /* + c3 */
        addpd   %xmm5,%xmm4                   /* + c3 */
#ifdef GH_TARGET
         movddup  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */
#else
        movlpd  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */
        movhpd  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */
#endif

        mulpd   %xmm2,%xmm6                   /* x2 * (c3 + ...) */
        mulpd   %xmm3,%xmm4                   /* x2 * (c3 + ...) */

        addpd   %xmm1,%xmm6                   /* + c2 */
        addpd   %xmm1,%xmm4                   /* + c2 */
        movapd  .L__dble_pq1+16(%rip),%xmm1   /* -0.5 */
        mulpd   %xmm2,%xmm6                   /* x2 * (c2 + ...) */
        mulpd   %xmm3,%xmm4                   /* x2 * (c2 + ...) */
        addpd   %xmm5,%xmm6                   /* + c1 */
        addpd   %xmm5,%xmm4                   /* + c1 */
        movapd  .L__real_one(%rip), %xmm5     /* 1.0 */
        mulpd   %xmm2,%xmm6                   /* x2 * (c1 + ...) */
        mulpd   %xmm3,%xmm4                   /* x2 * (c1 + ...) */
        addpd   %xmm1,%xmm6                   /* -0.5 */
        addpd   %xmm1,%xmm4                   /* -0.5 */
        mulpd   %xmm2,%xmm6                   /* - x2 * (0.5 + ...) */
        mulpd   %xmm3,%xmm4                   /* - x2 * (0.5 + ...) */
        addpd   %xmm5,%xmm6                   /* 1.0 - 0.5x2 + (...) done */
        addpd   %xmm5,%xmm4                   /* 1.0 - 0.5x2 + (...) done */
        cvtpd2ps %xmm6,%xmm1            /* cos(x2), cos(x1) */
        cvtpd2ps %xmm4,%xmm4            /* cos(x4), cos(x3) */
        shufps  $68, %xmm4, %xmm1       /* cos(x4),cos(x3),cos(x2),cos(x1) */

#if defined(_WIN64)
        movdqa  (%rsp), %xmm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsincos1a):
	test    $1, %eax
	jz	LBL(.L__Scalar_fvsincos3)
#ifdef GH_TARGET
	movss 16(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 16(%rsp),%xmm0               /* dble x(1) */
#endif
	movl	(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos5)

LBL(.L__Scalar_fvsincos2):
        movaps  %xmm0, (%rsp)                 /* Save xmm0 */
        movaps  %xmm1, %xmm0                  /* Save xmm1 */
        movaps  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvsincos3):
	movss   16(%rsp),%xmm0                /* x(1) */
	test    $1, %ecx
	jz	LBL(.L__Scalar_fvsincos4)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_sincos))             /* Here when big or a nan */
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvsincos5)

LBL(.L__Scalar_fvsincos4):
	mov     %eax, 32(%rsp)                /* Here when a scalar will do */
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fss_sincos))
#else
	CALL(ENT(__fmth_i_sincos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos5):
        movss   %xmm0, (%rsp)                 /* Move first result */
        movss   %xmm1, 16(%rsp)               /* Move first result */

	test    $2, %eax
	jz	LBL(.L__Scalar_fvsincos6)
#ifdef GH_TARGET
	movss 20(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 20(%rsp),%xmm0               /* dble x(2) */
#endif
	movl	4(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos8)

LBL(.L__Scalar_fvsincos6):
	movss   20(%rsp),%xmm0                /* x(2) */
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvsincos7)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_sincos))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvsincos8)

LBL(.L__Scalar_fvsincos7):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fss_sincos))
#else
	CALL(ENT(__fmth_i_sincos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos8):
        movss   %xmm0, 4(%rsp)               /* Move 2nd result */
        movss   %xmm1, 20(%rsp)              /* Move 2nd result */

	test    $4, %eax
	jz	LBL(.L__Scalar_fvsincos9)
#ifdef GH_TARGET
	movss 24(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 24(%rsp),%xmm0               /* dble x(3) */
#endif
	movl	8(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos11)

LBL(.L__Scalar_fvsincos9):
	movss   24(%rsp),%xmm0                /* x(3) */
	test    $4, %ecx
	jz	LBL(.L__Scalar_fvsincos10)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_sincos))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvsincos11)

LBL(.L__Scalar_fvsincos10):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fss_sincos))
#else
	CALL(ENT(__fmth_i_sincos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos11):
        movss   %xmm0, 8(%rsp)               /* Move 3rd result */
        movss   %xmm1, 24(%rsp)              /* Move 3rd result */

	test    $8, %eax
	jz	LBL(.L__Scalar_fvsincos12)
#ifdef GH_TARGET
	movss 28(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd 28(%rsp),%xmm0               /* dble x(4) */
#endif
	movl	12(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos14)

LBL(.L__Scalar_fvsincos12):
	movss   28(%rsp),%xmm0                /* x(4) */
	test    $8, %ecx
	jz	LBL(.L__Scalar_fvsincos13)
	CALL(ENT(__mth_i_sincos))
	jmp	LBL(.L__Scalar_fvsincos14)

LBL(.L__Scalar_fvsincos13):
#ifdef GH_TARGET
	CALL(ENT(__fss_sincos))
#else
	CALL(ENT(__fmth_i_sincos))
#endif

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos14):
        movss   %xmm0, 12(%rsp)               /* Move 4th result */
        movss   %xmm1, 28(%rsp)               /* Move 4th result */
	movaps	(%rsp), %xmm0
	movaps	16(%rsp), %xmm1
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvs_sincos_local):
	movlpdMR  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        movsdRR   %xmm0,%xmm2
        movsdRR   %xmm0,%xmm3
        shrl    $20,%edx
        cmpl    $0x0390,%edx
        jl      LBL(.L__fss_sincos_small)
        movsdRR   %xmm0,%xmm4
        mulsd   %xmm0,%xmm0
        mulsd   %xmm2,%xmm2
        mulsd   %xmm4,%xmm4
        mulsd   .L__dble_dsin_c4(%rip),%xmm0    /* x2 * s4 */
        mulsd   .L__dble_dcos_c4(%rip),%xmm2    /* x2 * c4 */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + s3 */
        addsd   .L__dble_dcos_c3(%rip),%xmm2    /* + c3 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s3 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + 22 */
        addsd   .L__dble_dcos_c2(%rip),%xmm2    /* + c2 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s2 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c2 + ...) */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + s1 */
        addsd   .L__dble_dcos_c1(%rip),%xmm2    /* + c1 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s1 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c1 + ...) */
        mulsd   %xmm3,%xmm0                     /* x3 * (s1 + ...) */
        addsd   .L__dble_pq1+16(%rip),%xmm2     /* - 0.5 */
        mulsd   %xmm4,%xmm2                     /* x2 * (0.5 + ...) */
        addsd   %xmm3,%xmm0                     /* x + x3 * (...) done */
        addsd   %xmm2,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
	shufpd  $0, %xmm1, %xmm0

LBL(.L__fvs_sincos_done1):
	cvtpd2ps %xmm0,%xmm0                 /* Try to do 2 converts at once */
	movaps   %xmm0, %xmm1
	shufps  $1, %xmm1, %xmm1             /* xmm1 now has cos(x) */
        ret

LBL(.L__fvs_sincos_small):
        cmpl    $0x0320,%edx
	shufpd  $0, %xmm1, %xmm0
        jl      LBL(.L__fvs_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_pq1(%rip),%xmm3
        mulsd   %xmm2,%xmm3
        mulsd   .L__dble_pq1+16(%rip),%xmm2
        addsd   %xmm3,%xmm0
        addsd   %xmm2,%xmm1
	shufpd  $0, %xmm1, %xmm0
	jmp 	LBL(.L__fvs_sincos_done1)

        ELF_FUNC(ENT_GH(__fvssincos))
        ELF_SIZE(ENT_GH(__fvssincos))
        IF_GH(ELF_FUNC(__fvs_sincos))
        IF_GH(ELF_SIZE(__fvs_sincos))


/* ============================================================
 * 
 *  A vector implementation of the double precision SINCOS() function.
 *
 *  __fvd_sincos(double)
 * 
 *  Entry:
 *	(%xmm0-pd)	Angle
 *
 *  Exit:
 *	(%xmm0-pd)	SIN(angle)
 *	(%xmm1-pd)	COS(angle)
 * 
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status. 
 */

        .text
        ALN_FUNC
        IF_GH(.globl ENT(__fvd_sincos))
        .globl ENT_GH(__fvdsincos)
IF_GH(ENT(__fvd_sincos):)
ENT_GH(__fvdsincos):
	movapd	%xmm0, %xmm1		/* Move input vector */
        andpd   .L__real_mask_unsign(%rip), %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

#ifdef GH_TARGET
         movddup  .L__dble_pi_over_fours(%rip),%xmm2
         movddup  .L__dble_needs_argreds(%rip),%xmm3
         movddup  .L__dble_sixteen_by_pi(%rip),%xmm4
#else
        movlpd  .L__dble_pi_over_fours(%rip),%xmm2
        movhpd  .L__dble_pi_over_fours(%rip),%xmm2
        movlpd  .L__dble_needs_argreds(%rip),%xmm3
        movhpd  .L__dble_needs_argreds(%rip),%xmm3
        movlpd  .L__dble_sixteen_by_pi(%rip),%xmm4
        movhpd  .L__dble_sixteen_by_pi(%rip),%xmm4
#endif


	cmppd   $5, %xmm0, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	cmppd   $2, %xmm0, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        movmskpd %xmm2, %eax
        movmskpd %xmm3, %ecx

	test	$3, %eax
        jnz	LBL(.L__Scalar_fvdsincos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        mulpd   %xmm1,%xmm4 

	test	$3, %ecx
        jnz	LBL(.L__Scalar_fvdsincos2)

#if defined(_WIN64)
        movdqa  %xmm6, (%rsp)
        movdqa  %xmm7, 16(%rsp)
        movdqa  %xmm8, 32(%rsp)
#endif

        /* Set n = nearest integer to r */
        cvtpd2dq %xmm4,%xmm5    /* convert to integer */
#ifdef GH_TARGET
         movddup   .L__dble_pi_by_16_ms(%rip), %xmm0
         movddup   .L__dble_pi_by_16_ls(%rip), %xmm2
         movddup   .L__dble_pi_by_16_us(%rip), %xmm3
#else
        movlpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movhpd   .L__dble_pi_by_16_ms(%rip), %xmm0
        movlpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movhpd   .L__dble_pi_by_16_ls(%rip), %xmm2
        movlpd   .L__dble_pi_by_16_us(%rip), %xmm3
        movhpd   .L__dble_pi_by_16_us(%rip), %xmm3
#endif

        cvtdq2pd %xmm5,%xmm4    /* and back to double */

        movd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        mulpd   %xmm4,%xmm0     /* n * p1 */
        mulpd   %xmm4,%xmm2   /* n * p2 == rt */
        mulpd   %xmm4,%xmm3   /* n * p3 */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */

        /* How to convert N into a table address */
        movapd  %xmm1,%xmm6   /* x in xmm6 */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        subpd   %xmm0,%xmm1   /* x - n * p1 == rh */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        subpd   %xmm0,%xmm6   /* x - n * p1 == rh == c */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        subpd   %xmm2,%xmm1   /* rh = rh - rt */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */
        subpd   %xmm1,%xmm6   /* (c - rh) */
        movq    %rax, %rdx    /* Duplicate it */
        movapd  %xmm1,%xmm0   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        movapd  %xmm1,%xmm4   /* Move rh */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        movapd  %xmm1,%xmm5   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        subpd   %xmm2,%xmm6   /* ((c - rh) - rt) */
        andq    $0xf,%rdx     /* And lower 5 bits */
        subpd   %xmm6,%xmm3   /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        addq    %rdx, %rax    /* Final tbl address */
        movapd  %xmm1,%xmm2   /* Move rh */
        shrq    $32, %r9
        subpd   %xmm3,%xmm0   /* c = rh - rt aka r */
        movq    %rcx, %rdx    /* Duplicate it */
        subpd   %xmm3,%xmm4   /* c = rh - rt aka r */
        sarq    $4,%rcx       /* Sign bits moved down */
        subpd   %xmm3,%xmm5   /* c = rh - rt aka r */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        subpd   %xmm0,%xmm1   /* (rh - c) */
        sarq    $4,%rcx       /* Sign bits moved down */
        mulpd   %xmm0,%xmm0   /* r^2 in xmm0 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        movapd  %xmm4,%xmm6   /* r in xmm6 */
        addq    %rdx, %rcx    /* Final tbl address */
        mulpd   %xmm4,%xmm4   /* r^2 in xmm4 */
        leaq    24(%r9),%r8   /* Add 24 for sine */
        movapd  %xmm5,%xmm7   /* r in xmm7 */
        andq    $0x1f,%r8     /* And lower 5 bits */
        mulpd   %xmm5,%xmm5   /* r^2 in xmm5 */
        andq    $0x1f,%r9     /* And lower 5 bits */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        mulpd   .L__dble_pq4(%rip), %xmm0     /* p4 * r^2 */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        subpd   %xmm6,%xmm2                   /* (rh - c) */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        mulpd   .L__dble_pq4+16(%rip), %xmm4  /* q4 * r^2 */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        subpd   %xmm3,%xmm1                   /* (rh - c) - rt aka rr */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */
        addpd   .L__dble_pq3(%rip), %xmm0     /* + p3 */
        movq    %r8, %rdx     /* Duplicate it */
        addpd   .L__dble_pq3+16(%rip), %xmm4  /* + q3 */
        sarq    $4,%r8        /* Sign bits moved down */
        subpd   %xmm3,%xmm2                   /* (rh - c) - rt aka rr */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        mulpd   %xmm5,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        sarq    $4,%r8        /* Sign bits moved down */
        mulpd   %xmm5,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        mulpd   %xmm5,%xmm7                   /* xmm7 = r^3 */
        addq    %rdx, %r8     /* Final tbl address */
        movapd  %xmm1,%xmm3                   /* Move rr */
        movq    %r9, %rdx     /* Duplicate it */
        mulpd   %xmm5,%xmm1                   /* r * r * rr */
        sarq    $4,%r9        /* Sign bits moved down */
        addpd   .L__dble_pq2(%rip), %xmm0     /* + p2 */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        addpd   .L__dble_pq2+16(%rip), %xmm4  /* + q2 */
        sarq    $4,%r9        /* Sign bits moved down */
        mulpd   .L__dble_pq1+16(%rip), %xmm1  /* r * r * rr * 0.5 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        mulpd   %xmm6, %xmm3                  /* r * rr */
        addq    %rdx, %r9     /* Final tbl address */
        mulpd   %xmm5,%xmm0                   /* * r^2 */
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        mulpd   %xmm5,%xmm4                   /* * r^2 */
        addq    %rcx,%rcx
        addq    %r9,%r9
        addpd   %xmm1,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
        addq    %rax,%rax
        addq    %r8,%r8
        movlpdMR  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        movhpd  8(%rdx,%r9,8),%xmm1           /* dc2 in xmm1 */

        movlpdMR  8(%rdx,%rax,8),%xmm8          /* ds2 in xmm8 */
        movhpd  8(%rdx,%r8,8),%xmm8           /* ds2 in xmm8 */


        /* xmm0 has dp, xmm4 has dq,
           xmm1 has dc2
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3, xmm8 has ds2 */

        addpd   .L__dble_pq1(%rip), %xmm0     /* + p1 */
        addpd   .L__dble_pq1+16(%rip), %xmm4  /* + q1 */

        mulpd   %xmm7,%xmm0                   /* * r^3 */
        mulpd   %xmm5,%xmm4                   /* * r^2 == dq aka q(r) */

        addpd   %xmm2,%xmm0                   /* + cs  == dp aka p(r) */
        subpd   %xmm3,%xmm4                   /* - cc  == dq aka q(r) */

	movapd  %xmm1,%xmm3                   /* dc2 in xmm3 */
        movlpdMR  (%rdx,%rax,8),%xmm5           /* ds1 in xmm5 */
        movhpd  (%rdx,%r8,8),%xmm5            /* ds1 in xmm5 */

        movlpdMR  (%rdx,%rcx,8),%xmm7           /* dc1 in xmm7 */
        movhpd  (%rdx,%r9,8),%xmm7            /* dc1 in xmm7 */


        addpd   %xmm6,%xmm0                   /* + r   == dp aka p(r) */
        movapd  %xmm8,%xmm2                   /* ds2 in xmm2 */

        mulpd   %xmm4,%xmm8                   /* ds2 * dq */
        mulpd   %xmm4,%xmm1                   /* dc2 * dq */

        addpd   %xmm2,%xmm8                   /* ds2 + ds2*dq */
        addpd   %xmm3,%xmm1                   /* dc2 + dc2*dq */

        mulpd   %xmm0,%xmm3                   /* dc2 * dp */
        mulpd   %xmm0,%xmm2                   /* ds2 * dp */
        movapd  %xmm4,%xmm6                   /* xmm6 = dq */

        addpd   %xmm3,%xmm8                   /* (ds2 + ds2*dq) + dc2*dp */
        subpd   %xmm2,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */

        movapd  %xmm5,%xmm3                   /* xmm3 = ds1 */
        mulpd   %xmm5,%xmm4                   /* ds1 * dq */
        mulpd   %xmm0,%xmm5                   /* ds1 * dp */
        mulpd   %xmm7,%xmm6                   /* dc1 * dq */

        mulpd   %xmm7,%xmm0                   /* dc1 * dp */
        addpd   %xmm4,%xmm8                   /* ((ds2...) + dc2*dp) + ds1*dq */
        subpd   %xmm5,%xmm1                   /* (() - ds2*dp) - ds1*dp */

        addpd   %xmm3,%xmm8                   /* + ds1 */
        addpd   %xmm6,%xmm1                   /* + dc1*dq */

        addpd   %xmm8,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        addpd   %xmm7,%xmm1                   /* cos(x) = (C + Cq(r)) + Sq(r) */

#if defined(_WIN64)
        movdqa  (%rsp),%xmm6
        movdqa  16(%rsp),%xmm7
        movdqa  32(%rsp),%xmm8
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsincos1):
        movapd  %xmm0, (%rsp)                 /* Save xmm0 */
	cmppd   $3, %xmm0, %xmm0              /* 3 is "unordered" */
        movapd  %xmm1, 16(%rsp)               /* Save xmm1 */
        movmskpd %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

        movapd  16(%rsp), %xmm0
	test    $1, %eax
	jz	LBL(.L__Scalar_fvdsincos3)
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsincos1a)

#if defined(_WIN64)
        movdqa  %xmm6, (%rsp)
        movdqa  %xmm7, 16(%rsp)
#endif
        movapd  %xmm0,%xmm2
        mulpd   %xmm0,%xmm0
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c6(%rip),%xmm4    /* s6 */
         movddup  .L__dble_dcos_c6(%rip),%xmm5    /* c6 */
#else
        movlpd  .L__dble_dsin_c6(%rip),%xmm4    /* s6 */
        movhpd  .L__dble_dsin_c6(%rip),%xmm4    /* s6 */
        movlpd  .L__dble_dcos_c6(%rip),%xmm5    /* c6 */
        movhpd  .L__dble_dcos_c6(%rip),%xmm5    /* c6 */
#endif

        movapd  %xmm0,%xmm1
        movapd  %xmm0,%xmm3
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c5(%rip),%xmm6    /* s5 */
         movddup  .L__dble_dcos_c5(%rip),%xmm7    /* c5 */
#else
        movlpd  .L__dble_dsin_c5(%rip),%xmm6    /* s5 */
        movhpd  .L__dble_dsin_c5(%rip),%xmm6    /* s5 */
        movlpd  .L__dble_dcos_c5(%rip),%xmm7    /* c5 */
        movhpd  .L__dble_dcos_c5(%rip),%xmm7    /* c5 */
#endif

        mulpd   %xmm4,%xmm0                     /* x2 * s6 */
        mulpd   %xmm5,%xmm1                     /* x2 * c6 */

#ifdef GH_TARGET
         movddup  .L__dble_dsin_c4(%rip),%xmm4    /* s4 */
         movddup  .L__dble_dcos_c4(%rip),%xmm5    /* c4 */
#else
        movlpd  .L__dble_dsin_c4(%rip),%xmm4    /* s4 */
        movhpd  .L__dble_dsin_c4(%rip),%xmm4    /* s4 */
        movlpd  .L__dble_dcos_c4(%rip),%xmm5    /* c4 */
        movhpd  .L__dble_dcos_c4(%rip),%xmm5    /* c4 */
#endif

        addpd   %xmm6,%xmm0                     /* + s5 */
        addpd   %xmm7,%xmm1                     /* + c5 */
        mulpd   %xmm3,%xmm0                     /* x2 * (s5 + ...) */
        mulpd   %xmm3,%xmm1                     /* x2 * (c5 + ...) */

#ifdef GH_TARGET
         movddup  .L__dble_dsin_c3(%rip),%xmm6    /* s3 */
         movddup  .L__dble_dcos_c3(%rip),%xmm7    /* c3 */
#else
        movlpd  .L__dble_dsin_c3(%rip),%xmm6    /* s3 */
        movhpd  .L__dble_dsin_c3(%rip),%xmm6    /* s3 */
        movlpd  .L__dble_dcos_c3(%rip),%xmm7    /* c3 */
        movhpd  .L__dble_dcos_c3(%rip),%xmm7    /* c3 */
#endif

        addpd   %xmm4,%xmm0                     /* + s4 */
        addpd   %xmm5,%xmm1                     /* + c4 */
        mulpd   %xmm3,%xmm0                     /* x2 * (s4 + ...) */
        mulpd   %xmm3,%xmm1                     /* x2 * (c4 + ...) */
#ifdef GH_TARGET
         movddup  .L__dble_dsin_c2(%rip),%xmm4    /* s2 */
         movddup  .L__dble_dcos_c2(%rip),%xmm5    /* c2 */
#else
        movlpd  .L__dble_dsin_c2(%rip),%xmm4    /* s2 */
        movhpd  .L__dble_dsin_c2(%rip),%xmm4    /* s2 */
        movlpd  .L__dble_dcos_c2(%rip),%xmm5    /* c2 */
        movhpd  .L__dble_dcos_c2(%rip),%xmm5    /* c2 */
#endif

        addpd   %xmm6,%xmm0                     /* + s3 */
        addpd   %xmm7,%xmm1                     /* + c3 */
        mulpd   %xmm3,%xmm0                     /* x2 * (s3 + ...) */
        mulpd   %xmm3,%xmm1                     /* x2 * (c3 + ...) */
#ifdef GH_TARGET
         movddup  .L__dble_pq1(%rip),%xmm6        /* s1 */
         movddup  .L__dble_dcos_c1(%rip),%xmm7    /* c1 */
#else
        movlpd  .L__dble_pq1(%rip),%xmm6        /* s1 */
        movhpd  .L__dble_pq1(%rip),%xmm6        /* s1 */
        movlpd  .L__dble_dcos_c1(%rip),%xmm7    /* c1 */
        movhpd  .L__dble_dcos_c1(%rip),%xmm7    /* c1 */
#endif

        addpd   %xmm4,%xmm0                     /* + s2 */
        addpd   %xmm5,%xmm1                     /* + c2 */
        mulpd   %xmm3,%xmm0                     /* x2 * (s2 + ...) */
        mulpd   %xmm3,%xmm1                     /* x2 * (c2 + ...) */

        addpd   %xmm6,%xmm0                     /* + s1 */
        addpd   %xmm7,%xmm1                     /* + c1 */
        mulpd   %xmm3,%xmm0                     /* x3 * (s1 + ...) */
        mulpd   %xmm3,%xmm1                     /* x2 * (c1 + ...) */
        mulpd   %xmm2,%xmm0                     /* x3 * (s1 + ...) */
        addpd   .L__dble_pq1+16(%rip),%xmm1     /* - 0.5 */
        addpd   %xmm2,%xmm0                     /* x + x3 * (...) done */
        mulpd   %xmm3,%xmm1                     /* x2 * (0.5 + ...) */
        addpd   .L__real_one(%rip),%xmm1        /* 1.0 - 0.5x2 + (...) done */

#if defined(_WIN64)
        movdqa  (%rsp),%xmm6
        movdqa  16(%rsp),%xmm7
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsincos1a):
	movq	(%rsp),%rdx
	call	LBL(.L__fvd_sincos_local)
	jmp	LBL(.L__Scalar_fvdsincos5)

LBL(.L__Scalar_fvdsincos2):
        movapd  %xmm0, (%rsp)                 /* Save xmm0 */
        movapd  %xmm1, %xmm0                  /* Save xmm1 */
        movapd  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvdsincos3):
	test    $1, %ecx
	jz	LBL(.L__Scalar_fvdsincos4)
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
	CALL(ENT(__mth_i_dsincos))
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax
	jmp	LBL(.L__Scalar_fvdsincos5)

LBL(.L__Scalar_fvdsincos4):
	mov     %eax, 32(%rsp)
	mov     %ecx, 36(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fsd_sincos))
#else
	CALL(ENT(__fmth_i_dsincos))
#endif
	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

LBL(.L__Scalar_fvdsincos5):
        movlpd  %xmm0, (%rsp)
        movlpd  %xmm1, 16(%rsp)
        movlpdMR  24(%rsp), %xmm0
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsincos6)
	movq	8(%rsp),%rdx
	call	LBL(.L__fvd_sincos_local)
	jmp	LBL(.L__Scalar_fvdsincos8)

LBL(.L__Scalar_fvdsincos6):
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvdsincos7)
	CALL(ENT(__mth_i_dsincos))
	jmp	LBL(.L__Scalar_fvdsincos8)

LBL(.L__Scalar_fvdsincos7):
#ifdef GH_TARGET
	CALL(ENT(__fsd_sincos))
#else
	CALL(ENT(__fmth_i_dsincos))
#endif

LBL(.L__Scalar_fvdsincos8):
        movlpd  %xmm0, 8(%rsp)
        movlpd  %xmm1, 24(%rsp)
	movapd	(%rsp), %xmm0
	movapd	16(%rsp), %xmm1
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvd_sincos_local):
        movlpdMR  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        movsdRR   %xmm0,%xmm2
        movsdRR   %xmm0,%xmm3
        shrq    $48,%rdx
        cmpl    $0x03f20,%edx
        jl      LBL(.L__fvd_sincos_small)
        movsdRR   %xmm0,%xmm4
        mulsd   %xmm0,%xmm0                     /* x2 */
        mulsd   %xmm2,%xmm2                     /* x2 */
        mulsd   %xmm4,%xmm4                     /* x2 */
        mulsd   .L__dble_dsin_c6(%rip),%xmm0    /* x2 * s6 */
        mulsd   .L__dble_dcos_c6(%rip),%xmm2    /* x2 * c6 */
        addsd   .L__dble_dsin_c5(%rip),%xmm0    /* + s5 */
        addsd   .L__dble_dcos_c5(%rip),%xmm2    /* + c5 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s5 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c5 + ...) */
        addsd   .L__dble_dsin_c4(%rip),%xmm0    /* + s4 */
        addsd   .L__dble_dcos_c4(%rip),%xmm2    /* + c4 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s4 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c4 + ...) */
        addsd   .L__dble_dsin_c3(%rip),%xmm0    /* + s3 */
        addsd   .L__dble_dcos_c3(%rip),%xmm2    /* + c3 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s3 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c3 + ...) */
        addsd   .L__dble_dsin_c2(%rip),%xmm0    /* + s2 */
        addsd   .L__dble_dcos_c2(%rip),%xmm2    /* + c2 */
        mulsd   %xmm4,%xmm0                     /* x2 * (s2 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c2 + ...) */
        addsd   .L__dble_pq1(%rip),%xmm0        /* + s1 */
        addsd   .L__dble_dcos_c1(%rip),%xmm2    /* + c1 */
        mulsd   %xmm4,%xmm0                     /* x3 * (s1 + ...) */
        mulsd   %xmm4,%xmm2                     /* x2 * (c1 + ...) */
        mulsd   %xmm3,%xmm0                     /* x3 */
        addsd   .L__dble_pq1+16(%rip),%xmm2     /* - 0.5 */
        mulsd   %xmm4,%xmm2                     /* x2 * (0.5 + ...) */
        addsd   %xmm3,%xmm0                     /* x + x3 * (...) done */
        addsd   %xmm2,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
        ret

LBL(.L__fvd_sincos_small):
        cmpl    $0x03e40,%edx
        jl      LBL(.L__fvd_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        mulsd   %xmm2,%xmm2
        mulsd   .L__dble_pq1(%rip),%xmm3
        mulsd   %xmm2,%xmm3
        mulsd   .L__dble_pq1+16(%rip),%xmm2
        addsd   %xmm3,%xmm0
        addsd   %xmm2,%xmm1
        ret

LBL(.L__fvd_sincos_done1):
	rep
        ret

        ELF_FUNC(ENT_GH(__fvdsincos))
        ELF_SIZE(ENT_GH(__fvdsincos))
        IF_GH(ELF_FUNC(__fvd_sincos))
        IF_GH(ELF_SIZE(__fvd_sincos))


/* ========================================================================= */

	.text
	ALN_FUNC
	IF_GH(.globl ENT(__fss_pow))
	.globl ENT_GH(__fmth_i_rpowr)
IF_GH(ENT(__fss_pow):)
ENT_GH(__fmth_i_rpowr):

	movssRR	%xmm1, %xmm2
	movssRR	%xmm1, %xmm3
	movssRR	%xmm1, %xmm4
	shufps	$0, %xmm0, %xmm2
	shufps	$0, %xmm0, %xmm3
	shufps	$0, %xmm0, %xmm4
	movd	%xmm0, %eax
	movd	%xmm1, %ecx
	cmpps	$0, .L4_100(%rip), %xmm2
	cmpps	$0, .L4_101(%rip), %xmm3
	andps	.L4_102(%rip), %xmm4
	movdqa	%xmm4, %xmm5
	orps	%xmm3, %xmm2
	pcmpeqd	.L4_103(%rip), %xmm5
	orps	%xmm5, %xmm2
	movmskps %xmm2, %r8d
	test	$15, %r8d
	jnz	LBL(.L__Special_Pow_Cases)
	comiss	.L4_104(%rip), %xmm4
	ja	LBL(.L__Y_is_large)
	comiss	.L4_105(%rip), %xmm4
	jb	LBL(.L__Y_near_zero)
#ifdef GH_TARGET
	unpcklps %xmm1, %xmm1
	cvtps2pd %xmm1, %xmm1
#else
	cvtss2sd %xmm1, %xmm1
#endif
#ifdef GH_TARGET
	unpcklps %xmm0, %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd %xmm0, %xmm0
#endif
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$128, %rsp
	movsd	%xmm1, 0(%rsp)
#ifdef GH_TARGET
	CALL(ENT(__fsd_log))
#else
	CALL(ENT(__fmth_i_dlog))
#endif
	mulsd	0(%rsp), %xmm0
	CALL(ENT(__fss_exp_dbl))
	movq	%rbp, %rsp
	popq	%rbp
	ret

LBL(.L__Special_Pow_Cases):
	/* if x == 1.0, return 1.0 */
	cmp	.L4_101(%rip), %eax
	je	LBL(.L__Special_Case_1)

	/* if y == 1.5, return x * sqrt(x) */
	cmp	.L4_100+4(%rip), %ecx
	je	LBL(.L__Special_Case_2)

	/* if y == 0.5, return sqrt(x) */
	cmp	.L4_100(%rip), %ecx
	je	LBL(.L__Special_Case_3)

	/* if y == 0.25, return sqrt(sqrt(x)) */
	cmp	.L4_101+4(%rip), %ecx
	je	LBL(.L__Special_Case_4)

	/* if abs(y) == 0, return 1.0 */
	test	.L4_102(%rip), %ecx
	je	LBL(.L__Special_Case_5)

	/* if x == nan or inf, handle */
	mov	%eax, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_6)

LBL(.L__Special_Pow_Case_7):
	/* if y == nan or inf, handle */
	mov	%ecx, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_7)

LBL(.L__Special_Pow_Case_8):
	/* if y == 1.0, return x */
	cmp	.L4_101(%rip), %ecx
	jne	LBL(.L__Special_Pow_Case_9)
	rep
	ret

LBL(.L__Special_Pow_Case_9):
	/* If sign of x is 1, jump away */
	test	.L4_102+8(%rip), %eax
	jne	LBL(.L__Special_Pow_Case_10)
	/* x is 0.0, pos, or +inf */
	mov	%eax, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_9b)
LBL(.L__Special_Case_9a):
	/* x is 0.0, test sign of y */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_102+4(%rip), %eax
#ifdef FMATH_EXCEPTIONS
        mov     .L4_104(%rip), %edx
        cmovne  .L4_100+12(%rip), %edx
        movd   %edx, %xmm0
        divss  %xmm0, %xmm1     /* Generate divide by zero op when y < 0 */
#endif
	movd	%eax, %xmm0
	ret
LBL(.L__Special_Case_9b):
	/* x is +inf, test sign of y */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_100+12(%rip), %eax
	movd	%eax, %xmm0
	ret
	
LBL(.L__Special_Pow_Case_10):
	/* x is -0.0, neg, or -inf */
	/* Need to compute y is integer, even, odd, etc. */
	mov	%ecx, %r8d
	mov	%ecx, %r9d
	mov	$150, %r10d
	and	.L4_102+4(%rip), %r8d
	sar	$23, %r8d
	sub	%r8d, %r10d 	/* 150 - ((y && 0x7f8) >> 23) */
	jb	LBL(.L__Y_inty_2)
	cmp	$24, %r10d
	jae	LBL(.L__Y_inty_0)
	mov	$1, %edx
	mov	%r10d, %ecx
	shl	%cl, %edx
	mov	%edx, %r10d
	sub	$1, %edx
	test	%r9d, %edx
	jne	LBL(.L__Y_inty_0)
	test	%r9d, %r10d
	jne	LBL(.L__Y_inty_1)
LBL(.L__Y_inty_2):
	mov	$2, %r8d
	jmp	LBL(.L__Y_inty_decided)
LBL(.L__Y_inty_1):
	mov	$1, %r8d
	jmp	LBL(.L__Y_inty_decided)
LBL(.L__Y_inty_0):
	xor	%r8d, %r8d

LBL(.L__Y_inty_decided):
	mov	%r9d, %ecx
	mov	%eax, %edx
	and	.L4_102+4(%rip), %edx
	cmp	.L4_102+4(%rip), %edx
	je	LBL(.L__Special_Case_10c)
LBL(.L__Special_Case_10a):
	test	.L4_102(%rip), %eax
	jne	LBL(.L__Special_Case_10e)
	/* x is -0.0, test sign of y */
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10b)
	xor	%eax, %eax
	test	.L4_102+8(%rip), %ecx
	cmovne 	.L4_102+4(%rip), %eax
	movd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10b):
	test	.L4_102+8(%rip), %ecx
	cmovne 	.L4_108(%rip), %eax
	movd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10c):
	/* x is -inf, test sign of y */
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10d)
	/* x is -inf, inty != 1 */
	mov	.L4_102+4(%rip), %eax
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_100+12(%rip), %eax
	movd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10d):
	/* x is -inf, inty == 1 */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_102+8(%rip), %eax
	movd	%eax, %xmm0
	ret

LBL(.L__Special_Case_10e):
	/* x is negative */
	comiss	.L4_104(%rip), %xmm4
	ja	LBL(.L__Y_is_large)
	test	$3, %r8d
	je	LBL(.L__Special_Case_10f)
	and	.L4_102(%rip), %eax
	movd	%eax, %xmm0
#ifdef GH_TARGET
	unpcklps %xmm1, %xmm1
	cvtps2pd %xmm1, %xmm1
#else
	cvtss2sd %xmm1, %xmm1
#endif
#ifdef GH_TARGET
	unpcklps %xmm0, %xmm0
	cvtps2pd %xmm0, %xmm0
#else
	cvtss2sd %xmm0, %xmm0
#endif
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$128, %rsp
	movsd	%xmm1, 0(%rsp)
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10g)

#ifdef GH_TARGET
	CALL(ENT(__fsd_log))
#else
	CALL(ENT(__fmth_i_dlog))
#endif
	mulsd	0(%rsp), %xmm0
	CALL(ENT(__fss_exp_dbl))
	movq	%rbp, %rsp
	popq	%rbp
	ret

LBL(.L__Special_Case_10f):
/*
	and	.L4_101+12(%rip), %eax
	or 	.L4_102+12(%rip), %eax
*/
/* Changing this on Sept 13, 2005, to return 0xffc00000 for neg ** neg */
	mov 	.L4_108(%rip), %eax
	or 	.L4_107(%rip), %eax
#ifdef FMATH_EXCEPTIONS
	sqrtss	%xmm0, %xmm0
#endif
	movd	%eax, %xmm0
	ret

LBL(.L__Special_Case_10g):
#ifdef GH_TARGET
	CALL(ENT(__fsd_log))
#else
	CALL(ENT(__fmth_i_dlog))
#endif
	mulsd	0(%rsp), %xmm0
	CALL(ENT(__fss_exp_dbl))
	movssRR	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	subps	%xmm1, %xmm0
	movq	%rbp, %rsp
	popq	%rbp
	ret


LBL(.L__Special_Case_1):
LBL(.L__Special_Case_5):
	movss	.L4_101(%rip), %xmm0
	ret
LBL(.L__Special_Case_2):
	sqrtss	%xmm0, %xmm1
	mulss	%xmm1, %xmm0
	ret
LBL(.L__Special_Case_3):
	sqrtss	%xmm0, %xmm0
	ret
LBL(.L__Special_Case_4):
	sqrtss	%xmm0, %xmm0
	sqrtss	%xmm0, %xmm0
	ret
LBL(.L__Special_Case_6):
	test	.L4_106(%rip), %eax
	je	LBL(.L__Special_Pow_Case_7)
	or	.L4_107(%rip), %eax
	movd	%eax, %xmm0
	ret

LBL(.L__Special_Case_7):
	test	.L4_106(%rip), %ecx
	je	LBL(.L__Y_is_large)
	or	.L4_107(%rip), %ecx
	movd	%ecx, %xmm0
	ret

/* This takes care of all the large Y cases */
LBL(.L__Y_is_large):
	comiss	.L4_103(%rip), %xmm1
	andps	.L4_102(%rip), %xmm0
	jb	LBL(.L__Y_large_negative)
LBL(.L__Y_large_positive):
	/* If abs(x) < 1.0, return 0 */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return Inf */
	comiss	.L4_101(%rip), %xmm0
	jb	LBL(.L__Y_large_pos_0)
	je	LBL(.L__Y_large_pos_1)
LBL(.L__Y_large_pos_i):
	movss	.L4_102+4(%rip), %xmm0
	ret
LBL(.L__Y_large_pos_1):
	movss	.L4_101(%rip), %xmm0
	ret
/* */
LBL(.L__Y_large_negative):
	/* If abs(x) < 1.0, return Inf */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return 0 */
	comiss	.L4_101(%rip), %xmm0
	jb	LBL(.L__Y_large_pos_i)
	je	LBL(.L__Y_large_pos_1)
LBL(.L__Y_large_pos_0):
	movss	.L4_103(%rip), %xmm0
	ret

LBL(.L__Y_near_zero):
	movss	.L4_101(%rip), %xmm0
	ret


        ELF_FUNC(ENT_GH(__fmth_i_rpowr))
        ELF_SIZE(ENT_GH(__fmth_i_rpowr))
        IF_GH(ELF_FUNC(__fss_pow))
        IF_GH(ELF_SIZE(__fss_pow))


/* ========================================================================= */

	.text
	ALN_FUNC
	IF_GH(.globl ENT(__fsd_pow))
	.globl ENT_GH(__fmth_i_dpowd)
IF_GH(ENT(__fsd_pow):)
ENT_GH(__fmth_i_dpowd):

	pushq	%rbp
	movq	%rsp, %rbp
	subq	$32, %rsp

	/* Save y for mpy and broadcast for special case tests */
	movsd	%xmm1, 0(%rsp)
	movsd	%xmm0, 8(%rsp)

#if defined(_WIN64)
	movdqa	%xmm6, 16(%rsp)
#endif
	/* r8 holds flags for x, in rax */
	/* r9 holds flags for y, in rcx */
	/* rdx holds 1 */
        /* Use r10 and r11 for scratch */
	xor	%r8d, %r8d
	xor	%r9d, %r9d
	movl	$1, %edx
	movq	0(%rsp), %rcx
	movq	8(%rsp), %rax

	cmpq	.L4_D100(%rip), %rcx
	cmove	%edx, %r9d
	cmpq	.L4_D100+8(%rip), %rax
	cmove	%edx, %r8d

	cmpq	.L4_D101(%rip), %rcx
	cmove	%edx, %r9d
	cmpq	.L4_D101+8(%rip), %rax
	cmove	%edx, %r8d

	cmpq	.L4_D102(%rip), %rcx
	cmove	%edx, %r9d

	movsd	.L4_D105(%rip), %xmm4

	cmpq	.L4_D103(%rip), %rcx
	cmove	%edx, %r9d

	movq 	.L4_D104(%rip), %r10
	movq 	.L4_D104+8(%rip), %r11
	andq	%rcx, %r10
	andq	%rax, %r11
	cmpq	.L4_D104(%rip), %r10
	cmove	%edx, %r9d
	cmpq	.L4_D104+8(%rip), %r11
	cmove	%edx, %r8d

	andpd	%xmm1, %xmm4

	movq	.L4_D105(%rip), %r10
	movq	.L4_D105+8(%rip), %r11
	andq	%rcx, %r10
	andq	%rax, %r11
	cmpq	.L4_D106(%rip), %r10
	cmove	%edx, %r9d
	cmpq	.L4_D106+8(%rip), %r11
	cmove	%edx, %r8d

	/* Check special cases */
	or 	%r9d, %r8d
	jnz	LBL(.L__DSpecial_Pow_Cases)
	comisd	.L4_D107(%rip), %xmm4
	ja	LBL(.L__DY_is_large)
	comisd	.L4_D108(%rip), %xmm4
	jb	LBL(.L__DY_near_zero)

LBL(.L__D_algo_start):

	CALL(ENT(__fsd_log_long))

	/* Head in xmm0, tail in xmm1 */
	/* Carefully compute w = y * log(x) */

	/* Split y into hy (head) + ty (tail). */

        movsd   0(%rsp), %xmm2  			/* xmm2 has copy y */
        movsd   0(%rsp), %xmm5
        movsdRR   %xmm0, %xmm3

	andpd   .L__real_fffffffff8000000(%rip), %xmm2	/* xmm2 = head(y) */

	mulsd   0(%rsp), %xmm3				/* y * hx */
	subsd   %xmm2, %xmm5				/* ty */

        movsdRR   %xmm0, %xmm4
        mulsd   %xmm2, %xmm4				/* hy*hx */
        movsdRR   %xmm0, %xmm6
        mulsd   %xmm5, %xmm6				/* ty*hx */
        subsd   %xmm3, %xmm4				/* hy*hx - y*hx */
        mulsd   %xmm1, %xmm2				/* hy*tx */

        mulsd   %xmm5, %xmm1				/* ty*tx */

        addsd   %xmm6, %xmm4				/* + ty*hx */
        addsd   %xmm2, %xmm4				/* + hy*tx */
        addsd   %xmm4, %xmm1				/* + ty*tx */

        movsdRR   %xmm3, %xmm0
        addsd   %xmm1, %xmm0
        subsd   %xmm0, %xmm3
        addsd   %xmm3, %xmm1

	CALL(ENT(__fsd_exp_long))

LBL(.L__Dpop_and_return):
#if defined(_WIN64)
	movdqa	16(%rsp), %xmm6
#endif
	movq	%rbp, %rsp
	popq	%rbp
	ret

LBL(.L__DSpecial_Pow_Cases):
	/* if x == 1.0, return 1.0 */
	cmpq	.L4_D102(%rip), %rax
	je	LBL(.L__DSpecial_Case_1)

	/* if y == 1.5, return x * sqrt(x) */
	cmpq	.L4_D101(%rip), %rcx
	je	LBL(.L__DSpecial_Case_2)

	/* if y == 0.5, return sqrt(x) */
	cmpq	.L4_D100(%rip), %rcx
	je	LBL(.L__DSpecial_Case_3)

	/* if y == 0.25, return sqrt(sqrt(x)) */
	cmpq	.L4_D103(%rip), %rcx
	je	LBL(.L__DSpecial_Case_4)

	/* if abs(y) == 0, return 1.0 */
	testq	.L4_D105(%rip), %rcx
	je	LBL(.L__DSpecial_Case_5)

	/* if x == nan or inf, handle */
	movq	%rax, %rdx
	andq	.L4_D104(%rip), %rdx
	cmpq	.L4_D104(%rip), %rdx
	je	LBL(.L__DSpecial_Case_6)

LBL(.L__DSpecial_Pow_Case_7):
	/* if y == nan or inf, handle */
	movq	%rcx, %rdx
	andq	.L4_D104(%rip), %rdx
	cmpq	.L4_D104(%rip), %rdx
	je	LBL(.L__DSpecial_Case_7)

LBL(.L__DSpecial_Pow_Case_8):
	/* if y == 1.0, return x */
	cmpq	.L4_D102(%rip), %rcx
	je	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Pow_Case_9):
	/* If sign of x is 1, jump away */
	testq	.L4_D105+8(%rip), %rax
	jne	LBL(.L__DSpecial_Pow_Case_10)
	/* x is 0.0, pos, or +inf */
	movq	%rax, %rdx
	andq	.L4_D104(%rip), %rdx
	cmpq	.L4_D104(%rip), %rdx
	je	LBL(.L__DSpecial_Case_9b)
LBL(.L__DSpecial_Case_9a):
	/* x is 0.0, test sign of y */
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D104(%rip), %rax
#ifdef FMATH_EXCEPTIONS
        movq    .L4_D105(%rip), %rdx
        cmovneq .L4_D106(%rip), %rdx
        movq    %rdx, %xmm0
        divsd  %xmm0, %xmm1     /* Generate divide by zero op when y < 0 */
#endif
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_9b):
	/* x is +inf, test sign of y */
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D101+8(%rip), %rax
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)
	
LBL(.L__DSpecial_Pow_Case_10):
	/* x is -0.0, neg, or -inf */
	/* Need to compute y is integer, even, odd, etc. */
	/* rax = x, rcx = y */
	movq	%rcx, %r8
	movq	%rcx, %r9
	movq	$1075, %r10
	andq	.L4_D104(%rip), %r8
	sarq	$52, %r8
	subq	%r8, %r10 	/* 1075 - ((y && 0x7ff) >> 52) */
	jb	LBL(.L__DY_inty_2)
	cmpq	$53, %r10
	jae	LBL(.L__DY_inty_0)
	movq	$1, %rdx
	movq	%r10, %rcx
	shlq	%cl, %rdx
	movq	%rdx, %r10
	subq	$1, %rdx
	testq	%r9, %rdx
	jne	LBL(.L__DY_inty_0)
	testq	%r9, %r10
	jne	LBL(.L__DY_inty_1)
LBL(.L__DY_inty_2):
	movq	$2, %r8
	jmp	LBL(.L__DY_inty_decided)
LBL(.L__DY_inty_1):
	movq	$1, %r8
	jmp	LBL(.L__DY_inty_decided)
LBL(.L__DY_inty_0):
	xorq	%r8, %r8

LBL(.L__DY_inty_decided):
	movq	%r9, %rcx
	movq	%rax, %rdx
	andq	.L4_D104(%rip), %rdx
	cmpq	.L4_D104(%rip), %rdx
	je	LBL(.L__DSpecial_Case_10c)
LBL(.L__DSpecial_Case_10a):
	testq	.L4_D105(%rip), %rax
	jne	LBL(.L__DSpecial_Case_10e)
	/* x is -0.0, test sign of y */
	cmpq	$1, %r8
	je	LBL(.L__DSpecial_Case_10b)
	xorq	%rax, %rax
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D104(%rip), %rax
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10b):
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D109(%rip), %rax
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10c):
	/* x is -inf, test sign of y */
	cmpq	$1, %r8 
	je	LBL(.L__DSpecial_Case_10d)
	/* x is -inf, inty != 1 */
	movq	.L4_D104(%rip), %rax
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D101+8(%rip), %rax
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10d):
	/* x is -inf, inty == 1 */
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D105+8(%rip), %rax
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10e):
	/* x is negative */
	comisd	.L4_D107(%rip), %xmm4
	ja	LBL(.L__DY_is_large)
	testq	$3, %r8
	je	LBL(.L__DSpecial_Case_10f)
	andq	.L4_D105(%rip), %rax
	movd 	%rax, %xmm0

	/* Do the regular pow stuff */
	cmp 	$1, %r8d
	jne	LBL(.L__D_algo_start)

LBL(.L__DSpecial_Case_10g):
	CALL(ENT(__fsd_log_long))

	/* Head in xmm0, tail in xmm1 */
	/* Carefully compute w = y * log(x) */

	/* Split y into hy (head) + ty (tail). */

        movsd   0(%rsp), %xmm2  			/* xmm2 has copy y */
        movsd   0(%rsp), %xmm5
        movsdRR   %xmm0, %xmm3

	andpd   .L__real_fffffffff8000000(%rip), %xmm2	/* xmm2 = head(y) */

	mulsd   0(%rsp), %xmm3				/* y * hx */
	subsd   %xmm2, %xmm5				/* ty */

        movsdRR   %xmm0, %xmm4
        mulsd   %xmm2, %xmm4				/* hy*hx */
        movsdRR   %xmm0, %xmm6
        mulsd   %xmm5, %xmm6				/* ty*hx */
        subsd   %xmm3, %xmm4				/* hy*hx - y*hx */
        mulsd   %xmm1, %xmm2				/* hy*tx */

        mulsd   %xmm5, %xmm1				/* ty*tx */

        addsd   %xmm6, %xmm4				/* + ty*hx */
        addsd   %xmm2, %xmm4				/* + hy*tx */
        addsd   %xmm4, %xmm1				/* + ty*tx */

        movsdRR   %xmm3, %xmm0
        addsd   %xmm1, %xmm0
        subsd   %xmm0, %xmm3
        addsd   %xmm3, %xmm1

	CALL(ENT(__fsd_exp_long))

	mulsd	.L4_D102+8(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)

/* Changing this on Sept 13, 2005, to return 0xfff80000 for neg ** neg */
LBL(.L__DSpecial_Case_10f):
	movq 	.L4_D109(%rip), %rax
	orq	.L4_D10a(%rip), %rax
#ifdef FMATH_EXCEPTIONS
	sqrtsd	%xmm0, %xmm0	/* Generate an invalid op */
#endif
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_1):
LBL(.L__DSpecial_Case_5):
	movsd	.L4_D102(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_2):
	sqrtsd	%xmm0, %xmm1
	mulsd	%xmm1, %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_3):
	sqrtsd	%xmm0, %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_4):
	sqrtsd	%xmm0, %xmm0
	sqrtsd	%xmm0, %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_6):
	testq	.L4_D10b(%rip), %rax
	je	LBL(.L__DSpecial_Pow_Case_7)
	orq	.L4_D10a(%rip), %rax
	movd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_7):
	testq	.L4_D10b(%rip), %rcx
	je	LBL(.L__DY_is_large)
	orq	.L4_D10a(%rip), %rcx
	movd 	%rcx, %xmm0
	jmp	LBL(.L__Dpop_and_return)

/* This takes care of all the large Y cases */
LBL(.L__DY_is_large):
	comisd	.L4_D106(%rip), %xmm1
	andpd	.L4_D105(%rip), %xmm0
	jb	LBL(.L__DY_large_negative)
LBL(.L__DY_large_positive):
	/* If abs(x) < 1.0, return 0 */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return Inf */
	comisd	.L4_D102(%rip), %xmm0
	jb	LBL(.L__DY_large_pos_0)
	je	LBL(.L__DY_large_pos_1)
LBL(.L__DY_large_pos_i):
	movsd	.L4_D104(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DY_large_pos_1):
	movsd	.L4_D102(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)
/* */
LBL(.L__DY_large_negative):
	/* If abs(x) < 1.0, return Inf */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return 0 */
	comisd	.L4_D102(%rip), %xmm0
	jb	LBL(.L__DY_large_pos_i)
	je	LBL(.L__DY_large_pos_1)
LBL(.L__DY_large_pos_0):
	movsd	.L4_D106(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DY_near_zero):
	movsd	.L4_D102(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)

/* -------------------------------------------------------------------------- */


        ELF_FUNC(ENT_GH(__fmth_i_dpowd))
        ELF_SIZE(ENT_GH(__fmth_i_dpowd))
        IF_GH(ELF_FUNC(__fsd_pow))
        IF_GH(ELF_SIZE(__fsd_pow))


/* ============================================================ */

	.text
	ALN_FUNC
	IF_GH(.globl ENT(__fvs_pow))
	.globl ENT_GH(__fvspow)
IF_GH(ENT(__fvs_pow):)
ENT_GH(__fvspow):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
	movaps	%xmm0, %xmm4
	movaps	%xmm1, %xmm5
        movaps  %xmm0, %xmm2
	xorps	%xmm3, %xmm3
	andps	.L4_fvspow_infinity_mask(%rip), %xmm4
	andps	.L4_fvspow_infinity_mask(%rip), %xmm5
	cmpps	$2, %xmm3, %xmm2
	cmpps	$0, .L4_fvspow_infinity_mask(%rip), %xmm4
	cmpps	$0, .L4_fvspow_infinity_mask(%rip), %xmm5
	orps	%xmm4, %xmm2
	/* Store input arguments onto stack */
        movaps  %xmm0, _SX0(%rsp)
	orps	%xmm5, %xmm2
        movaps  %xmm1, _SY0(%rsp)
	movmskps %xmm2, %r8d
	test	$15, %r8d
	jnz	LBL(.L__Scalar_fvspow)

	/* Convert x0, x1 to dbl and call log */
	cvtps2pd %xmm0, %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fvd_log))
#else
        CALL(ENT(__fvdlog))
#endif

	/* dble(y) * dlog(x) */
        movlps  _SY0(%rsp), %xmm1
	cvtps2pd %xmm1, %xmm1
	mulpd	%xmm1, %xmm0
        movapd  %xmm0, _SR0(%rsp)

	/* Convert x2, x3 to dbl and call log */
        movlps  _SX2(%rsp), %xmm0
	cvtps2pd %xmm0, %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fvd_log))
#else
        CALL(ENT(__fvdlog))
#endif

	/* dble(y) * dlog(x) */
        movlps  _SY2(%rsp), %xmm1
	cvtps2pd %xmm1, %xmm1
	mulpd	%xmm0, %xmm1
	movapd	_SR0(%rsp), %xmm0

        CALL(ENT(__fvs_exp_dbl))

        movq    %rbp, %rsp
        popq    %rbp
	ret

LBL(.L__Scalar_fvspow):
#ifdef GH_TARGET
        CALL(ENT(__fss_pow))
#else
        CALL(ENT(__fmth_i_rpowr))
#endif
        movss   %xmm0, _SR0(%rsp)

        movss   _SX1(%rsp), %xmm0
        movss   _SY1(%rsp), %xmm1
#ifdef GH_TARGET
        CALL(ENT(__fss_pow))
#else
        CALL(ENT(__fmth_i_rpowr))
#endif
        movss   %xmm0, _SR1(%rsp)

        movss   _SX2(%rsp), %xmm0
        movss   _SY2(%rsp), %xmm1
#ifdef GH_TARGET
        CALL(ENT(__fss_pow))
#else
        CALL(ENT(__fmth_i_rpowr))
#endif
        movss   %xmm0, _SR2(%rsp)

        movss   _SX3(%rsp), %xmm0
        movss   _SY3(%rsp), %xmm1
#ifdef GH_TARGET
        CALL(ENT(__fss_pow))
#else
        CALL(ENT(__fmth_i_rpowr))
#endif
        movss   %xmm0, _SR3(%rsp)

        movaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ENT_GH(__fvspow))
        ELF_SIZE(ENT_GH(__fvspow))
        IF_GH(ELF_FUNC(__fvs_pow))
        IF_GH(ELF_SIZE(__fvs_pow))


/* ========================================================================= */
#define _DX0 0
#define _DX1 8

#define _DY0 16
#define _DY1 24

#define _DR0 32
#define _DR1 40

	.text
	ALN_FUNC
	IF_GH(.globl ENT(__fvd_pow))
	.globl ENT_GH(__fvdpow)
IF_GH(ENT(__fvd_pow):)
ENT_GH(__fvdpow):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
	movapd	%xmm0, %xmm4
	movapd	%xmm1, %xmm5
        movapd  %xmm0, %xmm2
	xorpd	%xmm3, %xmm3
	andpd	.L4_fvdpow_infinity_mask(%rip), %xmm4
	andpd	.L4_fvdpow_infinity_mask(%rip), %xmm5
	cmppd	$2, %xmm3, %xmm2
	cmppd	$0, .L4_fvdpow_infinity_mask(%rip), %xmm4
	cmppd	$0, .L4_fvdpow_infinity_mask(%rip), %xmm5
	orpd	%xmm4, %xmm2
	/* Store input arguments onto stack */
        movapd  %xmm0, _DX0(%rsp)
	orpd	%xmm5, %xmm2
        movapd  %xmm1, _DY0(%rsp)
	movmskpd %xmm2, %r8d
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdpow)

#if defined(_WIN64)
	movdqa	%xmm6, 48(%rsp)
#endif
	/* Call log long version */
        CALL(ENT(__fvd_log_long))

	/* Head in xmm0, tail in xmm1 */
	/* Carefully compute w = y * log(x) */

	/* Split y into hy (head) + ty (tail). */
        movapd  _DY0(%rsp), %xmm2  			/* xmm2 has copy y */
        movapd  _DY0(%rsp), %xmm5
        movapd	%xmm0, %xmm3

	andpd   .L__real_fffffffff8000000(%rip), %xmm2	/* xmm2 = head(y) */

	mulpd   _DY0(%rsp), %xmm3			/* y * hx */
	subpd   %xmm2, %xmm5				/* ty */

        movapd	%xmm0, %xmm4
        mulpd   %xmm2, %xmm4				/* hy*hx */
        movapd	%xmm0, %xmm6
        mulpd   %xmm5, %xmm6				/* ty*hx */
        subpd   %xmm3, %xmm4				/* hy*hx - y*hx */
        mulpd   %xmm1, %xmm2				/* hy*tx */

        mulpd   %xmm5, %xmm1				/* ty*tx */

        addpd   %xmm6, %xmm4				/* + ty*hx */
        addpd   %xmm2, %xmm4				/* + hy*tx */
        addpd   %xmm4, %xmm1				/* + ty*tx */

        movapd  %xmm3, %xmm0
        addpd   %xmm1, %xmm0
        subpd   %xmm0, %xmm3
        addpd   %xmm3, %xmm1

	CALL(ENT(__fvd_exp_long))

#if defined(_WIN64)
	movdqa	48(%rsp), %xmm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
	ret

LBL(.L__Scalar_fvdpow):
#ifdef GH_TARGET
        CALL(ENT(__fsd_pow))
#else
        CALL(ENT(__fmth_i_dpowd))
#endif
        movsd   %xmm0, _DR0(%rsp)

        movsd   _DX1(%rsp), %xmm0
        movsd   _DY1(%rsp), %xmm1
#ifdef GH_TARGET
        CALL(ENT(__fsd_pow))
#else
        CALL(ENT(__fmth_i_dpowd))
#endif
        movsd   %xmm0, _DR1(%rsp)

        movapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ENT_GH(__fvdpow))
        ELF_SIZE(ENT_GH(__fvdpow))
        IF_GH(ELF_FUNC(__fvd_pow))
        IF_GH(ELF_SIZE(__fvd_pow))


/*
 * This version uses SSE and one table lookup which pulls out 3 values,
 * for the polynomial approximation ax**2 + bx + c, where x has been reduced
 * to the range [ 1/sqrt(2), sqrt(2) ] and has had 1.0 subtracted from it.
 * The bulk of the answer comes from the taylor series
 *   log(x) = (x-1) - (x-1)**2/2 + (x-1)**3/3 - (x-1)**4/4
 * Method for argument reduction and result reconstruction is from
 * Cody & Waite.
 *
 * 5/04/04  B. Leback
 *
 */

/*
 *  float __fmth_i_alogx(float f)
 *
 *  Expects its argument f in %xmm0 instead of on the floating point
 *  stack, and also returns the result in %xmm0 instead of on the
 *  floating point stack.
 *
 *   stack usage:
 *   +---------+
 *   |   ret   | 12    <-- prev %esp
 *   +---------+
 *   |         | 8
 *   +--     --+
 *   |   lcl   | 4
 *   +--     --+
 *   |         | 0     <-- %esp  (8-byte aligned)
 *   +---------+
 *
 */

	.text
	ALN_FUNC
	IF_GH(.globl	ENT(__fss_log10))
	.globl	ENT_GH(__fmth_i_alog10)
IF_GH(ENT(__fss_log10):)
ENT_GH(__fmth_i_alog10):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(24)(%rsp)
#endif
	/* First check for valid input:
	 * if (a .gt. 0.0) then !!! Also check if not +infinity */

	/* Get started:
	 * ra = a
	 * ma = IAND(ia,'007fffff'x)
	 * ms = ma - '3504f3'x
	 * ig = IOR(ma,'3f000000'x)
	 * nx = ISHFT(ia,-23) - 126
	 * mx = IAND(ms,'00800000'x)
	 * ig = IOR(ig,mx)
	 * nx = nx - ISHFT(mx,-23)
         * ms = IAND(ms,'007f0000'x)
         * mt1 = ISHFT(ms,-16)
         * mt2 = ISHFT(ms,-15)
         * mt = mt1 + mt2 */

	movss	%xmm0, RZ_OFF(4)(%rsp)
        movss	.L4_384(%rip), %xmm2	/* Move smallest normalized number */
	movl	RZ_OFF(4)(%rsp), %ecx
	andl	$8388607, %ecx		/* ma = IAND(ia,'007fffff'x) */
	leaq 	-3474675(%rcx), %rdx	/* ms = ma - '3504f3'x */
	orl	$1056964608, %ecx	/* ig = IOR(ma,'3f000000'x) */
	cmpnless %xmm0, %xmm2		/* '00800000'x <= a, xmm2 1 where not */
        cmpeqss	.L4_387(%rip), %xmm0	/* Test for == +inf */
	movl	%edx, %eax		/* move ms */
	andl	$8388608, %edx		/* mx = IAND(ms,'00800000'x) */
	orl	%edx, %ecx		/* ig = IOR(ig,mx) */
	movl	%ecx, RZ_OFF(8)(%rsp)	/* move back over to fp sse */
	shrl	$23, %edx		/* ISHFT(mx,-23) */
        unpcklps %xmm2, %xmm0		/* Mask for nan, inf, neg and 0.0 */

	leaq	.L_STATICS1(%rip), %r8
	movl	RZ_OFF(4)(%rsp), %ecx	/* ia */
	andl	$8323072, %eax		/* ms = IAND(ms,'007f0000'x) */
	movss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	movmskps %xmm0, %r9d		/* move exception mask to gp reg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	movss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$126, %ecx		/* nx = ISHFT(ia,-23) - 126 */
	movss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
        shrl    $14, %eax		/* mt1 */
	and	$3, %r9d		/* mask with 3 */
	movss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
	jnz	LBL(.LB1_800_log10)

LBL(.LB1_100_log10):
	/* Do fp portion
         * xn = real(nx)
         * x0 = rg - 1.0
         * xsq = x0 * x0
         * xcu = xsq * x0
         * x1 = 0.5 * xsq
         * x3 = x1 * x1
         * x2 = thrd * xcu
         * rp = (COEFFS(mt) * x0 + COEFFS(mt+1)) * x0 + COEFFS(mt+2)
         * rz = rp - x3 + x2 - x1 + x0
         * rr = (xn * c1 + rz) + xn * c2 */

#ifdef GH_TARGET
	movd %ecx, %xmm0
	cvtdq2ps %xmm0, %xmm0
#else
	cvtsi2ss %ecx, %xmm0		/* xn */
#endif
	subss	.L4_386(%rip), %xmm1	/* x0 = rg - 1.0 */
	subss	.L4_386(%rip), %xmm6	/* x0 = rg - 1.0 */
	subss	.L4_386(%rip), %xmm4	/* x0 = rg - 1.0 */
	subss	.L4_386(%rip), %xmm2	/* x0 = rg - 1.0 */
	mulss	(%r8,%rax,4), %xmm1	/* COEFFS(mt) * x0 */
	mulss   %xmm6, %xmm6		/* xsq = x0 * x0 */
	addss	4(%r8,%rax,4), %xmm1	/* COEFFS(mt) * x0 + COEFFS(mt+1) */
	mulss   %xmm6, %xmm4		/* xcu = xsq * x0 */
	mulss   .L4_383(%rip), %xmm6	/* x1 = 0.5 * xsq */
	mulss   %xmm2, %xmm1		/* * x0 */
	mulss	12(%r8,%rax,4), %xmm4	/* x2 = thrd * xcu */
	movssRR	%xmm6, %xmm3		/* move x1 */
	mulss	%xmm6, %xmm6		/* x3 = x1 * x1 */
	addss	8(%r8,%rax,4), %xmm1	/* + COEFFS(mt+2) = rp */
	subss	%xmm6, %xmm1		/* rp - x3 */
	movss	.L4_396(%rip), %xmm5	/* Move c1 */
        movss   .L4_397(%rip), %xmm6	/* Move c2 */
	addss	%xmm1, %xmm4		/* rp - x3 + x2 */
	subss	%xmm3, %xmm4		/* rp - x3 + x2 - x1 */
	mulss	.L4_395(%rip), %xmm2
	mulss	.L4_395(%rip), %xmm4
	addss	%xmm2, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */

	mulss   %xmm0, %xmm5		/* xn * c1 */
	addss   %xmm5, %xmm4		/* (xn * c1 + rz) */
        mulss   %xmm6, %xmm0		/* xn * c2 */
        addss   %xmm4, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */

LBL(.LB1_900_log10):

#if defined(_WIN64)
	movdqa	RZ_OFF(24)(%rsp), %xmm6
#endif
	RZ_POP
	rep
	ret

	ALN_WORD
LBL(.LB1_800_log10):
	/* ir = 'ff800000'x */
	xorq	%rax,%rax
	movss	RZ_OFF(4)(%rsp), %xmm0
	movd 	%rax, %xmm1
	comiss	%xmm1, %xmm0
	jp	LBL(.LB1_cvt_nan)
#ifdef FMATH_EXCEPTIONS
        movss  .L4_386(%rip), %xmm1
        divss  %xmm0, %xmm1     /* Generate div-by-zero op when x=0 */
#endif
	movss	.L4_391(%rip),%xmm0	/* Move -inf */
	je	LBL(.LB1_900)
#ifdef FMATH_EXCEPTIONS
        sqrtss %xmm0, %xmm0     /* Generate invalid op for x<0 */
#endif
	movss	.L4_390(%rip),%xmm0	/* Move -nan */
	jb	LBL(.LB1_900)
	movss	.L4_387(%rip), %xmm0	/* Move +inf */
	movss	RZ_OFF(4)(%rsp), %xmm1
	comiss	%xmm1, %xmm0
	je	LBL(.LB1_900_log10)

	/* Otherwise, we had a denormal as an input */
	mulss	.L4_392(%rip), %xmm1	/* a * scale factor */
	movss	%xmm1, RZ_OFF(4)(%rsp)
	movl	RZ_OFF(4)(%rsp), %ecx
	andl	$8388607, %ecx		/* ma = IAND(ia,'007fffff'x) */
	leaq	-3474675(%rcx), %rdx	/* ms = ma - '3504f3'x */
	orl	$1056964608, %ecx	/* ig = IOR(ma,'3f000000'x) */
	movl	%edx, %eax		/* move ms */
	andl	$8388608, %edx		/* mx = IAND(ms,'00800000'x) */
	orl	%edx, %ecx		/* ig = IOR(ig,mx) */
	movl	%ecx, RZ_OFF(8)(%rsp)	/* move back over to fp sse */
	shrl	$23, %edx		/* ISHFT(mx,-23) */
	movl	RZ_OFF(4)(%rsp), %ecx	/* ia */
	andl	$8323072, %eax		/* ms = IAND(ms,'007f0000'x) */
	movss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	movss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$149, %ecx		/* nx = ISHFT(ia,-23) - (126 + 23) */
	movss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
	movss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
        shrl    $14, %eax		/* mt1 */
	jmp	LBL(.LB1_100_log10)

LBL(.LB1_cvt_nan_log10):
	movss	.L4_394(%rip), %xmm1	/* nan bit */
	orps	%xmm1, %xmm0
	jmp	LBL(.LB1_900_log10)

        ELF_FUNC(ENT_GH(__fmth_i_alog10))
        ELF_SIZE(ENT_GH(__fmth_i_alog10))
	IF_GH(ELF_FUNC(__fss_log10))
	IF_GH(ELF_SIZE(__fss_log10))


/*============================================================ */

	.text
	ALN_FUNC
        IF_GH(.globl  ENT(__fsd_log10))
        .globl  ENT_GH(__fmth_i_dlog10)
IF_GH(ENT(__fsd_log10):)
ENT_GH(__fmth_i_dlog10):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(24)(%rsp)
#endif
	/* Get input x into the range [0.5,1) */
	/* compute the index into the log tables */

	comisd	.L__real_mindp(%rip), %xmm0
	movdqa	%xmm0,%xmm3
	movsdRR	%xmm0,%xmm1
	jb	LBL(.L__z_or_n_dlog10)

	psrlq	$52,%xmm3
	subsd	.L__real_one(%rip),%xmm1
	psubq	.L__mask_1023(%rip),%xmm3
	cvtdq2pd %xmm3,%xmm6	/* xexp */

LBL(.L__100_dlog10):
	movdqa	%xmm0,%xmm3
	pand	.L__real_mant(%rip),%xmm3
	xorq	%r8,%r8
	movdqa	%xmm3,%xmm4
	movlpdMR	.L__real_half(%rip),%xmm5	/* .5 */
	/* Now  x = 2**xexp  * f,  1/2 <= f < 1. */
	psrlq	$45,%xmm3
	movdqa	%xmm3,%xmm2
	psrlq	$1,%xmm3
	paddq	.L__mask_040(%rip),%xmm3
	pand	.L__mask_001(%rip),%xmm2
	paddq	%xmm2,%xmm3

	andpd	.L__real_notsign(%rip),%xmm1
	comisd	.L__real_threshold(%rip),%xmm1
	cvtdq2pd %xmm3,%xmm1
	jb	LBL(.L__near_one_dlog10)
	movd	%xmm3,%r8d

	/* reduce and get u */
	por	.L__real_half(%rip),%xmm4
	movdqa	%xmm4,%xmm2

	mulsd	.L__real_3f80000000000000(%rip),%xmm1	/* f1 = index/128 */
	leaq	.L__np_ln_lead_table(%rip),%r9
	subsd	%xmm1,%xmm2				/* f2 = f - f1 */

	mulsd	%xmm2,%xmm5
	addsd	%xmm5,%xmm1

	divsd	%xmm1,%xmm2				/* u */

	/* Check for +inf */
	comisd	.L__real_inf(%rip),%xmm0
	je	LBL(.L__finish_dlog10)

	movlpdMR	-512(%r9,%r8,8),%xmm0 			/* z1 */
	/* solve for ln(1+u) */
	movsdRR	%xmm2,%xmm1				/* u */
	mulsd	%xmm2,%xmm2				/* u^2 */
	movsdRR	%xmm2,%xmm5
	movapd	.L__real_cb3(%rip),%xmm3
	mulsd	%xmm2,%xmm3				/* Cu2 */
	mulsd	%xmm1,%xmm5				/* u^3 */
	addsd	.L__real_cb2(%rip),%xmm3 		/* B+Cu2 */
	movapd	%xmm2,%xmm4
	mulsd	%xmm5,%xmm4				/* u^5 */
	movlpdMR	.L__real_log2_lead(%rip),%xmm2
	mulsd	.L__real_cb1(%rip),%xmm5 		/* Au3 */
	addsd	%xmm5,%xmm1				/* u+Au3 */
	mulsd	%xmm3,%xmm4				/* u5(B+Cu2) */
	movapd	%xmm0,%xmm3
	addsd	%xmm4,%xmm1				/* poly */

	/* recombine */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	addsd	-512(%rdx,%r8,8),%xmm1 			/* z2	+=q */
	mulsd	%xmm6,%xmm2				/* npi2 * log2_lead */
	mulsd	.L__real_log2_tail(%rip),%xmm6
	addsd	%xmm2,%xmm0				/* r1 */
	addsd	%xmm3,%xmm2				/* r1 */
	addsd	%xmm6,%xmm1				/* r2 */
	mulsd	.L__log10_multiplier1(%rip), %xmm0
	mulsd	.L__log10_multiplier2(%rip), %xmm2
	mulsd	.L__log10_multiplier(%rip), %xmm1
	addsd	%xmm2,%xmm1
LBL(.L__cvt_to_dlog10):
	addsd	%xmm1,%xmm0

LBL(.L__finish_dlog10):
#if defined(_WIN64)
	movdqa	RZ_OFF(24)(%rsp), %xmm6
#endif

	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.L__near_one_dlog10):
	/* saves 10 cycles */
	/* r = x - 1.0; */
	movlpdMR	.L__real_two(%rip),%xmm2
	subsd	.L__real_one(%rip),%xmm0

	/* u = r / (2.0 + r); */
	addsd	%xmm0,%xmm2
	movsdRR	%xmm0,%xmm1
	divsd	%xmm2,%xmm1
	movlpdMR	.L__real_ca4(%rip),%xmm4
	movlpdMR	.L__real_ca3(%rip),%xmm5
	/* correction = r * u; */
	movsdRR	%xmm0,%xmm6
	mulsd	%xmm1,%xmm6

	/* u = u + u; */
	addsd	%xmm1,%xmm1
	movsdRR	%xmm1,%xmm2
	mulsd	%xmm2,%xmm2
	/* r2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction); */
	mulsd	%xmm1,%xmm5
	movsdRR	%xmm1,%xmm3
	mulsd	%xmm2,%xmm3
	mulsd	.L__real_ca2(%rip),%xmm2
	mulsd	%xmm3,%xmm4

	addsd	.L__real_ca1(%rip),%xmm2
	movsdRR	%xmm3,%xmm1
	mulsd	%xmm1,%xmm1
	addsd	%xmm4,%xmm5

	movsdRR	%xmm0,%xmm4
	mulsd	%xmm3,%xmm2
	mulsd	%xmm5,%xmm1
	addsd	%xmm2,%xmm1
	subsd	%xmm6,%xmm1
	mulsd	.L__log10_multiplier1(%rip), %xmm0
	mulsd	.L__log10_multiplier2(%rip), %xmm4
	mulsd	.L__log10_multiplier(%rip), %xmm1
	addsd	%xmm4,%xmm1
	jmp	LBL(.L__cvt_to_dlog10)

	/* Start here for all the conditional cases */
	/* we have a zero, a negative number, denorm, or nan. */
LBL(.L__z_or_n_dlog10):
	jp	LBL(.L__lnan_dlog10)
	xorpd	%xmm1, %xmm1
	comisd	%xmm1, %xmm0
	je	LBL(.L__zero_dlog10)
	jbe	LBL(.L__negative_x_dlog10)

	/* A Denormal input, scale appropriately */
	mulsd	.L__real_scale(%rip), %xmm0
	movdqa	%xmm0, %xmm3
	movsdRR	%xmm0, %xmm1

	psrlq	$52,%xmm3
	subsd	.L__real_one(%rip),%xmm1
	psubq	.L__mask_1075(%rip),%xmm3
	cvtdq2pd %xmm3,%xmm6
	jmp	LBL(.L__100_dlog10)

	/* x == +/-0.0 */
LBL(.L__zero_dlog10):
#ifdef FMATH_EXCEPTIONS
        movsd  .L__real_one(%rip), %xmm1
        divsd  %xmm0, %xmm1 /* Generate divide-by-zero op */
#endif

	movlpdMR	.L__real_ninf(%rip),%xmm0  /* C99 specs -inf for +-0 */
	jmp	LBL(.L__finish_dlog10)

	/* x < 0.0 */
LBL(.L__negative_x_dlog10):
#ifdef FMATH_EXCEPTIONS
        sqrtsd %xmm0, %xmm0
#endif

	movlpdMR	.L__real_nan(%rip),%xmm0
	jmp	LBL(.L__finish_dlog10)

	/* NaN */
LBL(.L__lnan_dlog10):
	xorpd	%xmm1, %xmm1
	movlpdMR	.L__real_qnanbit(%rip), %xmm1	/* convert to quiet */
	orpd	%xmm1, %xmm0
	jmp	LBL(.L__finish_dlog10)

        ELF_FUNC(ENT_GH(__fmth_i_dlog10))
        ELF_SIZE(ENT_GH(__fmth_i_dlog10))
        IF_GH(ELF_FUNC(__fsd_log10))
        IF_GH(ELF_SIZE(__fsd_log10))


/* --------------------------------------------------------------------------
 *
 * This version uses SSE and one table lookup which pulls out 3 values,
 * for the polynomial approximation ax**2 + bx + c, where x has been reduced
 * to the range [ 1/sqrt(2), sqrt(2) ] and has had 1.0 subtracted from it.
 * The bulk of the answer comes from the taylor series
 *   log(x) = (x-1) - (x-1)**2/2 + (x-1)**3/3 - (x-1)**4/4
 * Method for argument reduction and result reconstruction is from
 * Cody & Waite.
 *
 * 5/15/04  B. Leback
 *
 */
	.text
	ALN_FUNC
	IF_GH(.globl ENT(__fvs_log10))
	.globl ENT_GH(__fvslog10)
IF_GH(ENT(__fvs_log10):)
ENT_GH(__fvslog10):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
	movdqa	%xmm7, RZ_OFF(72)(%rsp)
#endif

/* Fast vector natural logarithm code goes here... */
        /* First check for valid input:
         * if (a .gt. 0.0) then */
	movaps  .L4_384(%rip), %xmm4	/* Move min arg to xmm4 */
	xorps	%xmm7, %xmm7		/* Still need 0.0 */
	movaps	%xmm0, %xmm2		/* Move for nx */
	movaps	%xmm0, %xmm1		/* Move to xmm1 for later ma */

	/* Check exceptions and valid range */
	cmpleps	%xmm0, %xmm4		/* '00800000'x <= a, xmm4 1 where true */
	cmpltps	%xmm0, %xmm7		/* Test for 0.0 < a, xmm7 1 where true */
	cmpneqps .L4_387(%rip), %xmm0	/* Test for == +inf */
	xorps	%xmm7, %xmm4		/* xor to find just denormal inputs */
	movmskps %xmm4, %eax		/* Move denormal mask to gp ref */
	movaps	%xmm2, RZ_OFF(24)(%rsp)	/* Move for exception processing */
	movaps	.L4_382(%rip), %xmm3	/* Move 126 */
	cmp	$0, %eax		/* Test for denormals */
	jne	LBL(.LB_DENORMs_log10)

        /* Get started:
         * ra = a
         * ma = IAND(ia,'007fffff'x)
         * ms = ma - '3504f3'x
         * ig = IOR(ma,'3f000000'x)
         * nx = ISHFT(ia,-23) - 126
         * mx = IAND(ms,'00800000'x)
         * ig = IOR(ig,mx)
         * nx = nx - ISHFT(mx,-23)
         * ms = IAND(ms,'007f0000'x)
         * mt = ISHFT(ms,-12) */

LBL(.LB_100_log10):
	leaq	.L_STATICS1(%rip),%r8
	andps	.L4_380(%rip), %xmm1	/* ma = IAND(ia,'007fffff'x) */
	psrld	$23, %xmm2		/* nx = ISHFT(ia,-23) */
	andps	%xmm0, %xmm7		/* Mask for nan, inf, neg and 0.0 */
	movaps	%xmm1, %xmm6		/* move ma for ig */
	psubd	.L4_381(%rip), %xmm1	/* ms = ma - '3504f3'x */
	psubd	%xmm3, %xmm2		/* nx = ISHFT(ia,-23) - 126 */
	orps	.L4_383(%rip), %xmm6	/* ig = IOR(ma,'3f000000'x) */
	movaps	%xmm1, %xmm0		/* move ms for tbl ms */
	andps	.L4_384(%rip), %xmm1	/* mx = IAND(ms,'00800000'x) */
	andps	.L4_385(%rip), %xmm0	/* ms = IAND(ms,'007f0000'x) */
	orps	%xmm1, %xmm6		/* ig = IOR(ig, mx) */
	psrad	$23, %xmm1		/* ISHFT(mx,-23) */
	psrad	$12, %xmm0		/* ISHFT(ms,-12) for 128 bit reads */
	movmskps %xmm7, %eax		/* Move xmm7 mask to eax */
	psubd	%xmm1, %xmm2		/* nx = nx - ISHFT(mx,-23) */
	movaps	%xmm0, RZ_OFF(40)(%rsp)	/* Move to memory */
	cvtdq2ps  %xmm2, %xmm0		/* xn = real(nx) */

	movl	RZ_OFF(40)(%rsp), %ecx		/* Move to gp register */
	movaps	(%r8,%rcx,1), %xmm1		/* Read from 1st table location */
	movl	RZ_OFF(36)(%rsp), %edx		/* Move to gp register */
	movaps	(%r8,%rdx,1), %xmm2		/* Read from 2nd table location */
	movl	RZ_OFF(32)(%rsp), %ecx		/* Move to gp register */
	movaps	(%r8,%rcx,1), %xmm3		/* Read from 3rd table location */
	movl	RZ_OFF(28)(%rsp), %edx		/* Move to gp register */
	movaps	(%r8,%rdx,1), %xmm4		/* Read from 4th table location */

	/* So, we do 4 reads of a,b,c into registers xmm1, xmm2, xmm3, xmm4
	 * Assume we need to keep rg in xmm6, xn in xmm0
	 * The following shuffle gets them into SIMD mpy form:
	 */

	subps	.L4_386(%rip), %xmm6 	/* x0 = rg - 1.0 */

	movaps	%xmm1, %xmm5		/* Store 1/3, c0, b0, a0 */
	movaps	%xmm3, %xmm7		/* Store 1/3, c2, b2, a2 */

	unpcklps %xmm2, %xmm1		/* b1, b0, a1, a0 */
	unpcklps %xmm4, %xmm3		/* b3, b2, a3, a2 */
	unpckhps %xmm2, %xmm5		/* 1/3, 1/3, c1, c0 */
	unpckhps %xmm4, %xmm7		/* 1/3, 1/3, c3, c2 */

	movaps	%xmm6, %xmm4		/* move x0 */

	movaps	%xmm1, %xmm2		/* Store b1, b0, a1, a0 */
	movlhps	%xmm3, %xmm1		/* a3, a2, a1, a0 */
	movlhps	%xmm7, %xmm5		/* c3, c2, c1, c0 */
	movhlps	%xmm2, %xmm3		/* b3, b2, b1, b0 */

	mulps	%xmm6, %xmm1		/* COEFFS(mt) * x0 */
	mulps	%xmm6, %xmm6		/* xsq = x0 * x0 */
	movhlps	%xmm7, %xmm7		/* 1/3, 1/3, 1/3, 1/3 */

	movaps	%xmm4, %xmm2		/* move x0 */

        /* Do fp portion
         * xn = real(nx)
         * x0 = rg - 1.0
         * xsq = x0 * x0
         * xcu = xsq * x0
         * x1 = 0.5 * xsq
         * x3 = x1 * x1
         * x2 = thrd * xcu
         * rp = (COEFFS(mt) * x0 + COEFFS(mt+1)) * x0 + COEFFS(mt+2)
         * rz = rp - x3 + x2 - x1 + x0
         * rr = (xn * c1 + rz) + xn * c2 */

	/* Now do the packed coefficient multiply and adds */
	/* x4 has x0 */
	/* x6 has xsq */
	/* x7 has thrd * x0 */
	/* x1, x3, and x5 have a, b, c */
	/* x0 has xn */
	addps	%xmm3, %xmm1		/* COEFFS(mt) * g + COEFFS(mt+1) */
	mulps	%xmm6, %xmm4		/* xcu = xsq * x0 */
	mulps	.L4_383(%rip), %xmm6	/* x1 = 0.5 * xsq */
	mulps	%xmm2, %xmm1		/* * x0 */
	mulps	%xmm7, %xmm4		/* x2 = thrd * xcu */
	movaps	%xmm6, %xmm3		/* move x1 */
	mulps	%xmm6, %xmm6		/* x3 = x1 * x1 */
	addps	%xmm5, %xmm1		/* + COEFFS(mt+2) = rp */
	subps	%xmm6, %xmm1		/* rp - x3 */
	movaps	.L4_396(%rip), %xmm7	/* Move c1 */
        movaps  .L4_397(%rip), %xmm6	/* Move c2 */
	addps	%xmm1, %xmm4		/* rp - x3 + x2 */
	subps	%xmm3, %xmm4		/* rp - x3 + x2 - x1 */
	mulps	.L4_395(%rip),%xmm2     /* mpy for log10 */
	mulps	.L4_395(%rip),%xmm4     /* mpy for log10 */
	addps	%xmm2, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
	mulps   %xmm0, %xmm7		/* xn * c1 */
	addps   %xmm7, %xmm4		/* (xn * c1 + rz) */
        mulps   %xmm6, %xmm0		/* xn * c2 */
        addps   %xmm4, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */

	/* Compare exception mask now and jump if no exceptions */
	cmp	$15, %eax
	jne 	LBL(.LB_EXCEPTs_log10)

LBL(.LB_900_log10):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
	movdqa	RZ_OFF(72)(%rsp), %xmm7
#endif

	RZ_POP
	rep
	ret

LBL(.LB_EXCEPTs_log10):
        /* Handle all exceptions by masking in xmm */
        movaps  RZ_OFF(24)(%rsp), %xmm1	/* original input */
        movaps  RZ_OFF(24)(%rsp), %xmm2	/* original input */
        movaps  RZ_OFF(24)(%rsp), %xmm3	/* original input */
        xorps   %xmm7, %xmm7            /* xmm7 = 0.0 */
        xorps   %xmm6, %xmm6            /* xmm6 = 0.0 */
	movaps	.L4_394(%rip), %xmm5	/* convert nan bit */
        xorps   %xmm4, %xmm4            /* xmm4 = 0.0 */
                                                                                
        cmpunordps %xmm1, %xmm7         /* Test if unordered */
        cmpltps %xmm6, %xmm2            /* Test if a < 0.0 */
        cmpordps %xmm1, %xmm6           /* Test if ordered */
                                                                                
        andps   %xmm7, %xmm5            /* And nan bit where unordered */
        orps    %xmm7, %xmm4            /* Or all masks together */
        andps   %xmm1, %xmm7            /* And input where unordered */
	orps	%xmm5, %xmm7		/* Convert unordered nans */
                                                                                
        xorps   %xmm5, %xmm5            /* xmm5 = 0.0 */
        andps   %xmm2, %xmm6            /* Must be ordered and < 0.0 */
        orps    %xmm6, %xmm4            /* Or all masks together */
        andps   .L4_390(%rip), %xmm6    /* And -nan if < 0.0 and ordered */
                                                                                
        cmpeqps .L4_387(%rip), %xmm3    /* Test if equal to infinity */
        cmpeqps %xmm5, %xmm1            /* Test if eq 0.0 */
        orps    %xmm6, %xmm7            /* or in < 0.0 */
                                                                                
        orps    %xmm3, %xmm4            /* Or all masks together */
        andps   .L4_387(%rip), %xmm3    /* inf and inf mask */
        movaps  %xmm0, %xmm2
        orps    %xmm3, %xmm7            /* or in infinity */
                                                                                
        orps    %xmm1, %xmm4            /* Or all masks together */
        andps   .L4_391(%rip), %xmm1    /* And -inf if == 0.0 */
        movaps  %xmm4, %xmm0
        orps    %xmm1, %xmm7            /* or in -infinity */
                                                                                
        andnps  %xmm2, %xmm0            /* Where mask not set, use result */
        orps    %xmm7, %xmm0            /* or in exceptional values */
	jmp	LBL(.LB_900_log10)

LBL(.LB_DENORMs_log10):
	/* Have the denorm mask in xmm4, so use it to scale a and the subtractor */
	movaps	%xmm4, %xmm5		/* Move mask */
	movaps	%xmm4, %xmm6		/* Move mask */
	andps	.L4_392(%rip), %xmm4	/* Have 2**23 where denorms are, 0 else */
	andnps	%xmm1, %xmm5		/* Have a where denormals aren't */
	mulps	%xmm4, %xmm1		/* denormals * 2**23 */
	andps	.L4_393(%rip), %xmm6	/* have 23 where denorms are, 0 else */
	orps	%xmm5, %xmm1		/* Or in the original a */
	paddd	%xmm6, %xmm3		/* Add 23 to 126 for offseting exponent */
	movaps	%xmm1, %xmm2		/* Move to the next location */
	jmp	LBL(.LB_100_log10)

        ELF_FUNC(ENT_GH(__fvslog10))
        ELF_SIZE(ENT_GH(__fvslog10))
	IF_GH(ELF_FUNC(__fvs_log10))
	IF_GH(ELF_SIZE(__fvs_log10))


/* ======================================================================== */

/* Log10, at the urging of John Levesque */

    	.text
    	ALN_FUNC
	IF_GH(.globl ENT(__fvd_log10))
	.globl ENT_GH(__fvdlog10)
IF_GH(ENT(__fvd_log10):)
ENT_GH(__fvdlog10):
	RZ_PUSH

#if defined(_WIN64)
	movdqa	%xmm6, RZ_OFF(56)(%rsp)
#endif

	movdqa	%xmm0, RZ_OFF(40)(%rsp)	/* save the input values */
	movapd	%xmm0, %xmm2
	movapd	%xmm0, %xmm4
	pxor	%xmm1, %xmm1
	cmppd	$6, .L__real_maxfp(%rip), %xmm2
	cmppd 	$1, .L__real_mindp(%rip), %xmm4
	movdqa	%xmm0, %xmm3
	psrlq	$52, %xmm3
	orpd	%xmm2, %xmm4
	psubq	.L__mask_1023(%rip),%xmm3
	movmskpd %xmm4, %r8d
	packssdw %xmm1, %xmm3
	cvtdq2pd %xmm3, %xmm6		/* xexp */
	movdqa	%xmm0, %xmm2
	xorq	%rax, %rax
	subpd	.L__real_one(%rip), %xmm2
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdlog10)

	movdqa	%xmm0,%xmm3
	andpd	.L__real_notsign(%rip),%xmm2
	pand	.L__real_mant(%rip),%xmm3
	movdqa	%xmm3,%xmm4
	movapd	.L__real_half(%rip),%xmm5	/* .5 */

	cmppd	$1,.L__real_threshold(%rip),%xmm2
	movmskpd %xmm2,%r10d
	cmp	$3,%r10d
	jz	LBL(.Lall_nearone_log10)

	psrlq	$45,%xmm3
	movdqa	%xmm3,%xmm2
	psrlq	$1,%xmm3
	paddq	.L__mask_040(%rip),%xmm3
	pand	.L__mask_001(%rip),%xmm2
	paddq	%xmm2,%xmm3

	packssdw %xmm1,%xmm3
	cvtdq2pd %xmm3,%xmm1
	xorq	 %rcx,%rcx
	movq	 %xmm3,RZ_OFF(24)(%rsp)

	por	.L__real_half(%rip),%xmm4
	movdqa	%xmm4,%xmm2
	mulpd	.L__real_3f80000000000000(%rip),%xmm1	/* f1 = index/128 */

	leaq	.L__np_ln_lead_table(%rip),%rdx
	mov	RZ_OFF(24)(%rsp),%eax

	subpd	%xmm1,%xmm2				/* f2 = f - f1 */
	mulpd	%xmm2,%xmm5
	addpd	%xmm5,%xmm1

	divpd	%xmm1,%xmm2				/* u */

	movlpdMR	 -512(%rdx,%rax,8),%xmm0		/* z1 */
	mov	RZ_OFF(20)(%rsp),%ecx
	movhpd	 -512(%rdx,%rcx,8),%xmm0		/* z1 */
	movapd	%xmm2,%xmm1				/* u */
	mulpd	%xmm2,%xmm2				/* u^2 */
	movapd	%xmm2,%xmm5
	movapd	.L__real_cb3(%rip),%xmm3
	mulpd	%xmm2,%xmm3				/* Cu2 */
	mulpd	%xmm1,%xmm5				/* u^3 */
	addpd	.L__real_cb2(%rip),%xmm3 		/* B+Cu2 */

	mulpd	%xmm5,%xmm2				/* u^5 */
	movapd	.L__real_log2_lead(%rip),%xmm4

	mulpd	.L__real_cb1(%rip),%xmm5 		/* Au3 */
	addpd	%xmm5,%xmm1				/* u+Au3 */
	mulpd	%xmm3,%xmm2				/* u5(B+Cu2) */

	movapd	%xmm0,%xmm3

	addpd	%xmm2,%xmm1				/* poly */
	mulpd	%xmm6,%xmm4				/* xexp * log2_lead */
	addpd	%xmm4,%xmm0				/* r1 */
	addpd	%xmm4,%xmm3				/* r1 */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	movlpdMR	 -512(%rdx,%rax,8),%xmm4		/* z2+=q */
	movhpd	 -512(%rdx,%rcx,8),%xmm4		/* z2+=q */

	addpd	%xmm4,%xmm1

	mulpd	.L__log10_multiplier2(%rip),%xmm3
	mulpd	.L__log10_multiplier1(%rip),%xmm0
	mulpd	.L__real_log2_tail(%rip),%xmm6

	addpd	%xmm6,%xmm1				/* r2 */

	mulpd	.L__log10_multiplier(%rip),%xmm1
	addpd	%xmm3,%xmm1
	addpd	%xmm1,%xmm0

	test		 $3,%r10d
	jnz		LBL(.Lnear_one_log10)

LBL(.Lfinishn1_log10):

#if defined(_WIN64)
	movdqa	RZ_OFF(56)(%rsp), %xmm6
#endif
	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.Lall_nearone_log10):
	movapd	.L__real_two(%rip),%xmm2
	subpd	.L__real_one(%rip),%xmm0	/* r */
	addpd	%xmm0,%xmm2
	movapd	%xmm0,%xmm1
	divpd	%xmm2,%xmm1			/* u */
	movapd	.L__real_ca4(%rip),%xmm4  	/* D */
	movapd	.L__real_ca3(%rip),%xmm5 	/* C */
	movapd	%xmm0,%xmm6
	mulpd	%xmm1,%xmm6			/* correction */
	addpd	%xmm1,%xmm1			/* u */
	movapd	%xmm1,%xmm2
	mulpd	%xmm2,%xmm2			/* v =u^2 */
	mulpd	%xmm1,%xmm5			/* Cu */
	movapd	%xmm1,%xmm3
	mulpd	%xmm2,%xmm3			/* u^3 */
	mulpd	.L__real_ca2(%rip),%xmm2	/* Bu^2 */
	mulpd	%xmm3,%xmm4			/* Du^3 */

	addpd	.L__real_ca1(%rip),%xmm2	/* +A */
	movapd	%xmm3,%xmm1
	mulpd	%xmm1,%xmm1			/* u^6 */
	addpd	%xmm4,%xmm5			/* Cu+Du3 */

	movapd	%xmm0,%xmm4
	mulpd	%xmm3,%xmm2			/* u3(A+Bu2) */
	mulpd	%xmm5,%xmm1			/* u6(Cu+Du3) */
	addpd	%xmm1,%xmm2
	subpd	%xmm6,%xmm2			/*  -correction */
	
	mulpd	.L__log10_multiplier1(%rip),%xmm0
	mulpd	.L__log10_multiplier2(%rip),%xmm4
	mulpd	.L__log10_multiplier(%rip),%xmm2
	addpd	%xmm4,%xmm2
	addpd	%xmm2,%xmm0
	jmp	LBL(.Lfinishn1_log10)

	ALN_QUAD
LBL(.Lnear_one_log10):
	test	$1,%r10d
	jz	LBL(.Llnn12_log10)

	movlpd	RZ_OFF(40)(%rsp),%xmm0
	call	LBL(.Lln1_log10)

LBL(.Llnn12_log10):
	test	$2,%r10d			/* second number? */
	jz	LBL(.Llnn1e_log10)
	movlpd	%xmm0,RZ_OFF(40)(%rsp)
	movlpdMR	RZ_OFF(32)(%rsp),%xmm0
	call	LBL(.Lln1_log10)
	movlpd	%xmm0,RZ_OFF(32)(%rsp)
	movapd	RZ_OFF(40)(%rsp),%xmm0

LBL(.Llnn1e_log10):
	jmp		LBL(.Lfinishn1_log10)

LBL(.Lln1_log10):
	movlpdMR	.L__real_two(%rip),%xmm2
	subsd	.L__real_one(%rip),%xmm0	/* r */
	addsd	%xmm0,%xmm2
	movsdRR	%xmm0,%xmm1
	divsd	%xmm2,%xmm1			/* u */
	movlpdMR	.L__real_ca4(%rip),%xmm4	/* D */
	movlpdMR	.L__real_ca3(%rip),%xmm5	/* C */
	movsdRR	%xmm0,%xmm6
	mulsd	%xmm1,%xmm6			/* correction */
	addsd	%xmm1,%xmm1			/* u */
	movsdRR	%xmm1,%xmm2
	mulsd	%xmm2,%xmm2			/* v =u^2 */
	mulsd	%xmm1,%xmm5			/* Cu */
	movsdRR	%xmm1,%xmm3
	mulsd	%xmm2,%xmm3			/* u^3 */
	mulsd	.L__real_ca2(%rip),%xmm2	/*Bu^2 */
	mulsd	%xmm3,%xmm4			/*Du^3 */

	addsd	.L__real_ca1(%rip),%xmm2	/* +A */
	movsdRR	%xmm3,%xmm1
	mulsd	%xmm1,%xmm1			/* u^6 */
	addsd	%xmm4,%xmm5			/* Cu+Du3 */

	movsdRR	%xmm0,%xmm4
	mulsd	%xmm3,%xmm2			/* u3(A+Bu2) */
	mulsd	%xmm5,%xmm1			/* u6(Cu+Du3) */
	addsd	%xmm1,%xmm2
	subsd	%xmm6,%xmm2			/* -correction */
	mulsd	.L__log10_multiplier1(%rip), %xmm0
	mulsd	.L__log10_multiplier2(%rip), %xmm4
	mulsd	.L__log10_multiplier(%rip),%xmm2
	
	addsd	%xmm4,%xmm2
	addsd	%xmm2,%xmm0
	ret

#define _X0 0
#define _X1 8

#define _R0 32
#define _R1 40

LBL(.L__Scalar_fvdlog10):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        movapd  %xmm0, _X0(%rsp)

#ifdef GH_TARGET
        CALL(ENT(__fsd_log10))
#else
        CALL(ENT(__fmth_i_dlog10))
#endif
        movsd   %xmm0, _R0(%rsp)

        movsd   _X1(%rsp), %xmm0
#ifdef GH_TARGET
        CALL(ENT(__fsd_log10))
#else
        CALL(ENT(__fmth_i_dlog10))
#endif
        movsd   %xmm0, _R1(%rsp)

        movapd  _R0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.Lfinishn1_log10)


        ELF_FUNC(ENT_GH(__fvdlog10))
        ELF_SIZE(ENT_GH(__fvdlog10))
        IF_GH(ELF_FUNC(__fvd_log10))
        IF_GH(ELF_SIZE(__fvd_log10))


#endif

#endif
