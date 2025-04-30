
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */


        ALN_QUAD

/* ============================================================
 *
 *     Constants for exponential functions
 *
 * ============================================================
 */

.L__ps_mask_unsign:             .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
                                .long 0x7fffffff
.L__sp_ln_max_singleval:	.long 0x42b17217		/* 88.72283172607422 */
				.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
				.long 0x42b17217
.L_sp_real_infinity:		.long 0x7f800000	/* +inf */
.L_sp_real_ninfinity:		.long 0xff800000	/* -inf */
.L_real_min_singleval:		.long 0xc2cff1b5	/* -103.9720840454102 */
.L_real_cvt_nan:      		.long 0x00400000

	ALN_QUAD

.L_s_real_3fe0000000000000:     .long 0x3F000000        /* 1/2 */
                                .long 0x3F000000
                                .long 0x3F000000
                                .long 0x3F000000
                                .long 0x3F000000
                                .long 0x3F000000
                                .long 0x3F000000
                                .long 0x3F000000
.L_s_real_thirtytwo_by_log2:    .long 0x4238AA3B        /* thirtytwo_by_log2 */
                                .long 0x4238AA3B
                                .long 0x4238AA3B
                                .long 0x4238AA3B
                                .long 0x4238AA3B
                                .long 0x4238AA3B
                                .long 0x4238AA3B
                                .long 0x4238AA3B
.L_s_real_log2_by_32:           .long 0x3CB17218        /* log2_by_32 */
                                .long 0x3CB17218
                                .long 0x3CB17218
                                .long 0x3CB17218
                                .long 0x3CB17218
                                .long 0x3CB17218
                                .long 0x3CB17218
                                .long 0x3CB17218
.L_s_real_3FC5555555548F7C:     .long 0x3E2AAAAB        /* 1.66666666665260878863e-01 */
                                .long 0x3E2AAAAB
                                .long 0x3E2AAAAB
                                .long 0x3E2AAAAB
                                .long 0x3E2AAAAB
                                .long 0x3E2AAAAB
                                .long 0x3E2AAAAB
                                .long 0x3E2AAAAB
.L_s_two_to_jby32_table:
        .long   0x3F800000	/* 1.0000000000000000 */
        .long   0x3F82CD87	/* 1.0218971486541166 */
        .long   0x3F85AAC3	/* 1.0442737824274138 */
        .long   0x3F88980F	/* 1.0671404006768237 */
        .long   0x3F8B95C2	/* 1.0905077326652577 */
        .long   0x3F8EA43A	/* 1.1143867425958924 */
        .long   0x3F91C3D3	/* 1.1387886347566916 */
        .long   0x3F94F4F0	/* 1.1637248587775775 */
        .long   0x3F9837F0	/* 1.1892071150027210 */
        .long   0x3F9B8D3A	/* 1.2152473599804690 */
        .long   0x3F9EF532	/* 1.2418578120734840 */
        .long   0x3FA27043	/* 1.2690509571917332 */
        .long   0x3FA5FED7	/* 1.2968395546510096 */
        .long   0x3FA9A15B	/* 1.3252366431597413 */
        .long   0x3FAD583F	/* 1.3542555469368927 */
        .long   0x3FB123F6	/* 1.3839098819638320 */
        .long   0x3FB504F3	/* 1.4142135623730951 */
        .long   0x3FB8FBAF	/* 1.4451808069770467 */
        .long   0x3FBD08A4	/* 1.4768261459394993 */
        .long   0x3FC12C4D	/* 1.5091644275934228 */
        .long   0x3FC5672A	/* 1.5422108254079407 */
        .long   0x3FC9B9BE	/* 1.5759808451078865 */
        .long   0x3FCE248C	/* 1.6104903319492543 */
        .long   0x3FD2A81E	/* 1.6457554781539649 */
        .long   0x3FD744FD	/* 1.6817928305074290 */
        .long   0x3FDBFBB8	/* 1.7186192981224779 */
        .long   0x3FE0CCDF	/* 1.7562521603732995 */
        .long   0x3FE5B907	/* 1.7947090750031072 */
        .long   0x3FEAC0C7	/* 1.8340080864093424 */
        .long   0x3FEFE4BA	/* 1.8741676341103000 */
        .long   0x3FF5257D	/* 1.9152065613971474 */
        .long   0x3FFA83B3	/* 1.9571441241754002 */


/* ============================================================
 *
 *     Constants for logarithm functions, single precision
 *
 * ============================================================
 */

	ALN_QUAD
.L4_386:
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */
        .long   0x3f800000      /* 1.0 */

/* ============================================================
 *
 *     Constants for logrithm functions, double precision
 *
 * ============================================================
 */

	ALN_QUAD

.L__np_ln_lead_table:
	.quad	0x0000000000000000 		/* 0.00000000000000000000e+00 */


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
        .long   0x7f800000
        .long   0x7f800000
        .long   0x7f800000
        .long   0x7f800000

/* ==============================================================
 *
 *    Constants for mask intrinsic functions
 *
 * ==============================================================
 */
        ALN_QUAD
.L_zeromask:    .quad 0xFFFFFFFFFFFFFFFF
                .quad 0xFFFFFFFFFFFFFFFF
                .quad 0xFFFFFFFFFFFFFFFF
                .quad 0xFFFFFFFFFFFFFFFF
.L_s_zeromask:  .long 0xFFFFFFFF
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
