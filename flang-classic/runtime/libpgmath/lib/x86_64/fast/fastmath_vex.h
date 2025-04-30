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

/* ============================================================ */

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
        .globl ENT(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA)):

	vmovapd	%xmm0, %xmm1		/* Move input vector */
	vandpd   .L__real_mask_unsign(%rip), %xmm0, %xmm0

	pushq   %rbp
	movq    %rsp, %rbp
	subq    $128, %rsp

	vmovddup  .L__dble_pi_over_fours(%rip),%xmm2
	vmovddup  .L__dble_needs_argreds(%rip),%xmm3
	vmovddup  .L__dble_sixteen_by_pi(%rip),%xmm4

	vcmppd   $5, %xmm0, %xmm2, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	vcmppd   $2, %xmm0, %xmm3, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        vmovmskpd %xmm2, %eax
        vmovmskpd %xmm3, %ecx

	test	$3, %eax
        jnz	LBL(.L__Scalar_fvdsin1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulpd   %xmm1,%xmm4,%xmm4

	test	$3, %ecx
        jnz	LBL(.L__Scalar_fvdsin2)

#if defined(_WIN64)
        vmovdqu  %ymm6, 64(%rsp)
        vmovdqu  %ymm7, 96(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5    /* convert to integer */
	vmovddup   .L__dble_pi_by_16_ms(%rip), %xmm0
	vmovddup   .L__dble_pi_by_16_ls(%rip), %xmm2
	vmovddup   .L__dble_pi_by_16_us(%rip), %xmm3

        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
/*	vmulpd   %xmm4,%xmm0,%xmm0 */     /* n * p1 */
        vmulpd   %xmm4,%xmm2,%xmm2   /* n * p2 == rt */
        vmulpd   %xmm4,%xmm3,%xmm3   /* n * p3 */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */

        /* How to convert N into a table address */
        vmovapd  %xmm1,%xmm6   /* x in xmm6 */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm1,%xmm0,%xmm4,%xmm1
	VFNMA_231PD	(%xmm0,%xmm4,%xmm1)
#	VFNMADDPD	%xmm6,%xmm0,%xmm4,%xmm6
	VFNMA_231PD	(%xmm0,%xmm4,%xmm6)
#else
	vmulpd   %xmm4,%xmm0,%xmm0     /* n * p1 */
        vsubpd   %xmm0,%xmm1,%xmm1   /* x - n * p1 == rh */
        vsubpd   %xmm0,%xmm6,%xmm6   /* x - n * p1 == rh == c */
#endif
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
/*	vsubpd   %xmm0,%xmm6,%xmm6 */   /* x - n * p1 == rh == c */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        vsubpd   %xmm2,%xmm1,%xmm1   /* rh = rh - rt */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */
        vsubpd   %xmm1,%xmm6,%xmm6   /* (c - rh) */
        movq    %rax, %rdx    /* Duplicate it */
        vmovapd  %xmm1,%xmm0   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        vmovapd  %xmm1,%xmm4   /* Move rh */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        vmovapd  %xmm1,%xmm5   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        vsubpd   %xmm2,%xmm6,%xmm6   /* ((c - rh) - rt) */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vsubpd   %xmm6,%xmm3,%xmm3   /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        addq    %rdx, %rax    /* Final tbl address */
        vmovapd  %xmm1,%xmm2   /* Move rh */
        shrq    $32, %r9
        vsubpd   %xmm3,%xmm0,%xmm0   /* c = rh - rt aka r */
        movq    %rcx, %rdx    /* Duplicate it */
        vsubpd   %xmm3,%xmm4,%xmm4   /* c = rh - rt aka r */
        sarq    $4,%rcx       /* Sign bits moved down */
        vsubpd   %xmm3,%xmm5,%xmm5   /* c = rh - rt aka r */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        vsubpd   %xmm0,%xmm1,%xmm1   /* (rh - c) */
        sarq    $4,%rcx       /* Sign bits moved down */
        vmulpd   %xmm0,%xmm0,%xmm0   /* r^2 in xmm0 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmovapd  %xmm4,%xmm6   /* r in xmm6 */
        addq    %rdx, %rcx    /* Final tbl address */
        vmulpd   %xmm4,%xmm4,%xmm4   /* r^2 in xmm4 */
        leaq    24(%r9),%r8   /* Add 24 for sine */
        vmovapd  %xmm5,%xmm7   /* r in xmm7 */
        andq    $0x1f,%r8     /* And lower 5 bits */
        vmulpd   %xmm5,%xmm5,%xmm5   /* r^2 in xmm5 */
        andq    $0x1f,%r9     /* And lower 5 bits */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        vmulpd   .L__dble_pq4(%rip), %xmm0,%xmm0     /* p4 * r^2 */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        vsubpd   %xmm6,%xmm2,%xmm2                   /* (rh - c) */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        vmulpd   .L__dble_pq4+16(%rip), %xmm4, %xmm4  /* q4 * r^2 */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        vsubpd   %xmm3,%xmm1, %xmm1                   /* (rh - c) - rt aka rr */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */
        vaddpd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* + p3 */
        movq    %r8, %rdx     /* Duplicate it */
        vaddpd   .L__dble_pq3+16(%rip), %xmm4, %xmm4  /* + q3 */
        sarq    $4,%r8        /* Sign bits moved down */
        vsubpd   %xmm3,%xmm2,%xmm2                   /* (rh - c) - rt aka rr */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq2(%rip),%xmm5,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq2(%rip),%xmm5,%xmm0)
#	VFMADDPD	.L__dble_pq2+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213PD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#else
        vmulpd   %xmm5,%xmm0,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        vaddpd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        vaddpd   .L__dble_pq2+16(%rip), %xmm4, %xmm4  /* + q2 */
#endif
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmulpd   %xmm5,%xmm7,%xmm7                   /* xmm7 = r^3 */
        addq    %rdx, %r8     /* Final tbl address */
        vmovapd  %xmm1,%xmm3                   /* Move rr */
        movq    %r9, %rdx     /* Duplicate it */
        vmulpd   %xmm5,%xmm1,%xmm1                   /* r * r * rr */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmulpd   %xmm6, %xmm3, %xmm3                  /* r * rr */
        addq    %rdx, %r9     /* Final tbl address */
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rax,%rax
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,.L__dble_pq1+16(%rip),%xmm1,%xmm2
	VFMA_231PD	(.L__dble_pq1+16(%rip),%xmm1,%xmm2)
#else
        vmulpd   .L__dble_pq1+16(%rip), %xmm1, %xmm1  /* r * r * rr * 0.5 */
        vaddpd   %xmm1,%xmm2, %xmm2                   /* cs = rr - r * r * rt * 0.5 */
#endif
        addq    %r8,%r8
        vmovsd  8(%rdx,%rax,8),%xmm1          /* ds2 in xmm1 */
        vmovhpd  8(%rdx,%r8,8),%xmm1,%xmm1           /* ds2 in xmm1 */


        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq1(%rip),%xmm5,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm5,%xmm0)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm5,%xmm4)
#	VFMADDPD	%xmm2,%xmm7,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm7,%xmm0)
#	VFMSUBPD	%xmm3,%xmm5,%xmm4,%xmm4
	VFMS_213PD	(%xmm3,%xmm5,%xmm4)
#else
        vmulpd   %xmm5,%xmm0, %xmm0                   /* * r^2 */
        vmulpd   %xmm5,%xmm4, %xmm4                   /* * r^2 */
        vaddpd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
        vaddpd   .L__dble_pq1+16(%rip), %xmm4, %xmm4  /* + q1 */
        vmulpd   %xmm7,%xmm0,%xmm0                   /* * r^3 */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* * r^2 == dq aka q(r) */
        vaddpd   %xmm2,%xmm0,%xmm0                   /* + cs  == dp aka p(r) */
        vsubpd   %xmm3,%xmm4,%xmm4                   /* - cc  == dq aka q(r) */
#endif

        addq    %rcx,%rcx
        addq    %r9,%r9
        vmovsd  8(%rdx,%rcx,8),%xmm3          /* dc2 in xmm3 */
        vmovhpd  8(%rdx,%r9,8),%xmm3,%xmm3           /* dc2 in xmm3 */

        vmovsd   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        vmovhpd   (%rdx,%r8,8),%xmm5,%xmm5           /* ds1 in xmm5 */

        vaddpd   %xmm6,%xmm0,%xmm0                   /* + r   == dp aka p(r) */
        vmovapd  %xmm1,%xmm2                   /* ds2 in xmm2 */
        vmovsd  (%rdx,%rcx,8),%xmm6           /* dc1 in xmm6 */
        vmovhpd  (%rdx,%r9,8),%xmm6,%xmm6            /* dc1 in xmm6 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm4,%xmm1,%xmm1
	VFMA_213PD	(%xmm2,%xmm4,%xmm1)
#	VFMADDPD	%xmm1,%xmm0,%xmm3,%xmm1
	VFMA_231PD	(%xmm0,%xmm3,%xmm1)
#	VFMADDPD	%xmm1,%xmm5,%xmm4,%xmm1
	VFMA_231PD	(%xmm5,%xmm4,%xmm1)
#else
        vmulpd   %xmm4,%xmm1,%xmm1      /* ds2 * dq */
        vaddpd   %xmm2,%xmm1,%xmm1      /* ds2 + ds2*dq */
        vmulpd   %xmm0,%xmm3,%xmm3      /* dc2 * dp */
        vaddpd   %xmm3,%xmm1,%xmm1      /* (ds2 + ds2*dq) + dc2*dp */
        vmulpd   %xmm5,%xmm4,%xmm4      /* ds1 * dq */
        vaddpd   %xmm4,%xmm1,%xmm1      /* ((ds2...) + dc2*dp) + ds1*dq */
#endif
        vmulpd   %xmm6,%xmm0,%xmm0                   /* dc1 * dp */

#if defined(_WIN64)
        vmovdqu  64(%rsp),%ymm6
        vmovdqu  96(%rsp),%ymm7
#endif
        vaddpd   %xmm5,%xmm1,%xmm1
	vaddpd   %xmm1,%xmm0,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsin1):
        vmovapd  %xmm0, (%rsp)                 /* Save xmm0 */
	vcmppd   $3, %xmm0, %xmm0, %xmm0              /* 3 is "unordered" */
        vmovapd  %xmm1, 16(%rsp)               /* Save xmm1 */
        vmovmskpd %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

        vmovapd  16(%rsp), %xmm0
	test    $1, %eax
	jz	LBL(.L__Scalar_fvdsin3)
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsin1a)

        vmovapd  %xmm0,%xmm1
        vmovapd  %xmm0,%xmm2
	vmovddup  .L__dble_dsin_c6(%rip),%xmm3    /* c6 */

        vmulpd   %xmm0,%xmm0,%xmm0
        vmulpd   %xmm1,%xmm1,%xmm1
	vmovddup  .L__dble_dsin_c5(%rip),%xmm4    /* c5 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm3,%xmm0,%xmm0
	VFMA_213PD	(%xmm4,%xmm3,%xmm0)
#else
        vmulpd   %xmm3,%xmm0,%xmm0                     /* x2 * c6 */
        vaddpd   %xmm4,%xmm0,%xmm0                     /* + c5 */
#endif
	vmovddup  .L__dble_dsin_c4(%rip),%xmm3    /* c4 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm3,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm3,%xmm1,%xmm0)
#else
        vmulpd   %xmm1,%xmm0,%xmm0                     /* x2 * (c5 + ...) */
        vaddpd   %xmm3,%xmm0,%xmm0                     /* + c4 */
#endif
	vmovddup  .L__dble_dsin_c3(%rip),%xmm4    /* c3 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm4,%xmm1,%xmm0)
#else
        vmulpd   %xmm1,%xmm0,%xmm0                     /* x2 * (c4 + ...) */
        vaddpd   %xmm4,%xmm0,%xmm0                     /* + c3 */
#endif
	vmovddup  .L__dble_dsin_c2(%rip),%xmm3    /* c2 */

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDPD	%xmm3,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm3,%xmm1,%xmm0)
#	VFMADDPD	.L__dble_pq1(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm1,%xmm0)
	vmulpd		%xmm2,%xmm1,%xmm1
#	VFMADDPD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm1,%xmm0)
#else */
        vmulpd   %xmm1,%xmm0,%xmm0                     /* x2 * (c3 + ...) */
        vaddpd   %xmm3,%xmm0,%xmm0                     /* + c2 */
        vmulpd   %xmm1,%xmm0,%xmm0                     /* x2 * (c2 + ...) */
        vaddpd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + c1 */
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x3 */
        vmulpd   %xmm1,%xmm0,%xmm0                     /* x3 * (c1 + ...) */
        vaddpd   %xmm2,%xmm0,%xmm0                     /* x + x3 * (...) done */
/* #endif */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsin1a):
	movq	(%rsp),%rdx
	call	LBL(.L__fvd_sin_local)
	jmp	LBL(.L__Scalar_fvdsin5)

LBL(.L__Scalar_fvdsin2):
        vmovapd  %xmm0, (%rsp)                 /* Save xmm0 */
        vmovapd  %xmm1, %xmm0                  /* Save xmm1 */
        vmovapd  %xmm1, 16(%rsp)               /* Save xmm1 */

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
	CALL(ENT(ASM_CONCAT(__fsd_sin_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

LBL(.L__Scalar_fvdsin5):
        vmovlpd  %xmm0, (%rsp)
        vmovsd  24(%rsp), %xmm0
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsin6)
	movq	8(%rsp),%rdx
	call	LBL(.L__fvd_sin_local)
	jmp	LBL(.L__Scalar_fvdsin8)

LBL(.L__Scalar_fvdsin6):
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvdsin7)
	CALL(ENT(__mth_i_dsin))
	jmp	LBL(.L__Scalar_fvdsin8)

LBL(.L__Scalar_fvdsin7):
	CALL(ENT(ASM_CONCAT(__fsd_sin_,TARGET_VEX_OR_FMA)))


LBL(.L__Scalar_fvdsin8):
        vmovlpd  %xmm0, 8(%rsp)
	vmovapd	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvd_sin_local):
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2
        shrq    $48,%rdx
        cmpl    $0x03f20,%edx
        jl      LBL(.L__fvd_sin_small)
        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   .L__dble_dsin_c6(%rip),%xmm0,%xmm0    /* x2 * c6 */
        vaddsd   .L__dble_dsin_c5(%rip),%xmm0,%xmm0    /* + c5 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c4(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c4(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dble_dsin_c3(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c3(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm1,%xmm0)
#else
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x2 * (c5 + ...) */
        vaddsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* + c4 */
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x2 * (c4 + ...) */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + c3 */
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + c2 */
#endif


/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm1,%xmm0)
	vmulsd		%xmm2,%xmm1,%xmm1
#	VFMADDSD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#else */
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x2 * (c2 + ...) */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x3 */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + c1 */
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x3 * (c1 + ...) */
        vaddsd   %xmm2,%xmm0,%xmm0                     /* x + x3 * (...) done */
/* #endif */
        ret

LBL(.L__fvd_sin_small):
        cmpl    $0x03e40,%edx
        jl      LBL(.L__fvd_sin_done1)
        /* return x - x * x * x * 1/3! */
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   .L__dble_pq1(%rip),%xmm2,%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm1,%xmm0
	VFMA_231SD	(%xmm2,%xmm1,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif
        ret

LBL(.L__fvd_sin_done1):
	rep
        ret

        ELF_FUNC(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */
/*
 *  vector cosine
 *
 *  An implementation of the cosine libm function.
 *
 *  Prototype:
 *
 *      double __fvd_cos(double *x);
 *
 *  Returns C99 values for error conditions, but may not
 *  set flags and other error status.
 *
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA)):

	vmovapd	%xmm0, %xmm1		/* Move input vector */
        vandpd   .L__real_mask_unsign(%rip), %xmm0, %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

	vmovddup  .L__dble_pi_over_fours(%rip),%xmm2
	vmovddup  .L__dble_needs_argreds(%rip),%xmm3
	vmovddup  .L__dble_sixteen_by_pi(%rip),%xmm4

	vcmppd   $5, %xmm0, %xmm2, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	vcmppd   $2, %xmm0, %xmm3, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        vmovmskpd %xmm2, %eax
        vmovmskpd %xmm3, %ecx

	test	$3, %eax
        jnz	LBL(.L__Scalar_fvdcos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulpd   %xmm1,%xmm4,%xmm4

	test	$3, %ecx
        jnz	LBL(.L__Scalar_fvdcos2)

#if defined(_WIN64)
        vmovdqu  %ymm6, 64(%rsp)
        vmovdqu  %ymm7, 96(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5    /* convert to integer */
	vmovddup   .L__dble_pi_by_16_ms(%rip), %xmm0
	vmovddup   .L__dble_pi_by_16_ls(%rip), %xmm2
	vmovddup   .L__dble_pi_by_16_us(%rip), %xmm3

        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm1,%xmm4,%xmm0,%xmm1
	VFNMA_231PD	(%xmm4,%xmm0,%xmm1)
#	VFNMADDPD	%xmm6,%xmm4,%xmm0,%xmm6
	VFNMA_231PD	(%xmm4,%xmm0,%xmm6)
#else
        vmulpd   %xmm4,%xmm0,%xmm0     /* n * p1 */
        vsubpd   %xmm0,%xmm1,%xmm1   /* x - n * p1 == rh */
        vsubpd   %xmm0,%xmm6,%xmm6   /* x - n * p1 == rh == c */
#endif
        vmulpd   %xmm4,%xmm2,%xmm2   /* n * p2 == rt */
        vmulpd   %xmm4,%xmm3,%xmm3   /* n * p3 */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */

        /* How to convert N into a table address */
        vmovapd  %xmm1,%xmm6   /* x in xmm6 */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
/*	vsubpd   %xmm0,%xmm1,%xmm1 */   /* x - n * p1 == rh */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
/*	vsubpd   %xmm0,%xmm6,%xmm6 */   /* x - n * p1 == rh == c */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        vsubpd   %xmm2,%xmm1,%xmm1   /* rh = rh - rt */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */
        vsubpd   %xmm1,%xmm6,%xmm6   /* (c - rh) */
        movq    %rax, %rdx    /* Duplicate it */
        vmovapd  %xmm1,%xmm0   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        vmovapd  %xmm1,%xmm4   /* Move rh */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        vmovapd  %xmm1,%xmm5   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        vsubpd   %xmm2,%xmm6,%xmm6   /* ((c - rh) - rt) */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vsubpd   %xmm6,%xmm3,%xmm3   /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        addq    %rdx, %rax    /* Final tbl address */
        vmovapd  %xmm1,%xmm2   /* Move rh */
        shrq    $32, %r9
        vsubpd   %xmm3,%xmm0,%xmm0   /* c = rh - rt aka r */
        movq    %rcx, %rdx    /* Duplicate it */
        vsubpd   %xmm3,%xmm4,%xmm4   /* c = rh - rt aka r */
        sarq    $4,%rcx       /* Sign bits moved down */
        vsubpd   %xmm3,%xmm5,%xmm5   /* c = rh - rt aka r */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        vsubpd   %xmm0,%xmm1,%xmm1   /* (rh - c) */
        sarq    $4,%rcx       /* Sign bits moved down */
        vmulpd   %xmm0,%xmm0,%xmm0   /* r^2 in xmm0 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmovapd  %xmm4,%xmm6   /* r in xmm6 */
        addq    %rdx, %rcx    /* Final tbl address */
        vmulpd   %xmm4,%xmm4,%xmm4   /* r^2 in xmm4 */
        leaq    24(%r9),%r8   /* Add 24 for sine */
        vmovapd  %xmm5,%xmm7   /* r in xmm7 */
        andq    $0x1f,%r8     /* And lower 5 bits */
        vmulpd   %xmm5,%xmm5,%xmm5   /* r^2 in xmm5 */
        andq    $0x1f,%r9     /* And lower 5 bits */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        vmulpd   .L__dble_pq4(%rip), %xmm0,%xmm0     /* p4 * r^2 */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        vsubpd   %xmm6,%xmm2,%xmm2                   /* (rh - c) */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        vmulpd   .L__dble_pq4+16(%rip), %xmm4,%xmm4  /* q4 * r^2 */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        vsubpd   %xmm3,%xmm1,%xmm1                   /* (rh - c) - rt aka rr */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */
        vaddpd   .L__dble_pq3(%rip), %xmm0,%xmm0     /* + p3 */
        movq    %r8, %rdx     /* Duplicate it */
        vaddpd   .L__dble_pq3+16(%rip), %xmm4,%xmm4  /* + q3 */
        sarq    $4,%r8        /* Sign bits moved down */
        vsubpd   %xmm3,%xmm2,%xmm2                   /* (rh - c) - rt aka rr */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq2(%rip),%xmm5,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq2(%rip),%xmm5,%xmm0)
#	VFMADDPD	.L__dble_pq2+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213PD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#else
        vmulpd   %xmm5,%xmm0,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        vaddpd   .L__dble_pq2(%rip), %xmm0,%xmm0     /* + p2 */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        vaddpd   .L__dble_pq2+16(%rip), %xmm4,%xmm4  /* + q2 */
#endif
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmulpd   %xmm5,%xmm7,%xmm7                   /* xmm7 = r^3 */
        addq    %rdx, %r8     /* Final tbl address */
        vmovapd  %xmm1,%xmm3                   /* Move rr */
        movq    %r9, %rdx     /* Duplicate it */
        vmulpd   %xmm5,%xmm1,%xmm1                   /* r * r * rr */
        sarq    $4,%r9        /* Sign bits moved down */
/*	vaddpd   .L__dble_pq2(%rip), %xmm0,%xmm0 */     /* + p2 */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
/*	vaddpd   .L__dble_pq2+16(%rip), %xmm4,%xmm4 */  /* + q2 */
        sarq    $4,%r9        /* Sign bits moved down */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,.L__dble_pq1+16(%rip),%xmm1,%xmm2
	VFMA_231PD	(.L__dble_pq1+16(%rip),%xmm1,%xmm2)
#else
        vmulpd   .L__dble_pq1+16(%rip), %xmm1,%xmm1  /* r * r * rr * 0.5 */
        vaddpd   %xmm1,%xmm2,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
#endif
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmulpd   %xmm6, %xmm3,%xmm3                  /* r * rr */
        addq    %rdx, %r9     /* Final tbl address */
/*	vmulpd   %xmm5,%xmm0,%xmm0 */                  /* * r^2 */
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
/*	vmulpd   %xmm5,%xmm4,%xmm4 */                  /* * r^2 */
        addq    %rcx,%rcx
        addq    %r9,%r9
/*	vaddpd   %xmm1,%xmm2,%xmm2 */                  /* cs = rr - r * r * rt * 0.5 */
        addq    %rax,%rax
        addq    %r8,%r8
        vmovsd	8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        vmovhpd	8(%rdx,%r9,8),%xmm1,%xmm1           /* dc2 in xmm1 */


        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

        vmulpd   %xmm5,%xmm0,%xmm0                   /* * r^2 */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* * r^2 */
        vaddpd   .L__dble_pq1(%rip), %xmm0,%xmm0     /* + p1 */
        vaddpd   .L__dble_pq1+16(%rip), %xmm4,%xmm4  /* + q1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm7,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm7,%xmm0)
#	VFMSUBPD	%xmm3,%xmm5,%xmm4,%xmm4
	VFMS_213PD	(%xmm3,%xmm5,%xmm4)
#else
        vmulpd   %xmm7,%xmm0,%xmm0                   /* * r^3 */
        vaddpd   %xmm2,%xmm0,%xmm0                   /* + cs  == dp aka p(r) */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* * r^2 == dq aka q(r) */
        vsubpd   %xmm3,%xmm4,%xmm4                   /* - cc  == dq aka q(r) */
#endif

        vmovsd  8(%rdx,%rax,8),%xmm3          /* ds2 in xmm3 */
        vmovhpd  8(%rdx,%r8,8),%xmm3,%xmm3           /* ds2 in xmm3 */

        vmovsd   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        vmovhpd   (%rdx,%r8,8),%xmm5,%xmm5           /* ds1 in xmm5 */

        vaddpd   %xmm0,%xmm6,%xmm6                   /* + r   == dp aka p(r) */
        vmovapd  %xmm1,%xmm2                   /* ds2 in xmm2 */

        vmovsd  (%rdx,%rcx,8),%xmm0           /* dc1 in xmm6 */
        vmovhpd  (%rdx,%r9,8),%xmm0,%xmm0            /* dc1 in xmm6 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm4,%xmm1,%xmm1
	VFMA_213PD	(%xmm2,%xmm4,%xmm1)
#	VFNMADDPD	%xmm1,%xmm6,%xmm3,%xmm1
	VFNMA_231PD	(%xmm6,%xmm3,%xmm1)
	vmulpd		%xmm0,%xmm4,%xmm4
#	VFNMADDPD	%xmm1,%xmm5,%xmm6,%xmm1
	VFNMA_231PD	(%xmm5,%xmm6,%xmm1)
#else
        vmulpd   %xmm4,%xmm1,%xmm1                   /* dc2 * dq */
        vaddpd   %xmm2,%xmm1,%xmm1                   /* dc2 + dc2*dq */
        vmulpd   %xmm6,%xmm3,%xmm3                   /* ds2 * dp */
        vsubpd   %xmm3,%xmm1,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */
        vmulpd   %xmm0,%xmm4,%xmm4                   /* dc1 * dq */
        vmulpd   %xmm5,%xmm6,%xmm6                   /* ds1 * dp */
        vsubpd   %xmm6,%xmm1,%xmm1                   /* ((dc2...) - ds2*dp) - ds1*dp */
#endif

#if defined(_WIN64)
        vmovdqu  64(%rsp),%ymm6
        vmovdqu  96(%rsp),%ymm7
#endif
        vaddpd   %xmm4,%xmm1,%xmm1
        vaddpd   %xmm1,%xmm0,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdcos1):
        vmovapd  %xmm0, (%rsp)                 /* Save xmm0 */
	vcmppd   $3, %xmm0, %xmm0,%xmm0              /* 3 is "unordered" */
        vmovapd  %xmm1, 16(%rsp)               /* Save xmm1 */
        vmovmskpd %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

        vmovapd  16(%rsp), %xmm0
	test    $1, %eax
	jz	LBL(.L__Scalar_fvdcos3)
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdcos1a)

        vmovapd  %xmm0,%xmm1
        vmovapd  %xmm0,%xmm2
	vmovddup  .L__dble_dcos_c6(%rip),%xmm3    /* c6 */

        vmulpd   %xmm1,%xmm1,%xmm1
        vmulpd   %xmm2,%xmm2,%xmm2
	vmovddup  .L__dble_dcos_c5(%rip),%xmm4    /* c5 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm3,%xmm1)
#else
        vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * c6 */
        vaddpd   %xmm4,%xmm1,%xmm1                     /* + c5 */
#endif
	vmovapd  .L__real_one(%rip), %xmm0       /* 1.0 */
	vmovddup  .L__dble_dcos_c4(%rip),%xmm3    /* c4 */

#ifdef TARGET_FMA
#        VFMADDPD	%xmm3,%xmm2,%xmm1,%xmm1
	VFMA_213PD	(%xmm3,%xmm2,%xmm1)
#else
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x2 * (c5 + ...) */
        vaddpd   %xmm3,%xmm1,%xmm1                     /* + c4 */
#endif
	vmovddup  .L__dble_dcos_c3(%rip),%xmm4    /* c3 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm2,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm2,%xmm1)
#else
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x2 * (c4 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1                     /* + c3 */
#endif
	vmovddup  .L__dble_dcos_c2(%rip),%xmm3    /* c2 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm3,%xmm2,%xmm1,%xmm1
	VFMA_213PD	(%xmm3,%xmm2,%xmm1)
#else
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x2 * (c3 + ...) */
        vaddpd   %xmm3,%xmm1,%xmm1                     /* + c2 */
#endif
	vmovddup  .L__dble_dcos_c1(%rip),%xmm4    /* c1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm2,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm2,%xmm1)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#	VFMADDPD	%xmm0,%xmm2,%xmm1,%xmm0
	VFMA_231PD	(%xmm2,%xmm1,%xmm0)
#else
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x2 * (c2 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1                     /* + c1 */
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x2 */
        vaddpd   .L__dble_pq1+16(%rip),%xmm1,%xmm1     /* - 0.5 */
        vmulpd   %xmm2,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
        vaddpd   %xmm1,%xmm0,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdcos1a):
	movq	(%rsp),%rdx
	call	LBL(.L__fvd_cos_local)
	jmp	LBL(.L__Scalar_fvdcos5)

LBL(.L__Scalar_fvdcos2):
        vmovapd  %xmm0, (%rsp)                 /* Save xmm0 */
        vmovapd  %xmm1, %xmm0                  /* Save xmm1 */
        vmovapd  %xmm1, 16(%rsp)               /* Save xmm1 */

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
	CALL(ENT(ASM_CONCAT(__fsd_cos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

LBL(.L__Scalar_fvdcos5):
        vmovlpd  %xmm0, (%rsp)
        vmovsd  24(%rsp), %xmm0
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdcos6)
	movq	8(%rsp),%rdx
	call	LBL(.L__fvd_cos_local)
	jmp	LBL(.L__Scalar_fvdcos8)

LBL(.L__Scalar_fvdcos6):
	test    $2, %ecx
	jz	LBL(.L__Scalar_fvdcos7)
	CALL(ENT(__mth_i_dcos))
	jmp	LBL(.L__Scalar_fvdcos8)

LBL(.L__Scalar_fvdcos7):
	CALL(ENT(ASM_CONCAT(__fsd_cos_,TARGET_VEX_OR_FMA)))


LBL(.L__Scalar_fvdcos8):
        vmovlpd  %xmm0, 8(%rsp)
	vmovapd	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvd_cos_local):
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2
        shrq    $48,%rdx
	vmovsd  .L__dble_sincostbl(%rip), %xmm0 /* 1.0 */
        cmpl    $0x03f20,%edx
        jl      LBL(.L__fvd_cos_small)
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_dcos_c6(%rip),%xmm1,%xmm1    /* x2 * c6 */
        vaddsd   .L__dble_dcos_c5(%rip),%xmm1,%xmm1    /* + c5 */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_dcos_c4(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c4(%rip),%xmm2,%xmm1)
#	VFMADDPD	.L__dble_dcos_c3(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c3(%rip),%xmm2,%xmm1)
#	VFMADDPD	.L__dble_dcos_c2(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c2(%rip),%xmm2,%xmm1)
#	VFMADDPD	.L__dble_dcos_c1(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c1(%rip),%xmm2,%xmm1)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#	VFMADDPD	%xmm0,%xmm2,%xmm1,%xmm0
	VFMA_231PD	(%xmm2,%xmm1,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c5 + ...) */
        vaddsd   .L__dble_dcos_c4(%rip),%xmm1,%xmm1    /* + c4 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c4 + ...) */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm1,%xmm1    /* + c3 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm1,%xmm1    /* + c2 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm1,%xmm1    /* + c1 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1     /* - 0.5 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm1,%xmm0,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
#endif
        ret

LBL(.L__fvd_cos_small):
        cmpl    $0x03e40,%edx
        jl      LBL(.L__fvd_cos_done1)
        /* return 1.0 - x * x * 0.5 */
        vmulsd   %xmm1,%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,.L__dble_pq1+16(%rip),%xmm1,%xmm0
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm1,%xmm0)
#else
        vmulsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif
        ret

LBL(.L__fvd_cos_done1):
	rep
        ret

        ELF_FUNC(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA))


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
        .globl ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)):

	vmovapd	%xmm0, %xmm1		/* Move input vector */
        vandpd   .L__real_mask_unsign(%rip), %xmm0, %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

	vmovddup  .L__dble_pi_over_fours(%rip),%xmm2
	vmovddup  .L__dble_needs_argreds(%rip),%xmm3
	vmovddup  .L__dble_sixteen_by_pi(%rip),%xmm4


	vcmppd   $5, %xmm0, %xmm2, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	vcmppd   $2, %xmm0, %xmm3, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        vmovmskpd %xmm2, %eax
        vmovmskpd %xmm3, %ecx

	test	$3, %eax
        jnz	LBL(.L__Scalar_fvdsincos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulpd   %xmm1,%xmm4,%xmm4

	test	$3, %ecx
        jnz	LBL(.L__Scalar_fvdsincos2)

#if defined(_WIN64)
        vmovdqu  %ymm6, 32(%rsp)
        vmovdqu  %ymm7, 64(%rsp)
        vmovdqu  %ymm8, 96(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5    /* convert to integer */
	vmovddup   .L__dble_pi_by_16_ms(%rip), %xmm0
	vmovddup   .L__dble_pi_by_16_ls(%rip), %xmm2
	vmovddup   .L__dble_pi_by_16_us(%rip), %xmm3

        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm1,%xmm4,%xmm0,%xmm1
	VFNMA_231PD	(%xmm4,%xmm0,%xmm1)
#	VFNMADDPD	%xmm6,%xmm0,%xmm1,%xmm6
	VFNMA_231PD	(%xmm0,%xmm1,%xmm6)
#else
        vmulpd   %xmm4,%xmm0,%xmm0     /* n * p1 */
        vsubpd   %xmm0,%xmm1,%xmm1   /* x - n * p1 == rh */
        vsubpd   %xmm0,%xmm6,%xmm6   /* x - n * p1 == rh == c */
#endif
        vmulpd   %xmm4,%xmm2,%xmm2   /* n * p2 == rt */
        vmulpd   %xmm4,%xmm3,%xmm3   /* n * p3 */
        leaq    24(%rcx),%rax /* Add 24 for sine */
	movq    %rcx, %r9     /* Move it to save it */

        /* How to convert N into a table address */
        vmovapd  %xmm1,%xmm6   /* x in xmm6 */
        andq    $0x1f,%rax    /* And lower 5 bits */
        andq    $0x1f,%rcx    /* And lower 5 bits */
/*	vsubpd   %xmm0,%xmm1,%xmm1 */   /* x - n * p1 == rh */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
/*	vsubpd   %xmm0,%xmm6,%xmm6 */   /* x - n * p1 == rh == c */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        vsubpd   %xmm2,%xmm1,%xmm1   /* rh = rh - rt */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */
        vsubpd   %xmm1,%xmm6,%xmm6   /* (c - rh) */
        movq    %rax, %rdx    /* Duplicate it */
        vmovapd  %xmm1,%xmm0   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        vmovapd  %xmm1,%xmm4   /* Move rh */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        vmovapd  %xmm1,%xmm5   /* Move rh */
        sarq    $4,%rax       /* Sign bits moved down */
        vsubpd   %xmm2,%xmm6,%xmm6   /* ((c - rh) - rt) */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vsubpd   %xmm6,%xmm3,%xmm3   /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        addq    %rdx, %rax    /* Final tbl address */
        vmovapd  %xmm1,%xmm2   /* Move rh */
        shrq    $32, %r9
        vsubpd   %xmm3,%xmm0,%xmm0   /* c = rh - rt aka r */
        movq    %rcx, %rdx    /* Duplicate it */
        vsubpd   %xmm3,%xmm4,%xmm4   /* c = rh - rt aka r */
        sarq    $4,%rcx       /* Sign bits moved down */
        vsubpd   %xmm3,%xmm5,%xmm5   /* c = rh - rt aka r */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        vsubpd   %xmm0,%xmm1,%xmm1   /* (rh - c) */
        sarq    $4,%rcx       /* Sign bits moved down */
        vmulpd   %xmm0,%xmm0,%xmm0   /* r^2 in xmm0 */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmovapd  %xmm4,%xmm6   /* r in xmm6 */
        addq    %rdx, %rcx    /* Final tbl address */
        vmulpd   %xmm4,%xmm4,%xmm4   /* r^2 in xmm4 */
        leaq    24(%r9),%r8   /* Add 24 for sine */
        vmovapd  %xmm5,%xmm7   /* r in xmm7 */
        andq    $0x1f,%r8     /* And lower 5 bits */
        vmulpd   %xmm5,%xmm5,%xmm5   /* r^2 in xmm5 */
        andq    $0x1f,%r9     /* And lower 5 bits */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        vmulpd   .L__dble_pq4(%rip), %xmm0,%xmm0     /* p4 * r^2 */
        rorq    $5,%r8        /* rotate right so bit 4 is sign bit */
        vsubpd   %xmm6,%xmm2,%xmm2                   /* (rh - c) */
        rorq    $5,%r9        /* rotate right so bit 4 is sign bit */
        vmulpd   .L__dble_pq4+16(%rip), %xmm4, %xmm4  /* q4 * r^2 */
        sarq    $4,%r8        /* Duplicate sign bit 4 times */
        sarq    $4,%r9        /* Duplicate sign bit 4 times */
        vsubpd   %xmm3,%xmm1,%xmm1                   /* (rh - c) - rt aka rr */
        rolq    $9,%r8        /* Shift back to original place */
        rolq    $9,%r9        /* Shift back to original place */
        vaddpd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* + p3 */
        movq    %r8, %rdx     /* Duplicate it */
        vaddpd   .L__dble_pq3+16(%rip), %xmm4, %xmm4  /* + q3 */
        sarq    $4,%r8        /* Sign bits moved down */
        vsubpd   %xmm3,%xmm2,%xmm2                   /* (rh - c) - rt aka rr */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq2(%rip),%xmm5,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq2(%rip),%xmm5,%xmm0)
#	VFMADDPD	.L__dble_pq2+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213PD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#else
	vmulpd   %xmm5,%xmm0,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
	vaddpd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
	vmulpd   %xmm5,%xmm4,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        vaddpd   .L__dble_pq2+16(%rip), %xmm4, %xmm4  /* + q2 */
#endif
	sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmulpd   %xmm5,%xmm7,%xmm7                   /* xmm7 = r^3 */
        addq    %rdx, %r8     /* Final tbl address */
        vmovapd  %xmm1,%xmm3                   /* Move rr */
        movq    %r9, %rdx     /* Duplicate it */
        vmulpd   %xmm5,%xmm1,%xmm1                   /* r * r * rr */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,.L__dble_pq1+16(%rip),%xmm1,%xmm2
	VFMA_231PD	(.L__dble_pq1+16(%rip),%xmm1,%xmm2)
#else
	vmulpd   .L__dble_pq1+16(%rip), %xmm1, %xmm1  /* r * r * rr * 0.5 */
	vaddpd   %xmm1,%xmm2,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
#endif
        andq    $0xf,%rdx     /* And lower 5 bits */
        vmulpd   %xmm6, %xmm3, %xmm3                  /* r * rr */
        addq    %rdx, %r9     /* Final tbl address */
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %r9,%r9
        addq    %rax,%rax
        addq    %r8,%r8
        vmovsd  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        vmovhpd  8(%rdx,%r9,8),%xmm1,%xmm1           /* dc2 in xmm1 */

        vmovsd  8(%rdx,%rax,8),%xmm8          /* ds2 in xmm8 */
        vmovhpd  8(%rdx,%r8,8),%xmm8,%xmm8           /* ds2 in xmm8 */


        /* xmm0 has dp, xmm4 has dq,
           xmm1 has dc2
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3, xmm8 has ds2 */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq1(%rip),%xmm5,%xmm0, %xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm5,%xmm0)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm5,%xmm4)
#	VFMADDPD	%xmm2,%xmm7,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm7,%xmm0)
#	VFMSUBPD	%xmm3,%xmm5,%xmm4,%xmm4
	VFMS_213PD	(%xmm3,%xmm5,%xmm4)
#else
        vmulpd   %xmm5,%xmm0, %xmm0                   /* * r^2 */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* * r^2 */
        vaddpd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
        vaddpd   .L__dble_pq1+16(%rip), %xmm4, %xmm4  /* + q1 */

        vmulpd   %xmm7,%xmm0,%xmm0                   /* * r^3 */
        vmulpd   %xmm5,%xmm4,%xmm4                   /* * r^2 == dq aka q(r) */
        vaddpd   %xmm2,%xmm0,%xmm0                   /* + cs  == dp aka p(r) */
        vsubpd   %xmm3,%xmm4,%xmm4                   /* - cc  == dq aka q(r) */
#endif

	vmovapd  %xmm1,%xmm3                   /* dc2 in xmm3 */
        vmovsd  (%rdx,%rax,8),%xmm5           /* ds1 in xmm5 */
        vmovhpd  (%rdx,%r8,8),%xmm5,%xmm5            /* ds1 in xmm5 */

        vmovsd  (%rdx,%rcx,8),%xmm7           /* dc1 in xmm7 */
        vmovhpd  (%rdx,%r9,8),%xmm7,%xmm7            /* dc1 in xmm7 */


        vaddpd   %xmm6,%xmm0,%xmm0                   /* + r   == dp aka p(r) */
        vmovapd  %xmm8,%xmm2                   /* ds2 in xmm2 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm4,%xmm8,%xmm8
	VFMA_213PD	(%xmm2,%xmm4,%xmm8)
#	VFMADDPD	%xmm3,%xmm4,%xmm1,%xmm1
	VFMA_213PD	(%xmm3,%xmm4,%xmm1)
#	VFMADDPD	%xmm8,%xmm0,%xmm3,%xmm8
	VFMA_231PD	(%xmm0,%xmm3,%xmm8)
#	VFNMADDPD	%xmm1,%xmm0,%xmm2,%xmm1
	VFNMA_231PD	(%xmm0,%xmm2,%xmm1)
#else
        vmulpd   %xmm4,%xmm8,%xmm8                   /* ds2 * dq */
        vaddpd   %xmm2,%xmm8,%xmm8                   /* ds2 + ds2*dq */
        vmulpd   %xmm4,%xmm1,%xmm1                   /* dc2 * dq */
        vaddpd   %xmm3,%xmm1,%xmm1                   /* dc2 + dc2*dq */

        vmulpd   %xmm0,%xmm3,%xmm3                   /* dc2 * dp */
        vaddpd   %xmm3,%xmm8,%xmm8                   /* (ds2 + ds2*dq) + dc2*dp */
        vmulpd   %xmm0,%xmm2,%xmm2                   /* ds2 * dp */
        vsubpd   %xmm2,%xmm1,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */
#endif

        vmovapd  %xmm4,%xmm6                   /* xmm6 = dq */
        vmovapd  %xmm5,%xmm3                   /* xmm3 = ds1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm8,%xmm5,%xmm4,%xmm8
	VFMA_231PD	(%xmm5,%xmm4,%xmm8)
#	VFNMADDPD	%xmm1,%xmm0,%xmm5,%xmm1
	VFNMA_231PD	(%xmm0,%xmm5,%xmm1)
#	VFMADDPD	%xmm1,%xmm7,%xmm6,%xmm1
	VFMA_231PD	(%xmm7,%xmm6,%xmm1)
#else
        vmulpd   %xmm5,%xmm4,%xmm4                   /* ds1 * dq */
        vaddpd   %xmm4,%xmm8,%xmm8                   /* ((ds2...) + dc2*dp) + ds1*dq */
        vmulpd   %xmm0,%xmm5,%xmm5                   /* ds1 * dp */
        vsubpd   %xmm5,%xmm1,%xmm1                   /* (() - ds2*dp) - ds1*dp */
	vmulpd   %xmm7,%xmm6,%xmm6                   /* dc1 * dq */
        vaddpd   %xmm6,%xmm1,%xmm1                   /* + dc1*dq */
#endif

        vaddpd   %xmm3,%xmm8,%xmm8                   /* + ds1 */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm8,%xmm7,%xmm0,%xmm0
	VFMA_213PD	(%xmm8,%xmm7,%xmm0)
#else
        vmulpd   %xmm7,%xmm0,%xmm0                   /* dc1 * dp */
        vaddpd   %xmm8,%xmm0,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
#endif
        vaddpd   %xmm7,%xmm1,%xmm1                   /* cos(x) = (C + Cq(r)) + Sq(r) */

#if defined(_WIN64)
        vmovdqu  32(%rsp),%ymm6
        vmovdqu  64(%rsp),%ymm7
        vmovdqu  96(%rsp),%ymm8
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsincos1):
        vmovapd  %xmm0, (%rsp)                 /* Save xmm0 */
	vcmppd   $3, %xmm0, %xmm0, %xmm0              /* 3 is "unordered" */
        vmovapd  %xmm1, 16(%rsp)               /* Save xmm1 */
        vmovmskpd %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

        vmovapd  16(%rsp), %xmm0
	test    $1, %eax
	jz	LBL(.L__Scalar_fvdsincos3)
	test    $2, %eax
	jz	LBL(.L__Scalar_fvdsincos1a)

#if defined(_WIN64)
        vmovdqu  %ymm6, 64(%rsp)
        vmovdqu  %ymm7, 96(%rsp)
#endif
        vmovapd  %xmm0,%xmm2
        vmulpd   %xmm0,%xmm0,%xmm0

        vmovapd  %xmm0,%xmm1
        vmovapd  %xmm0,%xmm3

	vmovddup  .L__dble_dsin_c5(%rip),%xmm6    /* s5 */
	vmovddup  .L__dble_dcos_c5(%rip),%xmm7    /* c5 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm6,.L__dble_dsin_c6(%rip),%xmm0,%xmm0
	VFMA_132PD	(.L__dble_dsin_c6(%rip),%xmm6,%xmm0)
#	VFMADDPD	%xmm7,.L__dble_dcos_c6(%rip),%xmm1,%xmm1
	VFMA_132PD	(.L__dble_dcos_c6(%rip),%xmm7,%xmm1)
#	VFMADDPD	.L__dble_dsin_c4(%rip),%xmm3,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_dsin_c4(%rip),%xmm3,%xmm0)
#	VFMADDPD	.L__dble_dcos_c4(%rip),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c4(%rip),%xmm3,%xmm1)
#	VFMADDPD	.L__dble_dsin_c3(%rip),%xmm3,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_dsin_c3(%rip),%xmm3,%xmm0)
#	VFMADDPD	.L__dble_dcos_c3(%rip),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c3(%rip),%xmm3,%xmm1)
#	VFMADDPD	.L__dble_dsin_c2(%rip),%xmm3,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_dsin_c2(%rip),%xmm3,%xmm0)
#	VFMADDPD	.L__dble_dcos_c2(%rip),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c2(%rip),%xmm3,%xmm1)
#	VFMADDPD	.L__dble_pq1(%rip),%xmm3,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm3,%xmm0)
#	VFMADDPD	.L__dble_dcos_c1(%rip),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c1(%rip),%xmm3,%xmm1)
#else
        vmulpd   .L__dble_dsin_c6(%rip),%xmm0,%xmm0    /* x2 * s6 */
        vaddpd   %xmm6,%xmm0,%xmm0                     /* + s5 */
        vmulpd   .L__dble_dcos_c6(%rip),%xmm1,%xmm1    /* x2 * c6 */
        vaddpd   %xmm7,%xmm1,%xmm1                     /* + c5 */
        vmulpd   %xmm3,%xmm0,%xmm0                     /* x2 * (s5 + ...) */
        vaddpd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* + s4 */
        vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * (c5 + ...) */
        vaddpd   .L__dble_dcos_c4(%rip),%xmm1,%xmm1    /* + c4 */
        vmulpd   %xmm3,%xmm0,%xmm0                     /* x2 * (s4 + ...) */
        vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * (c4 + ...) */
        vaddpd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + s3 */
        vaddpd   .L__dble_dcos_c3(%rip),%xmm1,%xmm1    /* + c3 */
	vmulpd   %xmm3,%xmm0,%xmm0                     /* x2 * (s3 + ...) */
        vaddpd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + s2 */
        vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * (c3 + ...) */
        vaddpd   .L__dble_dcos_c2(%rip),%xmm1,%xmm1    /* + c2 */
        vmulpd   %xmm3,%xmm0,%xmm0                     /* x2 * (s2 + ...) */
        vaddpd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + s1 */
        vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * (c2 + ...) */
        vaddpd   .L__dble_dcos_c1(%rip),%xmm1,%xmm1    /* + c1 */
#endif

	vmulpd		%xmm3,%xmm0,%xmm0

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm3,%xmm1)
#else
	vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
        vaddpd   .L__dble_pq1+16(%rip),%xmm1,%xmm1     /* - 0.5 */
#endif

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm2,%xmm0)
#	VFMADDPD	.L__real_one(%rip),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(.L__real_one(%rip),%xmm3,%xmm1)
#else */
        vmulpd   %xmm2,%xmm0,%xmm0                     /* x3 * (s1 + ...) */
        vaddpd   %xmm2,%xmm0,%xmm0                     /* x + x3 * (...) done */
        vmulpd   %xmm3,%xmm1,%xmm1                     /* x2 * (0.5 + ...) */
        vaddpd   .L__real_one(%rip),%xmm1,%xmm1        /* 1.0 - 0.5x2 + (...) done */
/* #endif */

#if defined(_WIN64)
        vmovdqu  64(%rsp),%ymm6
        vmovdqu  96(%rsp),%ymm7
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvdsincos1a):
	movq	(%rsp),%rdx
	call	LBL(.L__fvd_sincos_local)
	jmp	LBL(.L__Scalar_fvdsincos5)

LBL(.L__Scalar_fvdsincos2):
        vmovapd  %xmm0, (%rsp)                 /* Save xmm0 */
        vmovapd  %xmm1, %xmm0                  /* Save xmm1 */
        vmovapd  %xmm1, 16(%rsp)               /* Save xmm1 */

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
	CALL(ENT(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

LBL(.L__Scalar_fvdsincos5):
        vmovlpd  %xmm0, (%rsp)
        vmovlpd  %xmm1, 16(%rsp)
        vmovsd  24(%rsp), %xmm0
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
	CALL(ENT(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA)))


LBL(.L__Scalar_fvdsincos8):
        vmovlpd  %xmm0, 8(%rsp)
        vmovlpd  %xmm1, 24(%rsp)
	vmovapd	(%rsp), %xmm0
	vmovapd	16(%rsp), %xmm1
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvd_sincos_local):
        vmovsd  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        vmovapd   %xmm0,%xmm2
        vmovapd   %xmm0,%xmm3
        shrq    $48,%rdx
        cmpl    $0x03f20,%edx
        jl      LBL(.L__fvd_sincos_small)
        vmovapd   %xmm0,%xmm4
        vmulsd   %xmm0,%xmm0,%xmm0                     /* x2 */
        vmulsd   %xmm2,%xmm2,%xmm2                     /* x2 */
        vmulsd   %xmm4,%xmm4,%xmm4                     /* x2 */
        vmulsd   .L__dble_dsin_c6(%rip),%xmm0,%xmm0    /* x2 * s6 */
        vmulsd   .L__dble_dcos_c6(%rip),%xmm2,%xmm2    /* x2 * c6 */
        vaddsd   .L__dble_dsin_c5(%rip),%xmm0,%xmm0    /* + s5 */
        vaddsd   .L__dble_dcos_c5(%rip),%xmm2,%xmm2    /* + c5 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c4(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c4(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c4(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c4(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_dsin_c3(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c3(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c3(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c3(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c2(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c2(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c1(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c1(%rip),%xmm4,%xmm2)
#else
	vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s5 + ...) */
        vaddsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* + s4 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c5 + ...) */
        vaddsd   .L__dble_dcos_c4(%rip),%xmm2,%xmm2    /* + c4 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s4 + ...) */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + s3 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c4 + ...) */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm2,%xmm2    /* + c3 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + s2 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm2,%xmm2    /* + c2 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s2 + ...) */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + s1 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm2,%xmm2    /* + c1 */
#endif
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x3 * (s1 + ...) */
        vmulsd   %xmm3,%xmm0,%xmm0                     /* x3 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm2)
#else
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2     /* - 0.5 */
#endif
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm3,%xmm0,%xmm0                     /* x + x3 * (...) done */
        vaddsd   %xmm2,%xmm1,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
        ret

LBL(.L__fvd_sincos_small):
        cmpl    $0x03e40,%edx
        jl      LBL(.L__fvd_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_pq1(%rip),%xmm3,%xmm3
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm3,%xmm0
	VFMA_231SD	(%xmm2,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,.L__dble_pq1+16(%rip),%xmm2,%xmm1
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#else
        vmulsd   %xmm2,%xmm3,%xmm3
        vaddsd   %xmm3,%xmm0,%xmm0
        vmulsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2
        vaddsd   %xmm2,%xmm1,%xmm1
#endif
        ret

LBL(.L__fvd_sincos_done1):
	rep
        ret

        ELF_FUNC(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA))


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
        .globl ENT(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA)):

	vmovaps	%xmm0, %xmm1		/* Move input vector */
	vandps   .L__sngl_mask_unsign(%rip), %xmm0, %xmm0

	pushq   %rbp
	movq    %rsp, %rbp
	subq    $48, %rsp

	vmovlps  .L__sngl_pi_over_fours(%rip),%xmm2,%xmm2
	vmovhps  .L__sngl_pi_over_fours(%rip),%xmm2,%xmm2
	vmovlps  .L__sngl_needs_argreds(%rip),%xmm3,%xmm3
	vmovhps  .L__sngl_needs_argreds(%rip),%xmm3,%xmm3
	vmovlps  .L__sngl_sixteen_by_pi(%rip),%xmm4,%xmm4
	vmovhps  .L__sngl_sixteen_by_pi(%rip),%xmm4,%xmm4

	vcmpps   $5, %xmm0, %xmm2, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	vcmpps   $2, %xmm0, %xmm3, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
	vmovmskps %xmm2, %eax
	vmovmskps %xmm3, %ecx

	test	$15, %eax
        jnz	LBL(.L__Scalar_fvsin1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
	vmulps   %xmm1,%xmm4,%xmm4

	test	$15, %ecx
        jnz	LBL(.L__Scalar_fvsin2)

        /* Set n = nearest integer to r */
	vmovhps	%xmm1,(%rsp)                     /* Store x4, x3 */

	xorq	%r10, %r10
	vcvtps2pd %xmm1,%xmm1

        vcvtps2dq %xmm4,%xmm5    /* convert to integer, n4,n3,n2,n1 */
LBL(.L__fvsin_do_twice):
	vmovddup   .L__dble_pi_by_16_ms(%rip), %xmm0
	vmovddup   .L__dble_pi_by_16_ls(%rip), %xmm2
	vmovddup   .L__dble_pi_by_16_us(%rip), %xmm3

        vcvtdq2pd	%xmm5,%xmm4    /* and back to double */

	vmovd 		%xmm5, %rcx

        /* r = ((x - n*p1) - (n*p2 + n*p3) */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm1,%xmm4,%xmm0,%xmm1
	VFNMA_231PD	(%xmm4,%xmm0,%xmm1)
#else
	vmulpd   %xmm4,%xmm0,%xmm0	/* n * p1 */
	vsubpd   %xmm0,%xmm1,%xmm1   /* x - n * p1 == rh */
#endif
/*	vmulpd   %xmm4,%xmm2,%xmm2 */	/* n * p2 == rt */
/*	vmulpd   %xmm4,%xmm3,%xmm3 */	/* n * p3 */

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

/*	vsubpd   %xmm0,%xmm1,%xmm1 */   /* x - n * p1 == rh */
	vmulpd   %xmm4,%xmm2,%xmm2	/* n * p2 == rt */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm4,%xmm3,%xmm3
	VFMA_213PD	(%xmm2,%xmm4,%xmm3)
#else
	vmulpd   %xmm4,%xmm3,%xmm3	/* n * p3 */
	vaddpd   %xmm2,%xmm3,%xmm3
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

	vsubpd   %xmm3,%xmm1,%xmm1   /* c = rh - rt aka r */

        shrq    $32, %r9
        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

	vmovapd  %xmm1,%xmm0   /* r in xmm0 and xmm1 */
        vmovapd  %xmm1,%xmm2   /* r in xmm2 */
        vmovapd  %xmm1,%xmm4   /* r in xmm4 */
        vmulpd   %xmm1,%xmm1,%xmm1   /* r^2 in xmm1 */
        vmulpd   %xmm0,%xmm0,%xmm0   /* r^2 in xmm0 */
        vmulpd   %xmm4,%xmm4,%xmm4   /* r^2 in xmm4 */
        vmovapd  %xmm2,%xmm3   /* r in xmm2 and xmm3 */

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
	vmulpd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* p3 * r^2 */
	vmulpd   .L__dble_pq3+16(%rip), %xmm1, %xmm1  /* q3 * r^2 */

        movq    %r8, %rdx     /* Duplicate it */
        sarq    $4,%r8        /* Sign bits moved down */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r8     /* Final tbl address */

	vaddpd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
	vaddpd   .L__dble_pq2+16(%rip), %xmm1, %xmm1  /* + q2 */

        movq    %r9, %rdx     /* Duplicate it */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r9     /* Final tbl address */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm4,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm4,%xmm1)
#else
        vmulpd   %xmm4,%xmm0,%xmm0                   /* * r^2 */
        vaddpd   .L__dble_pq1(%rip), %xmm0,%xmm0     /* + p1 */
        vmulpd   %xmm4,%xmm1,%xmm1                   /* * r^2 */
        vaddpd   .L__dble_pq1+16(%rip), %xmm1,%xmm1  /* + q1 */
#endif

        vmulpd   %xmm4,%xmm3,%xmm3                   /* xmm3 = r^3 */

        addq    %rax,%rax
        addq    %r8,%r8
        addq    %rcx,%rcx
        addq    %r9,%r9

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm3,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm3,%xmm0)
#else
	vmulpd   %xmm3,%xmm0,%xmm0                   /* * r^3 */
	vaddpd   %xmm2,%xmm0,%xmm0                   /* + r = p(r) */
#endif
	vmulpd   %xmm4,%xmm1,%xmm1                   /* * r^2  = q(r) */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        vmovsd  (%rdx,%rax,8),%xmm4           /* S in xmm4 */
        vmovhpd  (%rdx,%r8,8),%xmm4,%xmm4            /* S in xmm4 */

        vmovsd  (%rdx,%rcx,8),%xmm3           /* C in xmm3 */
        vmovhpd  (%rdx,%r9,8),%xmm3,%xmm3            /* C in xmm3 */

/*	vaddpd   %xmm2,%xmm0,%xmm0 */                  /* + r = p(r) */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm4, %xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm4,%xmm1)
#	VFMADDPD	%xmm1,%xmm3, %xmm0,%xmm0
	VFMA_213PD	(%xmm1,%xmm3,%xmm0)
#else
	vmulpd   %xmm4, %xmm1,%xmm1                  /* S * q(r) */
	vaddpd   %xmm4, %xmm1,%xmm1                  /* S + S * q(r) */
	vmulpd   %xmm3, %xmm0,%xmm0                  /* C * p(r) */
	vaddpd   %xmm1, %xmm0,%xmm0                  /* sin(x) = Cp(r) + (S+Sq(r)) */
#endif

	vcvtpd2ps %xmm0,%xmm0
	cmp	$0, %r10                      /* Compare loop count */
	vshufps	$78, %xmm0, %xmm5, %xmm5             /* sin(x2), sin(x1), n4, n3 */
	jne 	LBL(.L__fvsin_done_twice)
	inc 	%r10
	vcvtps2pd (%rsp),%xmm1
	jmp 	LBL(.L__fvsin_do_twice)

LBL(.L__fvsin_done_twice):
	vmovaps  %xmm5, %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsin1):
        /* Here when at least one argument is less than pi/4,
           or, at least one is a Nan.  What we will do for now, is
           if all are less than pi/4, do them all.  Otherwise, call
           fss_sin or mth_i_sin one at a time.
        */
        vmovaps  %xmm0, (%rsp)                 /* Save xmm0, masked x */
	vcmpps   $3, %xmm0, %xmm0, %xmm0       /* 3 is "unordered" */
        vmovaps  %xmm1, 16(%rsp)               /* Save xmm1, input x */
        vmovmskps %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

	cmp	$15, %eax
	jne	LBL(.L__Scalar_fvsin1a)

	vcvtps2pd 16(%rsp),%xmm0               /* x(2), x(1) */
	vcvtps2pd 24(%rsp),%xmm1               /* x(4), x(3) */

        vmovapd  %xmm0,16(%rsp)
        vmovapd  %xmm1,32(%rsp)
	vmulpd   %xmm0,%xmm0,%xmm0                   /* x2 for x(2), x(1) */
	vmulpd   %xmm1,%xmm1,%xmm1                   /* x2 for x(4), x(3) */

	vmovddup  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
	vmovddup  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */


        vmovapd  %xmm0,%xmm2
        vmovapd  %xmm1,%xmm3
#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm4,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm4,%xmm0)
#	VFMADDPD	%xmm5,%xmm4,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm4,%xmm1)
#else
        vmulpd   %xmm4,%xmm0,%xmm0                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm0,%xmm0                   /* + c3 */
        vmulpd   %xmm4,%xmm1,%xmm1                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm1,%xmm1                   /* + c3 */
#endif
	vmovddup  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
        vmovapd  .L__dble_pq1(%rip),%xmm5      /* c1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm4,%xmm2,%xmm0)
#	VFMADDPD	%xmm4,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm3,%xmm1)
#	VFMADDPD	%xmm5,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm2,%xmm0)
#	VFMADDPD	%xmm5,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm3,%xmm1)
	vmulpd		16(%rsp),%xmm2,%xmm2                /* x3 */
        vmulpd		32(%rsp),%xmm3,%xmm3                /* x3 */
#	VFMADDPD	16(%rsp),%xmm2,%xmm0,%xmm0
	VFMA_213PD	(16(%rsp),%xmm2,%xmm0)
#	VFMADDPD	32(%rsp),%xmm3,%xmm1,%xmm1
	VFMA_213PD	(32(%rsp),%xmm3,%xmm1)
#else
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c3 + ...) */
        vaddpd   %xmm4,%xmm0,%xmm0                   /* + c2 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c3 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1                   /* + c2 */
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm0,%xmm0                   /* + c1 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm1,%xmm1                   /* + c1 */
	vmulpd   16(%rsp),%xmm2,%xmm2                /* x3 */
	vmulpd   32(%rsp),%xmm3,%xmm3                /* x3 */
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x3 * (c1 + ...) */
        vaddpd   16(%rsp),%xmm0,%xmm0                /* x + x3 * (...) done */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x3 * (c1 + ...) */
        vaddpd   32(%rsp),%xmm1,%xmm1                /* x + x3 * (...) done */
#endif
        vcvtpd2ps %xmm0,%xmm0            /* sin(x2), sin(x1) */
        vcvtpd2ps %xmm1,%xmm1            /* sin(x4), sin(x3) */
	vshufps	$68, %xmm1, %xmm0, %xmm0       /* sin(x4),sin(x3),sin(x2),sin(x1) */

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsin1a):
	test    $1, %eax
	jz	LBL(.L__Scalar_fvsin3)
	vmovss 16(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	(%rsp),%edx
	call	LBL(.L__fvs_sin_local)
	jmp	LBL(.L__Scalar_fvsin5)

LBL(.L__Scalar_fvsin2):
        vmovaps  %xmm0, (%rsp)                 /* Save xmm0 */
        vmovaps  %xmm1, %xmm0                  /* Save xmm1 */
        vmovaps  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvsin3):
	vmovss   16(%rsp),%xmm0                /* x(1) */
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
	CALL(ENT(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsin5):
        vmovss   %xmm0, (%rsp)                 /* Move first result */

	test    $2, %eax
	jz	LBL(.L__Scalar_fvsin6)
	vmovss 20(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	4(%rsp),%edx
	call	LBL(.L__fvs_sin_local)
	jmp	LBL(.L__Scalar_fvsin8)

LBL(.L__Scalar_fvsin6):
	vmovss   20(%rsp),%xmm0                /* x(2) */
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
	CALL(ENT(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsin8):
        vmovss   %xmm0, 4(%rsp)               /* Move 2nd result */

	test    $4, %eax
	jz	LBL(.L__Scalar_fvsin9)
	vmovss 24(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	8(%rsp),%edx
	call	LBL(.L__fvs_sin_local)
	jmp	LBL(.L__Scalar_fvsin11)

LBL(.L__Scalar_fvsin9):
	vmovss   24(%rsp),%xmm0                /* x(3) */
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
	CALL(ENT(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsin11):
        vmovss   %xmm0, 8(%rsp)               /* Move 3rd result */

	test    $8, %eax
	jz	LBL(.L__Scalar_fvsin12)
	vmovss 28(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	12(%rsp),%edx
	call	LBL(.L__fvs_sin_local)
	jmp	LBL(.L__Scalar_fvsin14)

LBL(.L__Scalar_fvsin12):
	vmovss   28(%rsp),%xmm0                /* x(4) */
	test    $8, %ecx
	jz	LBL(.L__Scalar_fvsin13)
	CALL(ENT(__mth_i_sin))
	jmp	LBL(.L__Scalar_fvsin14)

LBL(.L__Scalar_fvsin13):
	CALL(ENT(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA)))


/* ---------------------------------- */
LBL(.L__Scalar_fvsin14):
        vmovss   %xmm0, 12(%rsp)               /* Move 4th result */
	vmovaps	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvs_sin_local):
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2
        shrl    $20,%edx
        cmpl    $0x0390,%edx
        jl      LBL(.L__fvs_sin_small)
        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* x2 * c4 */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + c3 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dble_pq1(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm1,%xmm0)
	vmulsd		%xmm2,%xmm1,%xmm1
#	VFMADDSD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#else
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + c2 */
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + c1 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x3 */
        vmulsd   %xmm1,%xmm0,%xmm0                     /* x3 * (c1 + ...) */
        vaddsd   %xmm2,%xmm0,%xmm0                     /* x + x3 * (...) done */
#endif
	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

LBL(.L__fvs_sin_small):
        cmpl    $0x0320,%edx
        jl      LBL(.L__fvs_sin_done1)
        /* return x - x * x * x * 1/3! */
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   .L__dble_pq1(%rip),%xmm2,%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm1,%xmm0
	VFMA_231SD	(%xmm2,%xmm1,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif

LBL(.L__fvs_sin_done1):
	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

        ALN_QUAD

        ELF_FUNC(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA))

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
        .globl ENT(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA)):

	vmovaps	%xmm0, %xmm1		/* Move input vector */
        vandps   .L__sngl_mask_unsign(%rip), %xmm0, %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $48, %rsp

        vmovlps  .L__sngl_pi_over_fours(%rip),%xmm2,%xmm2
        vmovhps  .L__sngl_pi_over_fours(%rip),%xmm2,%xmm2
        vmovlps  .L__sngl_needs_argreds(%rip),%xmm3,%xmm3
        vmovhps  .L__sngl_needs_argreds(%rip),%xmm3,%xmm3
        vmovlps  .L__sngl_sixteen_by_pi(%rip),%xmm4,%xmm4
        vmovhps  .L__sngl_sixteen_by_pi(%rip),%xmm4,%xmm4

	vcmpps   $5, %xmm0, %xmm2,%xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	vcmpps   $2, %xmm0, %xmm3, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        vmovmskps %xmm2, %eax
        vmovmskps %xmm3, %ecx

	test	$15, %eax
        jnz	LBL(.L__Scalar_fvcos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulps   %xmm1,%xmm4,%xmm4

	test	$15, %ecx
        jnz	LBL(.L__Scalar_fvcos2)

        /* Set n = nearest integer to r */
	vmovhps	%xmm1,(%rsp)                     /* Store x4, x3 */

	xorq	%r10, %r10
	vcvtps2pd %xmm1, %xmm1

        vcvtps2dq %xmm4,%xmm5    /* convert to integer, n4,n3,n2,n1 */
LBL(.L__fvcos_do_twice):
	vmovddup   .L__dble_pi_by_16_ms(%rip), %xmm0
	vmovddup   .L__dble_pi_by_16_ls(%rip), %xmm2
	vmovddup   .L__dble_pi_by_16_us(%rip), %xmm3

        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - (n*p2 + n*p3) */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm1,%xmm4,%xmm0,%xmm1
	VFNMA_231PD	(%xmm4,%xmm0,%xmm1)
#else
	vmulpd   %xmm4,%xmm0,%xmm0     /* n * p1 */
	vsubpd   %xmm0,%xmm1,%xmm1    /* x - n * p1 == rh */
#endif
        vmulpd   %xmm4,%xmm3,%xmm3   /* n * p3 */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm3,%xmm4,%xmm2,%xmm3
	VFMA_231PD	(%xmm4,%xmm2,%xmm3)
#else
	vmulpd   %xmm4,%xmm2,%xmm2   /* n * p2 == rt */
	vaddpd   %xmm2,%xmm3,%xmm3
#endif
        vsubpd   %xmm3,%xmm1,%xmm1   /* c = rh - rt aka r */

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

/*	vsubpd   %xmm0,%xmm1,%xmm1 */   /* x - n * p1 == rh */
/*	vaddpd   %xmm2,%xmm3,%xmm3 */

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

/*	vsubpd   %xmm3,%xmm1,%xmm1 */   /* c = rh - rt aka r */

        shrq    $32, %r9
        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

	vmovapd  %xmm1,%xmm0   /* r in xmm0 and xmm1 */
        vmovapd  %xmm1,%xmm2   /* r in xmm2 */
        vmovapd  %xmm1,%xmm4   /* r in xmm4 */
        vmulpd   %xmm1,%xmm1,%xmm1   /* r^2 in xmm1 */
        vmulpd   %xmm0,%xmm0,%xmm0   /* r^2 in xmm0 */
        vmulpd   %xmm4,%xmm4,%xmm4   /* r^2 in xmm4 */
        vmovapd  %xmm2,%xmm3   /* r in xmm2 and xmm3 */

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
        vmulpd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* p3 * r^2 */
        vmulpd   .L__dble_pq3+16(%rip), %xmm1, %xmm1  /* q3 * r^2 */

        movq    %r8, %rdx     /* Duplicate it */
        sarq    $4,%r8        /* Sign bits moved down */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r8     /* Final tbl address */

        vaddpd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vaddpd   .L__dble_pq2+16(%rip), %xmm1, %xmm1  /* + q2 */

        movq    %r9, %rdx     /* Duplicate it */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r9     /* Final tbl address */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm4,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm4,%xmm1)
#else
        vmulpd   %xmm4,%xmm0,%xmm0                   /* * r^2 */
        vaddpd   .L__dble_pq1(%rip), %xmm0,%xmm0     /* + p1 */
        vmulpd   %xmm4,%xmm1,%xmm1                   /* * r^2 */
        vaddpd   .L__dble_pq1+16(%rip), %xmm1,%xmm1  /* + q1 */
#endif

        vmulpd   %xmm4,%xmm3,%xmm3                   /* xmm3 = r^3 */

        addq    %rax,%rax
        addq    %r8,%r8
        addq    %rcx,%rcx
        addq    %r9,%r9

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm3,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm3,%xmm0)
#else
	vmulpd   %xmm3,%xmm0,%xmm0                   /* * r^3 */
	vaddpd   %xmm2,%xmm0,%xmm0                   /* + r = p(r) */
#endif
	vmulpd   %xmm4,%xmm1,%xmm1                   /* * r^2  = q(r) */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        vmovsd  (%rdx,%rcx,8),%xmm3           /* C in xmm3 */
        vmovhpd  (%rdx,%r9,8),%xmm3,%xmm3            /* C in xmm3 */

        vmovsd  (%rdx,%rax,8),%xmm4           /* S in xmm4 */
        vmovhpd  (%rdx,%r8,8),%xmm4,%xmm4            /* S in xmm4 */

/*	vaddpd   %xmm2,%xmm0,%xmm0 */                  /* + r = p(r) */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm3,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm3,%xmm3,%xmm1)
#	VFNMADDPD	%xmm1,%xmm4,%xmm0,%xmm1
	VFNMA_231PD	(%xmm4,%xmm0,%xmm1)
#else
	vmulpd   %xmm3, %xmm1,%xmm1                  /* C * q(r) */
	vaddpd   %xmm3, %xmm1,%xmm1                  /* C + C * q(r) */
	vmulpd   %xmm4, %xmm0,%xmm0                  /* S * p(r) */
	vsubpd   %xmm0, %xmm1,%xmm1                  /* cos(x) = (C+Cq(r)) - Sp(r) */
#endif

	vcvtpd2ps %xmm1,%xmm0
	cmp	$0, %r10                      /* Compare loop count */
	vshufps	$78, %xmm0, %xmm5, %xmm5             /* sin(x2), sin(x1), n4, n3 */
	jne 	LBL(.L__fvcos_done_twice)
	inc 	%r10
	vcvtps2pd (%rsp),%xmm1
	jmp 	LBL(.L__fvcos_do_twice)

LBL(.L__fvcos_done_twice):
	vmovaps  %xmm5, %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvcos1):
        /* Here when at least one argument is less than pi/4,
           or, at least one is a Nan.  What we will do for now, is
           if all are less than pi/4, do them all.  Otherwise, call
           fss_cos or mth_i_cos one at a time.
        */
        vmovaps  %xmm0, (%rsp)                 /* Save xmm0, masked x */
	vcmpps   $3, %xmm0, %xmm0, %xmm0              /* 3 is "unordered" */
        vmovaps  %xmm1, 16(%rsp)               /* Save xmm1, input x */
        vmovmskps %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

	cmp	$15, %eax
	jne	LBL(.L__Scalar_fvcos1a)

	vcvtps2pd 16(%rsp),%xmm0               /* x(2), x(1) */
	vcvtps2pd 24(%rsp),%xmm1               /* x(4), x(3) */

        vmovapd  %xmm0,16(%rsp)
        vmovapd  %xmm1,32(%rsp)
	vmulpd   %xmm0,%xmm0,%xmm0                   /* x2 for x(2), x(1) */
	vmulpd   %xmm1,%xmm1,%xmm1                   /* x2 for x(4), x(3) */

	vmovddup  .L__dble_dcos_c4(%rip),%xmm4  /* c4 */
	vmovddup  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */


        vmovapd  %xmm0,%xmm2
        vmovapd  %xmm1,%xmm3
#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm4,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm4,%xmm0)
#	VFMADDPD	%xmm5,%xmm4,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm4,%xmm1)
#else
	vmulpd   %xmm4,%xmm0,%xmm0                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm0,%xmm0                   /* + c3 */
        vmulpd   %xmm4,%xmm1,%xmm1                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm1,%xmm1                   /* + c3 */
#endif
	vmovddup  .L__dble_dcos_c2(%rip),%xmm4  /* c2 */
	vmovddup  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm4,%xmm2,%xmm0)
#	VFMADDPD	%xmm4,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm3,%xmm1)
	vmovapd		.L__dble_pq1+16(%rip),%xmm4
#	VFMADDPD	%xmm5,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm2,%xmm0)
#	VFMADDPD	%xmm5,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm3,%xmm1)
	vmovapd		.L__real_one(%rip), %xmm5
#	VFMADDPD	%xmm4,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm4,%xmm2,%xmm0)
#	VFMADDPD	%xmm4,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm3,%xmm1)
#	VFMADDPD	%xmm5,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm2,%xmm0)
#	VFMADDPD	%xmm5,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm3,%xmm1)
#else
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c3 + ...) */
        vaddpd   %xmm4,%xmm0,%xmm0                   /* + c2 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c3 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1                   /* + c2 */

        vmovapd  .L__dble_pq1+16(%rip),%xmm4   /* -0.5 */
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm0,%xmm0                   /* + c1 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm1,%xmm1                   /* + c1 */

	vmovapd  .L__real_one(%rip), %xmm5     /* 1.0 */
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c1 + ...) */
        vaddpd   %xmm4,%xmm0,%xmm0                   /* -0.5 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c1 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1                   /* -0.5 */
        vmulpd   %xmm2,%xmm0,%xmm0                   /* - x2 * (0.5 + ...) */
	vaddpd   %xmm5,%xmm0,%xmm0                   /* 1.0 - 0.5x2 + (...) done */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* - x2 * (0.5 + ...) */
	vaddpd   %xmm5,%xmm1,%xmm1                   /* 1.0 - 0.5x2 + (...) done */
#endif

        vcvtpd2ps %xmm0,%xmm0            /* cos(x2), cos(x1) */
        vcvtpd2ps %xmm1,%xmm1            /* cos(x4), cos(x3) */
	vshufps	$68, %xmm1, %xmm0,%xmm0       /* cos(x4),cos(x3),cos(x2),cos(x1) */

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvcos1a):
	test    $1, %eax
	jz	LBL(.L__Scalar_fvcos3)
	vmovss 16(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	(%rsp),%edx
	call	LBL(.L__fvs_cos_local)
	jmp	LBL(.L__Scalar_fvcos5)

LBL(.L__Scalar_fvcos2):
        vmovaps  %xmm0, (%rsp)                 /* Save xmm0 */
        vmovaps  %xmm1, %xmm0                  /* Save xmm1 */
        vmovaps  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvcos3):
	vmovss   16(%rsp),%xmm0                /* x(1) */
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
	CALL(ENT(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvcos5):
        vmovss   %xmm0, (%rsp)                 /* Move first result */

	test    $2, %eax
	jz	LBL(.L__Scalar_fvcos6)
	vmovss 20(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	4(%rsp),%edx
	call	LBL(.L__fvs_cos_local)
	jmp	LBL(.L__Scalar_fvcos8)

LBL(.L__Scalar_fvcos6):
	vmovss   20(%rsp),%xmm0                /* x(2) */
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
	CALL(ENT(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvcos8):
        vmovss   %xmm0, 4(%rsp)               /* Move 2nd result */

	test    $4, %eax
	jz	LBL(.L__Scalar_fvcos9)
	vmovss 24(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	8(%rsp),%edx
	call	LBL(.L__fvs_cos_local)
	jmp	LBL(.L__Scalar_fvcos11)

LBL(.L__Scalar_fvcos9):
	vmovss   24(%rsp),%xmm0                /* x(3) */
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
	CALL(ENT(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvcos11):
        vmovss   %xmm0, 8(%rsp)               /* Move 3rd result */

	test    $8, %eax
	jz	LBL(.L__Scalar_fvcos12)
	vmovss 28(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	12(%rsp),%edx
	call	LBL(.L__fvs_cos_local)
	jmp	LBL(.L__Scalar_fvcos14)

LBL(.L__Scalar_fvcos12):
	vmovss   28(%rsp),%xmm0                /* x(4) */
	test    $8, %ecx
	jz	LBL(.L__Scalar_fvcos13)
	CALL(ENT(__mth_i_cos))
	jmp	LBL(.L__Scalar_fvcos14)

LBL(.L__Scalar_fvcos13):
	CALL(ENT(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA)))


/* ---------------------------------- */
LBL(.L__Scalar_fvcos14):
        vmovss   %xmm0, 12(%rsp)               /* Move 4th result */
	vmovaps	(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvs_cos_local):
        vmovapd	%xmm0,%xmm1
        vmovapd	%xmm0,%xmm2
        shrl    $20,%edx
	vmovsd	.L__dble_sincostbl(%rip), %xmm0 /* 1.0 */
        cmpl    $0x0390,%edx
        jl      LBL(.L__fvs_cos_small)
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_dcos_c4(%rip),%xmm1,%xmm1    /* x2 * c4 */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm1,%xmm1    /* + c3 */
#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_dcos_c2(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c2(%rip),%xmm2,%xmm1)
#	VFMADDPD	.L__dble_dcos_c1(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_dcos_c1(%rip),%xmm2,%xmm1)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#	VFMADDPD	%xmm0,%xmm2,%xmm1,%xmm0
	VFMA_231PD	(%xmm2,%xmm1,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm1,%xmm1    /* + c2 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm1,%xmm1    /* + c1 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
	vaddsd	.L__dble_pq1+16(%rip),%xmm1,%xmm1     /* -0.5 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
        vaddsd   %xmm1,%xmm0,%xmm0                     /* x + x3 * (...) done */
#endif
	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

LBL(.L__fvs_cos_small):
        cmpl    $0x0320,%edx
        jl      LBL(.L__fvs_cos_done1)
        /* return 1.0 - x * x * 0.5 */
        vmulsd   %xmm1,%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDPD	%xmm0,.L__dble_pq1+16(%rip),%xmm1,%xmm0
	VFMA_231PD	(.L__dble_pq1+16(%rip),%xmm1,%xmm0)
#else
        vmulsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif

LBL(.L__fvs_cos_done1):
	vunpcklpd %xmm0, %xmm0,%xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

        ELF_FUNC(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA))


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
        .globl ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)):

	vmovaps	%xmm0, %xmm1		/* Move input vector */
        vandps   .L__sngl_mask_unsign(%rip), %xmm0, %xmm0

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

	vmovlps  .L__sngl_pi_over_fours(%rip),%xmm2,%xmm2
	vmovhps  .L__sngl_pi_over_fours(%rip),%xmm2,%xmm2
	vmovlps  .L__sngl_needs_argreds(%rip),%xmm3,%xmm3
	vmovhps  .L__sngl_needs_argreds(%rip),%xmm3,%xmm3
	vmovlps  .L__sngl_sixteen_by_pi(%rip),%xmm4,%xmm4
	vmovhps  .L__sngl_sixteen_by_pi(%rip),%xmm4,%xmm4

	vcmpps   $5, %xmm0, %xmm2, %xmm2  /* 5 is "not less than" */
                                  /* pi/4 is not less than abs(x) */
                                  /* true if pi/4 >= abs(x) */
                                  /* also catches nans */

	vcmpps   $2, %xmm0, %xmm3, %xmm3  /* 2 is "less than or equal */
                                  /* 0x413... less than or equal to abs(x) */
                                  /* true if 0x413 is <= abs(x) */
        vmovmskps %xmm2, %eax
        vmovmskps %xmm3, %ecx

	test	$15, %eax
        jnz	LBL(.L__Scalar_fvsincos1)

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulps   %xmm1,%xmm4,%xmm4

	test	$15, %ecx
        jnz	LBL(.L__Scalar_fvsincos2)

        /* Set n = nearest integer to r */
	vmovhps	%xmm1,(%rsp)                     /* Store x4, x3 */

#if defined(_WIN64)
        vmovdqu  %ymm6, 64(%rsp)
        vmovdqu  %ymm7, 96(%rsp)
#endif
	xorq	%r10, %r10
	vcvtps2pd %xmm1, %xmm6
	vxorps	%xmm7, %xmm7, %xmm7

        vcvtps2dq %xmm4,%xmm5    /* convert to integer, n4,n3,n2,n1 */

LBL(.L__fvsincos_do_twice):
	vmovddup   .L__dble_pi_by_16_ms(%rip), %xmm0
	vmovddup   .L__dble_pi_by_16_ls(%rip), %xmm2
	vmovddup   .L__dble_pi_by_16_us(%rip), %xmm3

        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - (n*p2 + n*p3) */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm6,%xmm4,%xmm0,%xmm6
	VFNMA_231PD	(%xmm4,%xmm0,%xmm6)
#else
	vmulpd   %xmm4,%xmm0,%xmm0	/* n * p1 */
        vsubpd   %xmm0,%xmm6,%xmm6   /* x - n * p1 == rh */
#endif
/*	vmulpd   %xmm4,%xmm2,%xmm2 */	/* n * p2 == rt */
	vmulpd   %xmm4,%xmm3,%xmm3	/* n * p3 */

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

/*	vsubpd   %xmm0,%xmm6,%xmm6 */   /* x - n * p1 == rh */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm3,%xmm4,%xmm2,%xmm3
	VFMA_231PD	(%xmm4,%xmm2,%xmm3)
#else
	vmulpd   %xmm4,%xmm2,%xmm2	/* n * p2 == rt */
	vaddpd   %xmm2,%xmm3,%xmm3
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        vsubpd   %xmm3,%xmm6,%xmm6   /* c = rh - rt aka r */

        shrq    $32, %r9
        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

	vmovapd  %xmm6,%xmm0   /* r in xmm0 and xmm6 */
        vmovapd  %xmm6,%xmm2   /* r in xmm2 */
        vmovapd  %xmm6,%xmm4   /* r in xmm4 */
        vmulpd   %xmm6,%xmm6,%xmm6	/* r^2 in xmm6 */
        vmulpd   %xmm0,%xmm0,%xmm0	/* r^2 in xmm0 */
        vmulpd   %xmm4,%xmm4,%xmm4	/* r^2 in xmm4 */
        vmovapd  %xmm2,%xmm3   /* r in xmm2 and xmm3 */

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
        vmulpd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* p3 * r^2 */
        vmulpd   .L__dble_pq3+16(%rip), %xmm6, %xmm6  /* q3 * r^2 */

        movq    %r8, %rdx     /* Duplicate it */
        sarq    $4,%r8        /* Sign bits moved down */
        xorq    %r8, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r8        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r8     /* Final tbl address */

        vaddpd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vaddpd   .L__dble_pq2+16(%rip), %xmm6, %xmm6  /* + q2 */

        movq    %r9, %rdx     /* Duplicate it */
        sarq    $4,%r9        /* Sign bits moved down */
        xorq    %r9, %rdx     /* Xor bits, backwards over half the cycle */
        sarq    $4,%r9        /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %r9     /* Final tbl address */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm4,%xmm6,%xmm6
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm4,%xmm6)
#else
	vmulpd   %xmm4,%xmm0,%xmm0                    /* * r^2 */
        vaddpd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
        vmulpd   %xmm4,%xmm6,%xmm6                   /* * r^2 */
        vaddpd   .L__dble_pq1+16(%rip), %xmm6, %xmm6  /* + q1 */
#endif

        vmulpd   %xmm4,%xmm3,%xmm3                   /* xmm3 = r^3 */

        addq    %rax,%rax
        addq    %r8,%r8
        addq    %rcx,%rcx
        addq    %r9,%r9

#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm3,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm3,%xmm0)
#else
	vmulpd   %xmm3,%xmm0,%xmm0                   /* * r^3 */
	vaddpd   %xmm2,%xmm0,%xmm0                    /* + r = p(r) */
#endif
	vmulpd   %xmm4,%xmm6,%xmm6                   /* * r^2  = q(r) */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        vmovsd  (%rdx,%rax,8),%xmm4           /* S in xmm4 */
        vmovhpd  (%rdx,%r8,8),%xmm4,%xmm4            /* S in xmm4 */

        vmovsd  (%rdx,%rcx,8),%xmm3           /* C in xmm3 */
        vmovhpd  (%rdx,%r9,8),%xmm3,%xmm3            /* C in xmm3 */

	vmovapd  %xmm6,%xmm1                   /* Move for cosine */
	vmovapd  %xmm0,%xmm2                   /* Move for sine */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213PD	(%xmm4,%xmm4,%xmm6)
#	VFMADDPD	%xmm3,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm3,%xmm3,%xmm1)
#	VFNMADDPD	%xmm1,%xmm4,%xmm2,%xmm1
	VFNMA_231PD	(%xmm4,%xmm2,%xmm1)
#	VFMADDPD	%xmm6,%xmm3,%xmm0,%xmm0
	VFMA_213PD	(%xmm6,%xmm3,%xmm0)
#else
	vmulpd   %xmm4, %xmm6,%xmm6                  /* S * q(r) */
	vaddpd   %xmm4, %xmm6,%xmm6                  /* S + S * q(r) */
	vmulpd   %xmm3, %xmm1,%xmm1                  /* C * q(r) */
	vaddpd   %xmm3, %xmm1,%xmm1                  /* C + C * q(r) */
	vmulpd   %xmm4, %xmm2,%xmm2                  /* S * p(r) */
	vsubpd   %xmm2, %xmm1,%xmm1                  /* cos(x) = (C+Cq(r)) - Sp(r) */
	vmulpd   %xmm3, %xmm0,%xmm0                   /* C * p(r) */
	vaddpd   %xmm6, %xmm0,%xmm0                  /* sin(x) = Cp(r) + (S+Sq(r)) */
#endif

	vcvtpd2ps %xmm0,%xmm0
	vcvtpd2ps %xmm1,%xmm1
	cmp	$0, %r10                      /* Compare loop count */
	vshufps	$78, %xmm0, %xmm5,%xmm5             /* sin(x2), sin(x1), n4, n3 */
	vshufps	$78, %xmm1, %xmm7,%xmm7             /* cos(x2), cos(x1), 0, 0 */
	jne 	LBL(.L__fvsincos_done_twice)
	inc 	%r10
	vcvtps2pd (%rsp),%xmm6
	jmp 	LBL(.L__fvsincos_do_twice)

LBL(.L__fvsincos_done_twice):
	vmovaps  %xmm5, %xmm0
	vmovaps  %xmm7, %xmm1

#if defined(_WIN64)
        vmovdqu  64(%rsp), %ymm6
        vmovdqu  96(%rsp), %ymm7
#endif
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsincos1):
        /* Here when at least one argument is less than pi/4,
           or, at least one is a Nan.  What we will do for now, is
           if all are less than pi/4, do them all.  Otherwise, call
           fss_sincos or mth_i_sincos one at a time.
        */
        vmovaps  %xmm0, (%rsp)                 /* Save xmm0, masked x */
	vcmpps   $3, %xmm0, %xmm0, %xmm0              /* 3 is "unordered" */
        vmovaps  %xmm1, 16(%rsp)               /* Save xmm1, input x */
        vmovmskps %xmm0, %edx                  /* Move mask bits */

        xor	%edx, %eax
        or      %edx, %ecx

	cmp	$15, %eax
	jne	LBL(.L__Scalar_fvsincos1a)

	vcvtps2pd 16(%rsp),%xmm0               /* x(2), x(1) */
	vcvtps2pd 24(%rsp),%xmm1               /* x(4), x(3) */

#if defined(_WIN64)
        vmovdqu  %ymm6, 96(%rsp)
#endif
        vmovapd  %xmm0,16(%rsp)
        vmovapd  %xmm1,32(%rsp)
	vmulpd   %xmm0,%xmm0,%xmm0                   /* x2 for x(2), x(1) */
	vmulpd   %xmm1,%xmm1,%xmm1                   /* x2 for x(4), x(3) */

	vmovddup  .L__dble_dsin_c4(%rip),%xmm4  /* c4 */
	vmovddup  .L__dble_dsin_c3(%rip),%xmm5  /* c3 */


        vmovapd  %xmm0,%xmm2
        vmovapd  %xmm1,%xmm3

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm4,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm4,%xmm0)
#	VFMADDPD	%xmm5,%xmm4,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm4,%xmm1)
#else
        vmulpd   %xmm4,%xmm0,%xmm0                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm0,%xmm0                   /* + c3 */
        vmulpd   %xmm4,%xmm1,%xmm1                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm1,%xmm1                   /* + c3 */
#endif

	vmovddup  .L__dble_dsin_c2(%rip),%xmm4  /* c2 */
        vmovapd  .L__dble_pq1(%rip),%xmm5      /* c1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm4,%xmm2,%xmm0)
#	VFMADDPD	%xmm4,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm3,%xmm1)
#else
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c3 + ...) */
        vaddpd   %xmm4,%xmm0,%xmm0                   /* + c2 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c3 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1                   /* + c2 */
#endif

	vmovapd  16(%rsp), %xmm6               /* x */
	vmovapd  32(%rsp), %xmm4               /* x */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm2,%xmm0)
#	VFMADDPD	%xmm5,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm5,%xmm3,%xmm1)
	vmulpd		%xmm6,%xmm2,%xmm2
        vmulpd		%xmm4,%xmm3,%xmm3
#	VFMADDPD	%xmm6,%xmm2,%xmm0,%xmm0
	VFMA_213PD	(%xmm6,%xmm2,%xmm0)
#	VFMADDPD	%xmm4,%xmm3,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm3,%xmm1)
#else
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm0,%xmm0                   /* + c1 */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm1,%xmm1                   /* + c1 */
	vmulpd   %xmm6,%xmm2,%xmm2                   /* x3 */
	vmulpd   %xmm4,%xmm3,%xmm3                   /* x3 */
        vmulpd   %xmm2,%xmm0,%xmm0                   /* x3 * (c1 + ...) */
        vaddpd   %xmm6,%xmm0,%xmm0             /* x + x3 * (...) done */
        vmulpd   %xmm3,%xmm1,%xmm1                   /* x3 * (c1 + ...) */
        vaddpd   %xmm4,%xmm1,%xmm1             /* x + x3 * (...) done */
#endif

	vmulpd   %xmm6,%xmm6,%xmm6                   /* x2 for x(2), x(1) */
	vmulpd   %xmm4,%xmm4,%xmm4                   /* x2 for x(4), x(3) */

        vcvtpd2ps %xmm0,%xmm0            /* sin(x2), sin(x1) */
        vcvtpd2ps %xmm1,%xmm1            /* sin(x4), sin(x3) */
	vshufps	$68, %xmm1, %xmm0,%xmm0       /* sin(x4),sin(x3),sin(x2),sin(x1) */

	vmovddup  .L__dble_dcos_c4(%rip),%xmm1  /* c4 */
	vmovddup  .L__dble_dcos_c3(%rip),%xmm5  /* c3 */

        vmovapd  %xmm6,%xmm2
        vmovapd  %xmm4,%xmm3

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm1,%xmm6,%xmm6
	VFMA_213PD	(%xmm5,%xmm1,%xmm6)
#	VFMADDPD	%xmm5,%xmm1,%xmm4,%xmm4
	VFMA_213PD	(%xmm5,%xmm1,%xmm4)
#else
        vmulpd   %xmm1,%xmm6,%xmm6                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm6,%xmm6                   /* + c3 */
        vmulpd   %xmm1,%xmm4,%xmm4                   /* x2 * c4 */
        vaddpd   %xmm5,%xmm4,%xmm4                   /* + c3 */
#endif

	vmovddup  .L__dble_dcos_c2(%rip),%xmm1  /* c2 */
	vmovddup  .L__dble_dcos_c1(%rip),%xmm5  /* c1 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,%xmm2,%xmm6,%xmm6
	VFMA_213PD	(%xmm1,%xmm2,%xmm6)
#	VFMADDPD	%xmm1,%xmm3,%xmm4,%xmm4
	VFMA_213PD	(%xmm1,%xmm3,%xmm4)
#	VFMADDPD	%xmm5,%xmm2,%xmm6,%xmm6
	VFMA_213PD	(%xmm5,%xmm2,%xmm6)
#	VFMADDPD	%xmm5,%xmm3,%xmm4,%xmm4
	VFMA_213PD	(%xmm5,%xmm3,%xmm4)
#else
        vmulpd   %xmm2,%xmm6,%xmm6                   /* x2 * (c3 + ...) */
        vaddpd   %xmm1,%xmm6,%xmm6                   /* + c2 */
        vmulpd   %xmm3,%xmm4,%xmm4                   /* x2 * (c3 + ...) */
        vaddpd   %xmm1,%xmm4,%xmm4                   /* + c2 */
        vmulpd   %xmm2,%xmm6,%xmm6                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm6,%xmm6                   /* + c1 */
        vmulpd   %xmm3,%xmm4,%xmm4                   /* x2 * (c2 + ...) */
        vaddpd   %xmm5,%xmm4,%xmm4                   /* + c1 */
#endif

        vmovapd  .L__dble_pq1+16(%rip),%xmm1   /* -0.5 */
        vmovapd  .L__real_one(%rip), %xmm5     /* 1.0 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,%xmm2,%xmm6,%xmm6
	VFMA_213PD	(%xmm1,%xmm2,%xmm6)
#	VFMADDPD	%xmm1,%xmm3,%xmm4,%xmm4
	VFMA_213PD	(%xmm1,%xmm3,%xmm4)
#	VFMADDPD	%xmm5,%xmm2,%xmm6,%xmm6
	VFMA_213PD	(%xmm5,%xmm2,%xmm6)
#	VFMADDPD	%xmm5,%xmm3,%xmm4,%xmm4
	VFMA_213PD	(%xmm5,%xmm3,%xmm4)
#else
        vmulpd   %xmm2,%xmm6,%xmm6                   /* x2 * (c1 + ...) */
        vaddpd   %xmm1,%xmm6,%xmm6                   /* -0.5 */
        vmulpd   %xmm3,%xmm4,%xmm4                   /* x2 * (c1 + ...) */
        vaddpd   %xmm1,%xmm4,%xmm4                   /* -0.5 */
        vmulpd   %xmm2,%xmm6,%xmm6                   /* - x2 * (0.5 + ...) */
        vaddpd   %xmm5,%xmm6,%xmm6                   /* 1.0 - 0.5x2 + (...) done */
        vmulpd   %xmm3,%xmm4,%xmm4                   /* - x2 * (0.5 + ...) */
        vaddpd   %xmm5,%xmm4,%xmm4                   /* 1.0 - 0.5x2 + (...) done */
#endif

        vcvtpd2ps %xmm6,%xmm1            /* cos(x2), cos(x1) */
        vcvtpd2ps %xmm4,%xmm4            /* cos(x4), cos(x3) */
        vshufps  $68, %xmm4, %xmm1,%xmm1       /* cos(x4),cos(x3),cos(x2),cos(x1) */

#if defined(_WIN64)
        vmovdqu  96(%rsp), %ymm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__Scalar_fvsincos1a):
	test    $1, %eax
	jz	LBL(.L__Scalar_fvsincos3)
	vmovss 16(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos5)

LBL(.L__Scalar_fvsincos2):
        vmovaps  %xmm0, (%rsp)                 /* Save xmm0 */
        vmovaps  %xmm1, %xmm0                  /* Save xmm1 */
        vmovaps  %xmm1, 16(%rsp)               /* Save xmm1 */

LBL(.L__Scalar_fvsincos3):
	vmovss   16(%rsp),%xmm0                /* x(1) */
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
	CALL(ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos5):
        vmovss   %xmm0, (%rsp)                 /* Move first result */
        vmovss   %xmm1, 16(%rsp)               /* Move first result */

	test    $2, %eax
	jz	LBL(.L__Scalar_fvsincos6)
	vmovss 20(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	4(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos8)

LBL(.L__Scalar_fvsincos6):
	vmovss   20(%rsp),%xmm0                /* x(2) */
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
	CALL(ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos8):
        vmovss   %xmm0, 4(%rsp)               /* Move 2nd result */
        vmovss   %xmm1, 20(%rsp)              /* Move 2nd result */

	test    $4, %eax
	jz	LBL(.L__Scalar_fvsincos9)
	vmovss 24(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	8(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos11)

LBL(.L__Scalar_fvsincos9):
	vmovss   24(%rsp),%xmm0                /* x(3) */
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
	CALL(ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA)))

	mov     36(%rsp), %ecx
	mov     32(%rsp), %eax

/* ---------------------------------- */
LBL(.L__Scalar_fvsincos11):
        vmovss   %xmm0, 8(%rsp)               /* Move 3rd result */
        vmovss   %xmm1, 24(%rsp)              /* Move 3rd result */

	test    $8, %eax
	jz	LBL(.L__Scalar_fvsincos12)
	vmovss 28(%rsp), %xmm0
	vcvtps2pd %xmm0, %xmm0
	movl	12(%rsp),%edx
	call	LBL(.L__fvs_sincos_local)
	jmp	LBL(.L__Scalar_fvsincos14)

LBL(.L__Scalar_fvsincos12):
	vmovss   28(%rsp),%xmm0                /* x(4) */
	test    $8, %ecx
	jz	LBL(.L__Scalar_fvsincos13)
	CALL(ENT(__mth_i_sincos))
	jmp	LBL(.L__Scalar_fvsincos14)

LBL(.L__Scalar_fvsincos13):
	CALL(ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA)))


/* ---------------------------------- */
LBL(.L__Scalar_fvsincos14):
        vmovss   %xmm0, 12(%rsp)               /* Move 4th result */
        vmovss   %xmm1, 28(%rsp)               /* Move 4th result */
	vmovaps	(%rsp), %xmm0
	vmovaps	16(%rsp), %xmm1
        movq    %rbp, %rsp
        popq    %rbp
        ret

LBL(.L__fvs_sincos_local):
	vmovsd  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        vmovapd   %xmm0,%xmm2
        vmovapd   %xmm0,%xmm3
        shrl    $20,%edx
        cmpl    $0x0390,%edx
        jl      LBL(.L__fvs_sincos_small)
        vmovapd   %xmm0,%xmm4
        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   %xmm4,%xmm4,%xmm4
        vmulsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* x2 * s4 */
        vmulsd   .L__dble_dcos_c4(%rip),%xmm2,%xmm2    /* x2 * c4 */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + s3 */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm2,%xmm2    /* + c3 */

#ifdef TARGET_FMA
#	VFMADDPD	.L__dble_dsin_c2(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_dsin_c2(%rip),%xmm4,%xmm0)
#	VFMADDPD	.L__dble_dcos_c2(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213PD	(.L__dble_dcos_c2(%rip),%xmm4,%xmm2)
#	VFMADDPD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213PD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDPD	.L__dble_dcos_c1(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213PD	(.L__dble_dcos_c1(%rip),%xmm4,%xmm2)
	vmulsd		%xmm4,%xmm0,%xmm0
#	VFMADDPD	.L__dble_pq1+16(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213PD	(.L__dble_pq1+16(%rip),%xmm4,%xmm2)
#	VFMADDPD	%xmm3,%xmm3,%xmm0,%xmm0
	VFMA_213PD	(%xmm3,%xmm3,%xmm0)
#	VFMADDPD	%xmm1,%xmm4,%xmm2,%xmm1
	VFMA_231PD	(%xmm4,%xmm2,%xmm1)
#else
	vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + 22 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm2,%xmm2    /* + c2 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s2 + ...) */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + s1 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm2,%xmm2    /* + c1 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s1 + ...) */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2     /* - 0.5 */
        vmulsd   %xmm3,%xmm0,%xmm0                     /* x3 * (s1 + ...) */
        vaddsd   %xmm3,%xmm0,%xmm0                     /* x + x3 * (...) done */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm2,%xmm1,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
#endif
	vshufpd  $0, %xmm1, %xmm0,%xmm0

LBL(.L__fvs_sincos_done1):
	vcvtpd2ps %xmm0,%xmm0                 /* Try to do 2 converts at once */
	vmovaps   %xmm0, %xmm1
	vshufps  $1, %xmm1, %xmm1,%xmm1             /* xmm1 now has cos(x) */
        ret

LBL(.L__fvs_sincos_small):
        cmpl    $0x0320,%edx
	vshufpd  $0, %xmm1, %xmm0,%xmm0
        jl      LBL(.L__fvs_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_pq1(%rip),%xmm3,%xmm3
#ifdef TARGET_FMA
#	VFMADDPD	%xmm0,%xmm2,%xmm3,%xmm0
	VFMA_231PD	(%xmm2,%xmm3,%xmm0)
#	VFMADDPD	%xmm1,.L__dble_pq1+16(%rip),%xmm2,%xmm1
	VFMA_231PD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#else
        vmulsd   %xmm2,%xmm3,%xmm3
        vaddsd   %xmm3,%xmm0,%xmm0
        vmulsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2
        vaddsd   %xmm2,%xmm1,%xmm1
#endif
	vshufpd  $0, %xmm1, %xmm0,%xmm0
	jmp 	LBL(.L__fvs_sincos_done1)

        ELF_FUNC(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA))

/* ============================================================ */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA)):

        vmovd    %xmm0, %eax
        mov     $0x03f490fdb,%edx   /* pi / 4 */
        vmovss   .L__sngl_sixteen_by_pi(%rip),%xmm4
        and     .L__sngl_mask_unsign(%rip), %eax
        cmp     %edx,%eax
        jle     LBL(.L__fss_sin_shortcuts)
        shrl    $20,%eax
        cmpl    $0x498,%eax
        jge     GBLTXT(ENT(__mth_i_sin))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulss   %xmm0,%xmm4,%xmm4
        vunpcklps %xmm0, %xmm0, %xmm0
        vcvtps2pd %xmm0, %xmm0

        /* Set n = nearest integer to r */
        vcvtss2si %xmm4,%rcx    /* convert to integer */
        vmovsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        vmovsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovsd   .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtsi2sd %rcx,%xmm4,%xmm4    /* and back to double */

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm4,%xmm0
	VFNMA_231SD	(%xmm1,%xmm4,%xmm0)
	vmulsd		%xmm4,%xmm2,%xmm2
#	VFMADDSD	%xmm2,%xmm3,%xmm4,%xmm3
	VFMA_213SD	(%xmm2,%xmm4,%xmm3)
	vsubsd		%xmm3,%xmm0,%xmm0
#else
        vmulsd   %xmm4,%xmm1,%xmm1     /* n * p1 */
        vmulsd   %xmm4,%xmm2,%xmm2     /* n * p2 == rt */
        vmulsd   %xmm4,%xmm3,%xmm3     /* n * p3 */

        vsubsd   %xmm1,%xmm0,%xmm0     /* x - n * p1 == rh */
	vaddsd   %xmm2,%xmm3,%xmm3
        vsubsd   %xmm3,%xmm0,%xmm0     /* c = rh - rt */
#endif

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

/*...	vsubsd   %xmm1,%xmm0,%xmm0 ...*/    /* x - n * p1 == rh */
/*...	vaddsd   %xmm2,%xmm3,%xmm3 ...*/

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

/*...	vsubsd   %xmm3,%xmm0,%xmm0 ...*/    /* c = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vmovapd   %xmm0,%xmm1     /* r in xmm1 */
        vmovapd   %xmm0,%xmm2     /* r in xmm2 */
        vmovapd   %xmm0,%xmm4     /* r in xmm4 */
        vmulsd   %xmm0,%xmm0,%xmm0     /* r^2 in xmm0 */
        vmulsd   %xmm1,%xmm1,%xmm1     /* r^2 in xmm1 */
        vmulsd   %xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd   %xmm2,%xmm3     /* r in xmm3 */

        /* xmm0, xmm1, xmm4 have r^2, xmm2, xmm3 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        vmulsd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* p4 * r^2 */
        vmulsd   .L__dble_pq3+16(%rip), %xmm1, %xmm1  /* q4 * r^2 */
        vaddsd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm1, %xmm1  /* + q2 */

        vmulsd   %xmm4,%xmm3,%xmm3                   /* xmm3 = r^3 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm0,%xmm4,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm1,%xmm1
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm1)
#	VFMADDSD	%xmm2,%xmm0,%xmm3,%xmm0
	VFMA_213SD	(%xmm2,%xmm3,%xmm0)
#else
        vmulsd   %xmm4,%xmm0,%xmm0                   /* * r^2 */
        vmulsd   %xmm4,%xmm1,%xmm1                   /* * r^2 */
        vaddsd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
        vaddsd   .L__dble_pq1+16(%rip), %xmm1, %xmm1  /* + q1 */
        vmulsd   %xmm3,%xmm0,%xmm0                   /* * r^3 */
        vaddsd   %xmm2,%xmm0,%xmm0                   /* + r */
#endif
        vmulsd   %xmm4,%xmm1,%xmm1                   /* * r^2 */

        addq    %rax,%rax
        addq    %rcx,%rcx
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */


        vmulsd   (%rdx,%rax,8),%xmm1,%xmm1           /* S * q(r) */
	vaddsd   (%rdx,%rax,8),%xmm1,%xmm1           /* S + S * q(r) */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,(%rdx,%rcx,8),%xmm0,%xmm0
	VFMA_132SD	((%rdx,%rcx,8),%xmm1,%xmm0)
#else
	vmulsd   (%rdx,%rcx,8),%xmm0,%xmm0           /* C * p(r) */
        vaddsd   %xmm1,%xmm0,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
#endif

	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

LBL(.L__fss_sin_shortcuts):
        vunpcklps %xmm0, %xmm0,%xmm0
        vcvtps2pd %xmm0, %xmm0
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2
        shrl    $20,%eax
        cmpl    $0x0390,%eax
        jl      LBL(.L__fss_sin_small)

        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* x2 * c4 */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + c3 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dble_pq1(%rip),%xmm0,%xmm1,%xmm0	/* x2 * (c2 + ...) + c1 */
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm1,%xmm0)
	vmulsd   	%xmm2,%xmm1,%xmm1                     /* x3 */
#	VFMADDSD	%xmm2,%xmm1,%xmm0,%xmm0		/* x + x3 * (...) done */
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#else
        vmulsd   %xmm1,%xmm0,%xmm0			/* x2 * (c3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0	/* + c2 */
        vmulsd   %xmm1,%xmm0,%xmm0			/* x2 * (c2 + ...) */
        vmulsd   %xmm2,%xmm1,%xmm1			/* x3 */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0		/* + c1 */
        vmulsd   %xmm1,%xmm0,%xmm0			/* x3 * (c1 + ...) */
        vaddsd   %xmm2,%xmm0,%xmm0			/* x + x3 * (...) done */
#endif
	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

LBL(.L__fss_sin_small):
        cmpl    $0x0320,%eax
        jl      LBL(.L__fss_sin_done1)
        /* return x - x * x * x * 1/3! */
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   .L__dble_pq1(%rip),%xmm2,%xmm2

#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm1,%xmm2,%xmm0
	VFMA_231SD	(%xmm1,%xmm2,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif

LBL(.L__fss_sin_done1):
	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

        ELF_FUNC(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fss_sin_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA)):

        vmovd    %xmm0, %eax
        mov     $0x03f490fdb,%edx   /* pi / 4 */
        vmovss   .L__sngl_sixteen_by_pi(%rip),%xmm4
        and     .L__sngl_mask_unsign(%rip), %eax
        cmp     %edx,%eax
        jle     LBL(.L__fss_cos_shortcuts)
        shrl    $20,%eax
        cmpl    $0x498,%eax
        jge     GBLTXT(ENT(__mth_i_cos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulss   %xmm0,%xmm4,%xmm4
        vunpcklps %xmm0, %xmm0, %xmm0
        vcvtps2pd %xmm0, %xmm0

        /* Set n = nearest integer to r */
        vcvtss2si %xmm4,%rcx    /* convert to integer */
        vmovsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        vmovsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovsd   .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtsi2sd %rcx,%xmm4,%xmm4    /* and back to double */

        /* r = (x - n*p1) - (n*p2 + n*p3)  */
/*...	vmulsd   %xmm4,%xmm1,%xmm1 ...*/    /* n * p1 */
/*...	vmulsd   %xmm4,%xmm2,%xmm2 ...*/    /* n * p2 == rt */
	vmulsd   %xmm4,%xmm3,%xmm3     /* n * p3 */

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

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm4,%xmm0
	VFNMA_231SD	(%xmm1,%xmm4,%xmm0)
#	VFMADDSD	%xmm3,%xmm2,%xmm4,%xmm3
	VFMA_231SD	(%xmm2,%xmm4,%xmm3)
#else
        vmulsd   %xmm4,%xmm1,%xmm1     /* n * p1 */
        vsubsd   %xmm1,%xmm0,%xmm0     /* x - n * p1 == rh */
        vmulsd   %xmm4,%xmm2,%xmm2     /* n * p2 == rt */
	vaddsd   %xmm2,%xmm3,%xmm3
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

	vsubsd   %xmm3,%xmm0,%xmm0     /* c = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vmovapd	%xmm0,%xmm1     /* r in xmm1 */
        vmovapd	%xmm0,%xmm2     /* r in xmm2 */
        vmovapd	%xmm0,%xmm4     /* r in xmm4 */
        vmulsd	%xmm0,%xmm0,%xmm0     /* r^2 in xmm0 */
        vmulsd	%xmm1,%xmm1,%xmm1     /* r^2 in xmm1 */
        vmulsd	%xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd	%xmm2,%xmm3     /* r in xmm3 */

        /* xmm0, xmm1, xmm4 have r^2, xmm2, xmm3 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        vmulsd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* p4 * r^2 */
        vmulsd   .L__dble_pq3+16(%rip), %xmm1, %xmm1  /* q4 * r^2 */
        vaddsd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm1, %xmm1  /* + q2 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm0,%xmm4,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm1,%xmm4,%xmm1
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm1)
#else
	vmulsd   %xmm4,%xmm0,%xmm0                   /* * r^2 */
        vaddsd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
        vmulsd   %xmm4,%xmm1,%xmm1                   /* * r^2 */
        vaddsd   .L__dble_pq1+16(%rip), %xmm1, %xmm1  /* + q1 */
#endif

	vmulsd   %xmm4,%xmm3,%xmm3                   /* xmm3 = r^3 */
/*...	vmulsd   %xmm3,%xmm0,%xmm0 ...*/                  /* * r^3 */
        vmulsd   %xmm4,%xmm1,%xmm1                   /* * r^2 */

        addq    %rax,%rax
        addq    %rcx,%rcx
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm0,%xmm3,%xmm0
	VFMA_213SD	(%xmm2,%xmm3,%xmm0)
#else
	vmulsd   %xmm3,%xmm0,%xmm0                   /* * r^3 */
        vaddsd   %xmm2,%xmm0,%xmm0                   /* + r  = p(r) */
#endif

        vmulsd   (%rdx,%rcx,8),%xmm1,%xmm1           /* C * q(r) */
        vaddsd   (%rdx,%rcx,8),%xmm1,%xmm1           /* C + C * q(r) */
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm1,(%rdx,%rax,8),%xmm0,%xmm1
	VFNMA_231SD	((%rdx,%rax,8),%xmm0,%xmm1)
#else
	vmulsd   (%rdx,%rax,8),%xmm0,%xmm0           /* S * p(r) */
        vsubsd   %xmm0,%xmm1,%xmm1                   /* cos(x) = (C + Cq(r)) - Sp(r) */
#endif
	vunpcklpd %xmm1, %xmm1, %xmm1
	vcvtpd2ps %xmm1, %xmm0
        ret

LBL(.L__fss_cos_shortcuts):
        vunpcklps %xmm0, %xmm0, %xmm0
        vcvtps2pd %xmm0, %xmm0
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2
        shrl    $20,%eax
	vmovsd  .L__dble_sincostbl(%rip), %xmm0  /* 1.0 */
        cmpl    $0x0390,%eax
        jl      LBL(.L__fss_cos_small)
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_dcos_c4(%rip),%xmm1,%xmm1    /* x2 * c4 */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm1,%xmm1    /* + c3 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dcos_c2(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_dcos_c2(%rip),%xmm2,%xmm1)
#	VFMADDSD	.L__dble_dcos_c1(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_dcos_c1(%rip),%xmm2,%xmm1)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#	VFMADDSD	%xmm0,%xmm1,%xmm2,%xmm0
	VFMA_231SD	(%xmm1,%xmm2,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm1,%xmm1    /* + c2 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm1,%xmm1    /* + c1 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1     /* - 0.5 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm1,%xmm0,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
#endif
	vunpcklpd %xmm0, %xmm0, %xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

LBL(.L__fss_cos_small):
        cmpl    $0x0320,%eax
        jl      LBL(.L__fss_cos_done1)
        /* return 1.0 - x * x * 0.5 */
        vmulsd   %xmm1,%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,.L__dble_pq1+16(%rip),%xmm1,%xmm0
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm1,%xmm0)
#else
        vmulsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif

LBL(.L__fss_cos_done1):
	vunpcklpd %xmm0, %xmm0,%xmm0
	vcvtpd2ps %xmm0, %xmm0
        ret

        ELF_FUNC(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fss_cos_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA)):

        vmovd    %xmm0, %eax
        mov     $0x03f490fdb,%edx   /* pi / 4 */
        vmovss   .L__sngl_sixteen_by_pi(%rip),%xmm4
        and     .L__sngl_mask_unsign(%rip), %eax
        cmp     %edx,%eax
        jle     LBL(.L__fss_sincos_shortcuts)
        shrl    $20,%eax
        cmpl    $0x498,%eax
        jge     GBLTXT(ENT(__mth_i_sincos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulss   %xmm0,%xmm4,%xmm4
	vunpcklps %xmm0, %xmm0,%xmm0
	vcvtps2pd %xmm0, %xmm0

        /* Set n = nearest integer to r */
        vcvtss2si %xmm4,%rcx    /* convert to integer */
        vmovsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        vmovsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovsd   .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtsi2sd %rcx,%xmm4,%xmm4    /* and back to double */

        /* r = (x - n*p1) - (n*p2 + n*p3)  */

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

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm4,%xmm0
	VFNMA_231SD	(%xmm1,%xmm4,%xmm0)
	vmulsd		%xmm4,%xmm2,%xmm2
#	VFMADDSD	%xmm2,%xmm3,%xmm4,%xmm3
	VFMA_213SD	(%xmm2,%xmm4,%xmm3)
#else
        vmulsd   %xmm4,%xmm1,%xmm1     /* n * p1 */
	vsubsd   %xmm1,%xmm0,%xmm0     /* x - n * p1 == rh */
        vmulsd   %xmm4,%xmm2,%xmm2     /* n * p2 == rt */
        vmulsd   %xmm4,%xmm3,%xmm3     /* n * p3 */
	vaddsd   %xmm2,%xmm3,%xmm3
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        vsubsd   %xmm3,%xmm0,%xmm0     /* c = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vmovapd   %xmm0,%xmm1     /* r in xmm1 */
        vmovapd   %xmm0,%xmm2     /* r in xmm2 */
        vmovapd   %xmm0,%xmm4     /* r in xmm4 */
        vmulsd   %xmm0,%xmm0,%xmm0     /* r^2 in xmm0 */
        vmulsd   %xmm1,%xmm1,%xmm1     /* r^2 in xmm1 */
        vmulsd   %xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd   %xmm2,%xmm3     /* r in xmm3 */

        /* xmm0, xmm1, xmm4 have r^2, xmm2, xmm3 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        vmulsd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* p3 * r^2 */
        vmulsd   .L__dble_pq3+16(%rip), %xmm1, %xmm1  /* q3 * r^2 */
        vaddsd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm1, %xmm1  /* + q2 */

        vmulsd   %xmm4,%xmm3, %xmm3                   /* xmm3 = r^3 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm1,%xmm1
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm1)
#else
	vmulsd   %xmm4,%xmm0, %xmm0                   /* * r^2 */
	vaddsd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
	vmulsd   %xmm4,%xmm1, %xmm1                   /* * r^2 */
	vaddsd   .L__dble_pq1+16(%rip), %xmm1, %xmm1  /* + q1 */
#endif
        vmulsd   %xmm4,%xmm1, %xmm1                   /* * r^2 = q(r) */

        addq    %rax,%rax
        addq    %rcx,%rcx
        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm3,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm3,%xmm0)
#else
	vmulsd   %xmm3,%xmm0, %xmm0                   /* * r^3 */
        vaddsd   %xmm2,%xmm0,%xmm0                   /* + r  = p(r) */
#endif

	vmovsd   (%rdx,%rcx,8), %xmm5          /* Move C */
	vmovsd   (%rdx,%rax,8), %xmm2          /* Move S */
	vmovapd	%xmm1,%xmm3                   /* Move for cosine */
	vmovapd	%xmm0,%xmm4                   /* Move for sine */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm2,%xmm1,%xmm1
	VFMA_213SD	(%xmm2,%xmm2,%xmm1)
#	VFMADDSD	%xmm5,%xmm5,%xmm3,%xmm3
	VFMA_213SD	(%xmm5,%xmm5,%xmm3)
#	VFMADDSD	%xmm1,%xmm5,%xmm0,%xmm0
	VFMA_213SD	(%xmm1,%xmm5,%xmm0)
#	VFNMADDSD	%xmm3,%xmm2,%xmm4,%xmm3
	VFNMA_231SD	(%xmm2,%xmm4,%xmm3)
#else
        vmulsd   %xmm2,%xmm1,%xmm1                   /* S * q(r) */
        vaddsd   %xmm2,%xmm1,%xmm1                   /* S + S * q(r) */
        vmulsd   %xmm5,%xmm3,%xmm3                   /* C * q(r) */
        vaddsd   %xmm5,%xmm3,%xmm3                   /* C + C * q(r) */
        vmulsd   %xmm5,%xmm0,%xmm0                   /* C * p(r) */
        vaddsd   %xmm1,%xmm0,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
        vmulsd   %xmm2,%xmm4,%xmm4                   /* S * p(r) */
        vsubsd   %xmm4,%xmm3,%xmm3                   /* cos(x) = (C + Cq(r)) - Sp(r) */
#endif

        vshufpd  $0, %xmm3, %xmm0, %xmm0              /* Shuffle it in */

LBL(.L__fss_sincos_done1):
	vcvtpd2ps %xmm0,%xmm0
	vmovaps   %xmm0, %xmm1
	vshufps  $1, %xmm1, %xmm1, %xmm1             /* xmm1 now has cos(x) */
        ret

LBL(.L__fss_sincos_shortcuts):
        vunpcklps %xmm0, %xmm0, %xmm0
        vcvtps2pd %xmm0, %xmm0
	vmovsd  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
        vmovapd   %xmm0,%xmm2
        vmovapd   %xmm0,%xmm3
        shrl    $20,%eax
        cmpl    $0x0390,%eax
        jl      LBL(.L__fss_sincos_small)
        vmovapd   %xmm0,%xmm4
        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   %xmm4,%xmm4,%xmm4

        vmulsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* x2 * s4 */
        vmulsd   .L__dble_dcos_c4(%rip),%xmm2,%xmm2    /* x2 * c4 */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + s3 */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm2,%xmm2    /* + c3 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c2(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c2(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c1(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c1(%rip),%xmm4,%xmm2)
	vmulsd		%xmm4,%xmm0,%xmm0
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm2)
#	VFMADDSD	%xmm3,%xmm3,%xmm0,%xmm0
	VFMA_213SD	(%xmm3,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,%xmm4,%xmm2,%xmm1
	VFMA_231SD	(%xmm4,%xmm2,%xmm1)
#else
	vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + 22 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm2,%xmm2    /* + c2 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s2 + ...) */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + s1 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm2,%xmm2    /* + c1 */

        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s1 + ...) */

        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2     /* - 0.5 */
        vmulsd   %xmm3,%xmm0,%xmm0                     /* x3 * (s1 + ...) */
        vaddsd   %xmm3,%xmm0,%xmm0                     /* x + x3 * (...) done */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm2,%xmm1,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
#endif
	vshufpd  $0, %xmm1, %xmm0, %xmm0
	jmp 	LBL(.L__fss_sincos_done1)

LBL(.L__fss_sincos_small):
        cmpl    $0x0320,%eax
	vshufpd  $0, %xmm1, %xmm0, %xmm0
        jl      LBL(.L__fss_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_pq1(%rip),%xmm3,%xmm3
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm3,%xmm0
	VFMA_231SD	(%xmm2,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,.L__dble_pq1+16(%rip),%xmm2,%xmm1
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#else
	vmulsd   %xmm2,%xmm3,%xmm3
        vaddsd   %xmm3,%xmm0,%xmm0
        vmulsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2
        vaddsd   %xmm2,%xmm1,%xmm1
#endif
	vshufpd  $0, %xmm1, %xmm0, %xmm0
	jmp 	LBL(.L__fss_sincos_done1)

        ELF_FUNC(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fss_sincos_,TARGET_VEX_OR_FMA))


/* ============================================================ */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fsd_sin_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_sin_,TARGET_VEX_OR_FMA)):


        vmovd    %xmm0, %rax
        mov     $0x03fe921fb54442d18,%rdx
        vmovapd  .L__dble_sixteen_by_pi(%rip),%xmm4
        andq    .L__real_mask_unsign(%rip), %rax
        cmpq    %rdx,%rax
        jle     LBL(.L__fsd_sin_shortcuts)
        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     GBLTXT(ENT(__mth_i_dsin))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulsd   %xmm0,%xmm4,%xmm4

        RZ_PUSH
#if defined(_WIN64)
        vmovdqu  %ymm6, RZ_OFF(64)(%rsp)
        vmovdqu  %ymm7, RZ_OFF(96)(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5    /* convert to integer */
        vmovsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        vmovsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovsd   .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
/*...	vmulsd   %xmm4,%xmm1,%xmm1 ...*/    /* n * p1 */
/*...	vmulsd   %xmm4,%xmm2,%xmm2 ...*/    /* n * p2 == rt */
	vmulsd   %xmm4,%xmm3,%xmm3     /* n * p3 */

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

        vmovapd   %xmm0,%xmm6     /* x in xmm6 */

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm4,%xmm0
	VFNMA_231SD	(%xmm1,%xmm4,%xmm0)
#	VFNMADDSD	%xmm6,%xmm1,%xmm4,%xmm6
	VFNMA_231SD	(%xmm1,%xmm4,%xmm6)
#else
	vmulsd   %xmm4,%xmm1,%xmm1     /* n * p1 */
        vsubsd	%xmm1,%xmm0,%xmm0     /* x - n * p1 == rh */
	vsubsd	%xmm1,%xmm6,%xmm6     /* x - n * p1 == rh == c */
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm2,%xmm4,%xmm0
	VFNMA_231SD	(%xmm2,%xmm4,%xmm0)
	vsubsd		%xmm0,%xmm6,%xmm6
#	VFNMADDSD	%xmm6,%xmm2,%xmm4,%xmm6
	VFNMA_231SD	(%xmm2,%xmm4,%xmm6)
#else
	vmulsd	%xmm4,%xmm2,%xmm2     /* n * p2 == rt */
        vsubsd	%xmm2,%xmm0,%xmm0     /* rh = rh - rt */
	vsubsd	%xmm0,%xmm6,%xmm6     /* (c - rh) */
        vsubsd	%xmm2,%xmm6,%xmm6     /* ((c - rh) - rt) */
#endif

        vmovapd	%xmm0,%xmm1     /* Move rh */
        vmovapd	%xmm0,%xmm4     /* Move rh */
        vmovapd	%xmm0,%xmm5     /* Move rh */

/*...	vsubsd	%xmm2,%xmm6,%xmm6 ...*/    /* ((c - rh) - rt) */
        vsubsd	%xmm6,%xmm3,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        vmovapd	%xmm1,%xmm2     /* Move rh */
        vsubsd	%xmm3,%xmm0,%xmm0     /* c = rh - rt aka r */
        vsubsd	%xmm3,%xmm4,%xmm4     /* c = rh - rt aka r */
        vsubsd	%xmm3,%xmm5,%xmm5     /* c = rh - rt aka r */

	movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vsubsd	%xmm0,%xmm1,%xmm1     /* (rh - c) */

        vmulsd	%xmm0,%xmm0,%xmm0     /* r^2 in xmm0 */
        vmovapd	%xmm4,%xmm6     /* r in xmm6 */
        vmulsd	%xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd	%xmm5,%xmm7     /* r in xmm7 */
        vmulsd	%xmm5,%xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */

        vmulsd   .L__dble_pq4(%rip), %xmm0, %xmm0     /* p4 * r^2 */
        vsubsd   %xmm6,%xmm2,%xmm2                   /* (rh - c) */
        vmulsd   .L__dble_pq4+16(%rip), %xmm4, %xmm4  /* q4 * r^2 */
        vsubsd   %xmm3,%xmm1,%xmm1                   /* (rh - c) - rt aka rr */

        vaddsd   .L__dble_pq3(%rip), %xmm0, %xmm0     /* + p3 */
        vaddsd   .L__dble_pq3+16(%rip), %xmm4, %xmm4  /* + q3 */
        vsubsd   %xmm3,%xmm2,%xmm2                   /* (rh - c) - rt aka rr */

        vmulsd   %xmm5,%xmm7,%xmm7                   /* xmm7 = r^3 */

        vmovapd   %xmm1,%xmm3                   /* Move rr */
        vmulsd   %xmm5,%xmm1,%xmm1                   /* r * r * rr */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq2(%rip),%xmm0,%xmm5,%xmm0
	VFMA_213SD	(.L__dble_pq2(%rip),%xmm5,%xmm0)
#	VFMADDSD	.L__dble_pq2+16(%rip),%xmm4,%xmm5,%xmm4
	VFMA_213SD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#else
        vmulsd   %xmm5,%xmm0,%xmm0                   /* (p4 * r^2 + p3) * r^2 */
        vmulsd   %xmm5,%xmm4,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        vaddsd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm4, %xmm4  /* + q2 */
#endif
/*...	vmulsd   .L__dble_pq1+16(%rip), %xmm1, %xmm1 ...*/ /* r * r * rr * 0.5 */
        vmulsd   %xmm6, %xmm3, %xmm3                  /* r * rr */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        vmulsd   %xmm5,%xmm0,%xmm0			/* * r^2 */
        vmulsd   %xmm5,%xmm4,%xmm4			/* * r^2 */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,.L__dble_pq1+16(%rip),%xmm1,%xmm2
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm1,%xmm2)
#else
        vmulsd   .L__dble_pq1+16(%rip), %xmm1, %xmm1  /* r * r * rr * 0.5 */
        vaddsd   %xmm1,%xmm2,%xmm2			/* cs = rr - r * r * rt * 0.5 */
#endif
        vmovsd  8(%rdx,%rax,8),%xmm1			/* ds2 in xmm1 */

        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

        vaddsd   .L__dble_pq1(%rip), %xmm0, %xmm0	/* + p1 */
        vaddsd   .L__dble_pq1+16(%rip), %xmm4, %xmm4	/* + q1 */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm7,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm7,%xmm0)
#	VFMSUBSD	%xmm3,%xmm4,%xmm5,%xmm4
	VFMS_213SD	(%xmm3,%xmm5,%xmm4)
#else
        vmulsd   %xmm7,%xmm0,%xmm0			/* * r^3 */
        vmulsd   %xmm5,%xmm4,%xmm4			/* * r^2 == dq aka q(r) */

        vaddsd   %xmm2,%xmm0,%xmm0			/* + cs  == dp aka p(r) */
        vsubsd   %xmm3,%xmm4,%xmm4			/* - cc  == dq aka q(r) */
#endif

        vmovsd  8(%rdx,%rcx,8),%xmm3			/* dc2 in xmm3 */
        vmovsd   (%rdx,%rax,8),%xmm5			/* ds1 in xmm5 */

        vaddsd   %xmm6,%xmm0,%xmm0			/* + r   == dp aka p(r) */
        vmovapd   %xmm1,%xmm2				/* ds2 in xmm2 */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm1,%xmm4,%xmm1
	VFMA_213SD	(%xmm2,%xmm4,%xmm1)
#	VFMADDSD	%xmm1,%xmm0,%xmm3,%xmm1
	VFMA_231SD	(%xmm0,%xmm3,%xmm1)
#	VFMADDSD	%xmm1,%xmm4,%xmm5,%xmm1
	VFMA_231SD	(%xmm4,%xmm5,%xmm1)
#else
	vmulsd	%xmm4,%xmm1,%xmm1			/* ds2 * dq */
	vaddsd	%xmm2,%xmm1,%xmm1			/* ds2 + ds2*dq */
	vmulsd	%xmm0,%xmm3,%xmm3			/* dc2 * dp */
	vaddsd	%xmm3,%xmm1,%xmm1			/* (ds2 + ds2*dq) + dc2*dp */
	vmulsd	%xmm5,%xmm4,%xmm4			/* ds1 * dq */
	vaddsd	%xmm4,%xmm1,%xmm1			/* ((ds2...) + dc2*dp) + ds1*dq */
#endif
	vaddsd	%xmm5,%xmm1,%xmm1

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDSD	%xmm1,(%rdx,%rcx,8),%xmm0,%xmm0
	VFMA_132SD	((%rdx,%rcx,8),%xmm1,%xmm0)
#else */
	vmulsd	(%rdx,%rcx,8),%xmm0,%xmm0		/* dc1 * dp */
	vaddsd	%xmm1,%xmm0,%xmm0			/* sin(x) = Cp(r) + (S+Sq(r)) */
/* #endif */

#if defined(_WIN64)
	vmovdqu	RZ_OFF(64)(%rsp),%ymm6
	vmovdqu	RZ_OFF(96)(%rsp),%ymm7
#endif
        RZ_POP
        ret

LBL(.L__fsd_sin_shortcuts):
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2

	shrq	$48,%rax
	cmpl	$0x03f20,%eax
	jl	LBL(.L__fsd_sin_small)
	vmulsd	%xmm0,%xmm0,%xmm0
	vmulsd	%xmm1,%xmm1,%xmm1
	vmulsd	.L__dble_dsin_c6(%rip),%xmm0,%xmm0	/* x2 * c6 */
	vaddsd	.L__dble_dsin_c5(%rip),%xmm0,%xmm0	/* + c5 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c4(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dble_dsin_c4(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dble_dsin_c3(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dble_dsin_c3(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm1,%xmm0)
#else
        vmulsd  %xmm1,%xmm0,%xmm0                       /* x2 * (c5 + ...) */
        vaddsd  .L__dble_dsin_c4(%rip),%xmm0,%xmm0      /* + c4 */
        vmulsd  %xmm1,%xmm0,%xmm0                       /* x2 * (c4 + ...) */
        vaddsd  .L__dble_dsin_c3(%rip),%xmm0,%xmm0      /* + c3 */
        vmulsd  %xmm1,%xmm0,%xmm0                       /* x2 * (c3 + ...) */
        vaddsd  .L__dble_dsin_c2(%rip),%xmm0,%xmm0      /* + c2 */
#endif

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm1,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm1,%xmm0)
        vmulsd		%xmm2,%xmm1,%xmm1
#	VFMADDSD	%xmm2,%xmm0,%xmm1,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#else */
        vmulsd		%xmm1,%xmm0,%xmm0               /* x2 * (c2 + ...) */
        vaddsd		.L__dble_pq1(%rip),%xmm0,%xmm0  /* + c1 */
        vmulsd		%xmm2,%xmm1,%xmm1               /* x3 */
	vmulsd	%xmm1,%xmm0,%xmm0			/* x3 * (c1 + ...) */
	vaddsd	%xmm2,%xmm0,%xmm0			/* x + x3 * (...) done */
/* #endif */

        ret

LBL(.L__fsd_sin_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fsd_sin_done1)
        /* return x - x * x * x * 1/3! */
	vmulsd   %xmm1,%xmm1,%xmm1
	vmulsd   .L__dble_pq1(%rip),%xmm2,%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm1,%xmm2,%xmm0
	VFMA_231SD	(%xmm1,%xmm2,%xmm0)
#else
	vmulsd   %xmm2,%xmm1,%xmm1
	vaddsd   %xmm1,%xmm0,%xmm0
#endif
	ret

LBL(.L__fsd_sin_done1):
	rep
        ret

        ELF_FUNC(ASM_CONCAT(__fsd_sin_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_sin_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fsd_cos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_cos_,TARGET_VEX_OR_FMA)):

        vmovd    %xmm0, %rax
        mov     $0x03fe921fb54442d18,%rdx
        vmovapd  .L__dble_sixteen_by_pi(%rip),%xmm4
        andq    .L__real_mask_unsign(%rip), %rax
        cmpq    %rdx,%rax
        jle     LBL(.L__fsd_cos_shortcuts)
        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     GBLTXT(ENT(__mth_i_dcos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulsd   %xmm0,%xmm4,%xmm4

        RZ_PUSH
#if defined(_WIN64)
        vmovdqu  %ymm6, RZ_OFF(64)(%rsp)
        vmovdqu  %ymm7, RZ_OFF(96)(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5    /* convert to integer */
        vmovsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        vmovsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovsd   .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
/*...	vmulsd   %xmm4,%xmm1,%xmm1 ...*/    /* n * p1 */
/*...	vmulsd   %xmm4,%xmm2,%xmm2 ...*/    /* n * p2 == rt */
/*...	vmulsd   %xmm4,%xmm3,%xmm3 ...*/    /* n * p3 */

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

	vmovapd		%xmm0,%xmm6
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm4,%xmm0
	VFNMA_231SD	(%xmm1,%xmm4,%xmm0)
#	VFNMADDSD	%xmm6,%xmm1,%xmm4,%xmm6
	VFNMA_231SD	(%xmm1,%xmm4,%xmm6)
#	VFNMADDSD	%xmm0,%xmm2,%xmm4,%xmm0
	VFNMA_231SD	(%xmm2,%xmm4,%xmm0)
	vsubsd  	%xmm0,%xmm6,%xmm6
#	VFNMADDSD	%xmm6,%xmm2,%xmm4,%xmm6
	VFNMA_231SD	(%xmm2,%xmm4,%xmm6)
#else
	vmulsd	%xmm4,%xmm1,%xmm1     /* n * p1 */
        vsubsd	%xmm1,%xmm0,%xmm0     /* x - n * p1 == rh */
        vsubsd	%xmm1,%xmm6,%xmm6     /* x - n * p1 == rh == c */

	vmulsd	%xmm4,%xmm2,%xmm2     /* n * p2 == rt */
        vsubsd	%xmm2,%xmm0,%xmm0     /* rh = rh - rt */
        vsubsd	%xmm0,%xmm6,%xmm6     /* (c - rh) */
        vsubsd	%xmm2,%xmm6,%xmm6     /* ((c - rh) - rt) */
#endif

#ifdef TARGET_FMA
#	VFMSUBSD        %xmm6,%xmm3,%xmm4,%xmm3
	VFMS_213SD	(%xmm6,%xmm4,%xmm3)
#else
	vmulsd	%xmm4,%xmm3,%xmm3     /* n * p3 */
        vsubsd	%xmm6,%xmm3,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */


        vmovapd   %xmm0,%xmm1     /* Move rh */
        vmovapd   %xmm0,%xmm4     /* Move rh */
        vmovapd   %xmm0,%xmm5     /* Move rh */
        vmovapd   %xmm1,%xmm2     /* Move rh */
        vsubsd   %xmm3,%xmm0,%xmm0     /* c = rh - rt aka r */
        vsubsd   %xmm3,%xmm4,%xmm4     /* c = rh - rt aka r */
        vsubsd   %xmm3,%xmm5,%xmm5     /* c = rh - rt aka r */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vsubsd   %xmm0,%xmm1,%xmm1     /* (rh - c) */

        vmulsd   %xmm0,%xmm0,%xmm0     /* r^2 in xmm0 */
        vmovapd   %xmm4,%xmm6     /* r in xmm6 */
        vmulsd   %xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd   %xmm5,%xmm7     /* r in xmm7 */
        vmulsd   %xmm5,%xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        vmulsd   .L__dble_pq4(%rip),%xmm0 ,%xmm0	/* p4 * r^2 */
        vsubsd   %xmm6,%xmm2,%xmm2			/* (rh - c) */
        vmulsd   .L__dble_pq4+16(%rip),%xmm4,%xmm4	/* q4 * r^2 */
        vsubsd   %xmm3,%xmm1,%xmm1			/* (rh - c) - rt aka rr */

        vaddsd   .L__dble_pq3(%rip),%xmm0,%xmm0		/* + p3 */
        vaddsd   .L__dble_pq3+16(%rip),%xmm4,%xmm4	/* + q3 */
        vsubsd   %xmm3,%xmm2,%xmm2			/* (rh - c) - rt aka rr */

/*...	vmulsd   %xmm5,%xmm0,%xmm0 ...*/			/* (p4 * r^2 + p3) * r^2 */
/*...	vmulsd   %xmm5,%xmm4,%xmm4 ...*/			/* (q4 * r^2 + q3) * r^2 */
	vmulsd   %xmm5,%xmm7,%xmm7			/* xmm7 = r^3 */
	vmovapd   %xmm1,%xmm3                   /* Move rr */
	vmulsd   %xmm5,%xmm1,%xmm1                   /* r * r * rr */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq2(%rip),%xmm0,%xmm5,%xmm0
	VFMA_213SD	(.L__dble_pq2(%rip),%xmm5,%xmm0)
#	VFMADDSD	.L__dble_pq2+16(%rip),%xmm4,%xmm5,%xmm4
	VFMA_213SD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#	VFMADDSD	%xmm2,.L__dble_pq1+16(%rip),%xmm1,%xmm2
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm1,%xmm2)
#else
        vmulsd   %xmm5,%xmm0,%xmm0			/* (p4 * r^2 + p3) * r^2 */
        vaddsd   .L__dble_pq2(%rip), %xmm0, %xmm0     /* + p2 */
        vmulsd   %xmm5,%xmm4,%xmm4			/* (q4 * r^2 + q3) * r^2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm4, %xmm4  /* + q2 */
        vmulsd   .L__dble_pq1+16(%rip), %xmm1, %xmm1   /* r * r * rr * 0.5 */
        vaddsd   %xmm1,%xmm2,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
#endif
        vmulsd   %xmm6, %xmm3, %xmm3                  /* r * rr */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        vmovsd  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm0,%xmm5,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm5,%xmm0)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm5,%xmm4
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm5,%xmm4)
#	VFMADDSD	%xmm2,%xmm0,%xmm7,%xmm0
	VFMA_213SD	(%xmm2,%xmm7,%xmm0)
#	VFMSUBSD	%xmm3,%xmm4,%xmm5,%xmm4
	VFMS_213SD	(%xmm3,%xmm5,%xmm4)
#else
        vmulsd   %xmm5,%xmm0,%xmm0                   /* * r^2 */
	vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0		/* + p1 */
        vmulsd   %xmm5,%xmm4,%xmm4                   /* * r^2 */
	vaddsd   .L__dble_pq1+16(%rip),%xmm4,%xmm4	/* + q1 */

	vmulsd   %xmm7,%xmm0,%xmm0			/* * r^3 */
	vaddsd   %xmm2,%xmm0,%xmm0			/* + cs  == dp aka p(r) */
	vmulsd   %xmm5,%xmm4,%xmm4			/* * r^2 == dq aka q(r) */
	vsubsd   %xmm3,%xmm4,%xmm4			/* - cc  == dq aka q(r) */
#endif

        vmovsd  8(%rdx,%rax,8),%xmm3          /* ds2 in xmm3 */
        vmovsd   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        vaddsd   %xmm0,%xmm6,%xmm6                   /* + r   == dp aka p(r) */
        vmovapd   %xmm1,%xmm2                   /* dc2 in xmm2 */
        vmovsd  (%rdx,%rcx,8),%xmm0			/* dc1 */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm1,%xmm4,%xmm1
	VFMA_213SD	(%xmm2,%xmm4,%xmm1)
#	VFNMADDSD	%xmm1,%xmm3,%xmm6,%xmm1
	VFNMA_231SD	(%xmm3,%xmm6,%xmm1)
#	VFNMADDSD	%xmm1,%xmm5,%xmm6,%xmm1
	VFNMA_231SD	(%xmm5,%xmm6,%xmm1)
#	VFMADDSD	%xmm1,%xmm0,%xmm4,%xmm1
	VFMA_231SD	(%xmm0,%xmm4,%xmm1)
#else
        vmulsd   %xmm4,%xmm1,%xmm1			/* dc2 * dq */
        vaddsd   %xmm2,%xmm1,%xmm1			/* dc2 + dc2*dq */
        vmulsd   %xmm6,%xmm3,%xmm3			/* ds2 * dp */
        vsubsd   %xmm3,%xmm1,%xmm1			/* (dc2 + dc2*dq) - ds2*dp */
        vmulsd   %xmm5,%xmm6,%xmm6			/* ds1 * dp */
        vsubsd   %xmm6,%xmm1,%xmm1			/* (() - ds2*dp) - ds1*dp */

        vmulsd   %xmm0,%xmm4,%xmm4			/* dc1 * dq */
        vaddsd   %xmm4,%xmm1,%xmm1
#endif

        vaddsd   %xmm1,%xmm0,%xmm0			/* cos(x) = (C + Cq(r)) + Sq(r) */

#if defined(_WIN64)
        vmovdqu  RZ_OFF(64)(%rsp),%ymm6
        vmovdqu  RZ_OFF(96)(%rsp),%ymm7
#endif
        RZ_POP
        ret

LBL(.L__fsd_cos_shortcuts):
        vmovapd   %xmm0,%xmm1
        vmovapd   %xmm0,%xmm2
        shrq    $48,%rax
        vmovsd  .L__dble_sincostbl(%rip),%xmm0  /* 1.0 */
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fsd_cos_small)
        vmulsd   %xmm1,%xmm1,%xmm1
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_dcos_c6(%rip),%xmm1,%xmm1    /* x2 * c6 */
        vaddsd   .L__dble_dcos_c5(%rip),%xmm1,%xmm1    /* + c5 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dcos_c4(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_dcos_c4(%rip),%xmm2,%xmm1)
#	VFMADDSD	.L__dble_dcos_c3(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_dcos_c3(%rip),%xmm2,%xmm1)
#	VFMADDSD	.L__dble_dcos_c2(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_dcos_c2(%rip),%xmm2,%xmm1)
#	VFMADDSD	.L__dble_dcos_c1(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__dble_dcos_c1(%rip),%xmm2,%xmm1)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#	VFMADDSD	%xmm0,%xmm1,%xmm2,%xmm0
	VFMA_231SD	(%xmm1,%xmm2,%xmm0)
#else
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c5 + ...) */
        vaddsd   .L__dble_dcos_c4(%rip),%xmm1,%xmm1    /* + c4 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c4 + ...) */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm1,%xmm1    /* + c3 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm1,%xmm1    /* + c2 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm1,%xmm1    /* + c1 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1     /* - 0.5 */
        vmulsd   %xmm2,%xmm1,%xmm1                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm1,%xmm0,%xmm0                     /* 1.0 - 0.5x2 + (...) done */
#endif
        ret

LBL(.L__fsd_cos_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fsd_cos_done1)
        /* return 1.0 - x * x * 0.5 */
        vmulsd   %xmm1,%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,.L__dble_pq1+16(%rip),%xmm1,%xmm0
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm1,%xmm0)
#else
        vmulsd   .L__dble_pq1+16(%rip),%xmm1,%xmm1
        vaddsd   %xmm1,%xmm0,%xmm0
#endif
        ret

LBL(.L__fsd_cos_done1):
	rep
        ret

        ELF_FUNC(ASM_CONCAT(__fsd_cos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_cos_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA)):

##..        vmovd    %xmm0, %rax
        vmovd   %xmm0, %rax
        mov     $0x03fe921fb54442d18,%rdx
        vmovapd  .L__dble_sixteen_by_pi(%rip),%xmm4
        andq    .L__real_mask_unsign(%rip), %rax
        cmpq    %rdx,%rax
        jle     LBL(.L__fsd_sincos_shortcuts)
        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     GBLTXT(ENT(__mth_i_dsincos))

        /* Step 1. Reduce the argument x. */
        /* Find N, the closest integer to 16x / pi */
        vmulsd   %xmm0,%xmm4,%xmm4

        RZ_PUSH
#if defined(_WIN64)
        vmovdqu  %ymm6, RZ_OFF(32)(%rsp)
        vmovdqu  %ymm7, RZ_OFF(64)(%rsp)
        vmovdqu  %ymm8, RZ_OFF(96)(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5    /* convert to integer */
        vmovsd   .L__dble_pi_by_16_ms(%rip), %xmm1
        vmovsd   .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovsd   .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtdq2pd %xmm5,%xmm4    /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        vmulsd   %xmm4,%xmm2,%xmm2     /* n * p2 == rt */
        vmulsd   %xmm4,%xmm3,%xmm3     /* n * p3 */

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

        vmovapd   %xmm0,%xmm6     /* x in xmm6 */
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm4,%xmm0
	VFNMA_231SD	(%xmm1,%xmm4,%xmm0)
#	VFNMADDSD	%xmm6,%xmm1,%xmm4,%xmm6
	VFNMA_231SD	(%xmm1,%xmm4,%xmm6)
#else
	vmulsd   %xmm4,%xmm1,%xmm1     /* n * p1 */
        vsubsd   %xmm1,%xmm0,%xmm0     /* x - n * p1 == rh */
        vsubsd   %xmm1,%xmm6,%xmm6     /* x - n * p1 == rh == c */
#endif

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        vsubsd   %xmm2,%xmm0,%xmm0     /* rh = rh - rt */
        vsubsd   %xmm0,%xmm6,%xmm6     /* (c - rh) */

        vmovapd   %xmm0,%xmm1     /* Move rh */
        vmovapd   %xmm0,%xmm4     /* Move rh */
        vmovapd   %xmm0,%xmm5     /* Move rh */
        vsubsd   %xmm2,%xmm6,%xmm6     /* ((c - rh) - rt) */
        vsubsd   %xmm6,%xmm3,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        vmovapd   %xmm1,%xmm2     /* Move rh */
        vsubsd   %xmm3,%xmm0,%xmm0     /* c = rh - rt aka r */
        vsubsd   %xmm3,%xmm4,%xmm4     /* c = rh - rt aka r */
        vsubsd   %xmm3,%xmm5,%xmm5     /* c = rh - rt aka r */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vsubsd   %xmm0,%xmm1,%xmm1     /* (rh - c) */

        vmulsd   %xmm0,%xmm0,%xmm0     /* r^2 in xmm0 */
        vmovapd   %xmm4,%xmm6     /* r in xmm6 */
        vmulsd   %xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd   %xmm5,%xmm7     /* r in xmm7 */
        vmulsd   %xmm5,%xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm0, xmm4, xmm5 have r^2, xmm1, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        vmulsd   .L__dble_pq4(%rip),%xmm0,%xmm0		/* p4 * r^2 */
        vsubsd   %xmm6,%xmm2,%xmm2			/* (rh - c) */
        vmulsd   .L__dble_pq4+16(%rip),%xmm4,%xmm4	/* q4 * r^2 */
        vsubsd   %xmm3,%xmm1,%xmm1			/* (rh - c) - rt aka rr */

        vaddsd   .L__dble_pq3(%rip),%xmm0,%xmm0		/* + p3 */
        vaddsd   .L__dble_pq3+16(%rip),%xmm4,%xmm4	/* + q3 */
        vsubsd   %xmm3,%xmm2,%xmm2			/* (rh - c) - rt aka rr */

        vmulsd   %xmm5,%xmm7,%xmm7			/* xmm7 = r^3 */
        vmovapd   %xmm1,%xmm3				/* Move rr */
        vmulsd   %xmm5,%xmm1,%xmm1			/* r * r * rr */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq2(%rip),%xmm5,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq2(%rip),%xmm5,%xmm0)
#	VFMADDSD	.L__dble_pq2+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213SD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#	VFMADDSD	%xmm2,.L__dble_pq1+16(%rip),%xmm1,%xmm2
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm1,%xmm2)
#else
        vmulsd   %xmm5,%xmm0,%xmm0	            /* (p4 * r^2 + p3) * r^2 */
        vaddsd   .L__dble_pq2(%rip),%xmm0,%xmm0     /* + p2 */
        vmulsd   %xmm5,%xmm4,%xmm4	            /* (q4 * r^2 + q3) * r^2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm4,%xmm4 /* + q2 */
        vmulsd   .L__dble_pq1+16(%rip), %xmm1,%xmm1 /* r * r * rr * 0.5 */
        vaddsd   %xmm1,%xmm2,%xmm2                  /* cs = rr - r * r * rt * 0.5 */
#endif
        vmulsd   %xmm6, %xmm3,%xmm3                 /* r * rr */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

        vmovsd  8(%rdx,%rax,8),%xmm8          /* ds2 in xmm8 */
        vmovsd  8(%rdx,%rcx,8),%xmm1          /* dc2 in xmm1 */
        /* xmm0 has dp, xmm4 has dq,
           xmm1 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3
           xmm8 is ds2 */

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1(%rip),%xmm5,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm5,%xmm0)
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm5,%xmm4)
#	VFMADDSD	%xmm2,%xmm7,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm7,%xmm0)
#	VFMSUBSD	%xmm3,%xmm4,%xmm5,%xmm4
	VFMS_213SD	(%xmm3,%xmm5,%xmm4)
#else
        vmulsd   %xmm5,%xmm0,%xmm0                   /* * r^2 */
        vaddsd   .L__dble_pq1(%rip), %xmm0, %xmm0     /* + p1 */
        vmulsd   %xmm5,%xmm4,%xmm4                   /* * r^2 */
        vaddsd   .L__dble_pq1+16(%rip), %xmm4, %xmm4  /* + q1 */

        vmulsd   %xmm7,%xmm0,%xmm0                   /* * r^3 */
        vaddsd   %xmm2,%xmm0,%xmm0                   /* + cs  == dp aka p(r) */
	vmulsd   %xmm5,%xmm4,%xmm4                   /* * r^2 == dq aka q(r) */
        vsubsd   %xmm3,%xmm4,%xmm4                   /* - cc  == dq aka q(r) */
#endif
        vmovapd   %xmm1,%xmm3                   /* dc2 in xmm3 */
        vmovsd   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        vmovsd   (%rdx,%rcx,8),%xmm7          /* dc1 in xmm7 */
        vaddsd   %xmm6,%xmm0,%xmm0                   /* + r   == dp aka p(r) */
        vmovapd   %xmm8,%xmm2                   /* ds2 in xmm2 */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm4,%xmm8,%xmm8
	VFMA_213SD	(%xmm2,%xmm4,%xmm8)
#	VFMADDSD	%xmm3,%xmm4,%xmm1,%xmm1
	VFMA_213SD	(%xmm3,%xmm4,%xmm1)
#	VFMADDSD	%xmm8,%xmm3,%xmm0,%xmm8
	VFMA_231SD	(%xmm3,%xmm0,%xmm8)
#	VFNMADDSD	%xmm1,%xmm0,%xmm2,%xmm1
	VFNMA_231SD	(%xmm0,%xmm2,%xmm1)
#else
        vmulsd   %xmm4,%xmm8,%xmm8                   /* ds2 * dq */
        vaddsd   %xmm2,%xmm8,%xmm8                   /* ds2 + ds2*dq */
        vmulsd   %xmm4,%xmm1,%xmm1                   /* dc2 * dq */
        vaddsd   %xmm3,%xmm1,%xmm1                   /* dc2 + dc2*dq */

        vmulsd   %xmm0,%xmm3,%xmm3                   /* dc2 * dp */
        vaddsd   %xmm3,%xmm8,%xmm8                   /* (ds2 + ds2*dq) + dc2*dp */
        vmulsd   %xmm0,%xmm2,%xmm2                   /* ds2 * dp */
        vsubsd   %xmm2,%xmm1,%xmm1                   /* (dc2 + dc2*dq) - ds2*dp */
#endif

	vmovapd	%xmm4,%xmm6                   /* xmm6 = dq */
	vmovapd	%xmm5,%xmm3                   /* xmm3 = ds1 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm8,%xmm4,%xmm5,%xmm8
	VFMA_231SD	(%xmm4,%xmm5,%xmm8)
#	VFNMADDSD	%xmm1,%xmm0,%xmm5,%xmm1
	VFNMA_231SD	(%xmm0,%xmm5,%xmm1)
#	VFMADDSD	%xmm1,%xmm6,%xmm7,%xmm1
	VFMA_231SD	(%xmm6,%xmm7,%xmm1)
	vaddsd		%xmm3,%xmm8,%xmm8
#	VFMADDSD	%xmm8,%xmm0,%xmm7,%xmm0
	VFMA_213SD	(%xmm8,%xmm7,%xmm0)
#else
        vmulsd   %xmm5,%xmm4,%xmm4                   /* ds1 * dq */
        vaddsd   %xmm4,%xmm8,%xmm8                   /* ((ds2...) + dc2*dp) + ds1*dq */
        vmulsd   %xmm0,%xmm5,%xmm5                   /* ds1 * dp */
        vsubsd   %xmm5,%xmm1,%xmm1                   /* (() - ds2*dp) - ds1*dp */
	vmulsd   %xmm7,%xmm6,%xmm6                   /* dc1 * dq */
        vaddsd   %xmm6,%xmm1,%xmm1                   /* + dc1*dq */

        vaddsd   %xmm3,%xmm8,%xmm8                   /* + ds1 */

        vmulsd   %xmm7,%xmm0,%xmm0                   /* dc1 * dp */
        vaddsd   %xmm8,%xmm0,%xmm0                   /* sin(x) = Cp(r) + (S+Sq(r)) */
#endif
        vaddsd   %xmm7,%xmm1,%xmm1                   /* cos(x) = (C + Cq(r)) + Sq(r) */

#if defined(_WIN64)
        vmovdqu  RZ_OFF(32)(%rsp),%ymm6
        vmovdqu  RZ_OFF(64)(%rsp),%ymm7
        vmovdqu  RZ_OFF(96)(%rsp),%ymm8
#endif
        RZ_POP
        ret

LBL(.L__fsd_sincos_shortcuts):
	vmovsd  .L__dble_sincostbl(%rip), %xmm1  /* table */
        vmovapd   %xmm0,%xmm2
        vmovapd   %xmm0,%xmm3
        shrq    $48,%rax
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fsd_sincos_small)
        vmovapd   %xmm0,%xmm4
        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   %xmm4,%xmm4,%xmm4

#ifdef TARGET_FMA
	vmovsd		.L__dble_dsin_c5(%rip),%xmm5
#	VFMADDSD	%xmm5,.L__dble_dsin_c6(%rip),%xmm0,%xmm0
	VFMA_132SD	(.L__dble_dsin_c6(%rip),%xmm5,%xmm0)
	vmovsd		.L__dble_dcos_c5(%rip),%xmm5
#	VFMADDSD	%xmm5,.L__dble_dcos_c6(%rip),%xmm2,%xmm2
	VFMA_132SD	(.L__dble_dcos_c6(%rip),%xmm5,%xmm2)
#	VFMADDSD	.L__dble_dsin_c4(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c4(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c4(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c4(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_dsin_c3(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c3(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c3(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c3(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c2(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c2(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c1(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c1(%rip),%xmm4,%xmm2)
#else
        vmulsd   .L__dble_dsin_c6(%rip),%xmm0,%xmm0    /* x2 * s6 */
        vmulsd   .L__dble_dcos_c6(%rip),%xmm2,%xmm2    /* x2 * c6 */
        vaddsd   .L__dble_dsin_c5(%rip),%xmm0,%xmm0                     /* + s5 */
        vaddsd   .L__dble_dcos_c5(%rip),%xmm2,%xmm2                     /* + c5 */
	vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s5 + ...) */
        vaddsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* + s4 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c5 + ...) */
        vaddsd   .L__dble_dcos_c4(%rip),%xmm2,%xmm2    /* + c4 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s4 + ...) */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + s3 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c4 + ...) */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm2,%xmm2    /* + c3 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + s2 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm2,%xmm2    /* + c2 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s2 + ...) */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + s1 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm2,%xmm2    /* + c1 */
#endif

	vmulsd		%xmm4,%xmm0,%xmm0

#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm2)
#else
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2     /* - 0.5 */
#endif

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDSD	%xmm3,%xmm3,%xmm0,%xmm0
	VFMA_213SD	(%xmm3,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,%xmm4,%xmm2,%xmm1
	VFMA_231SD	(%xmm4,%xmm2,%xmm1)
#else */
        vmulsd   %xmm3,%xmm0,%xmm0                     /* x3 */
        vaddsd   %xmm3,%xmm0,%xmm0                     /* x + x3 * (...) done */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm2,%xmm1,%xmm1        /* 1.0 - 0.5x2 + (...) done */
/* #endif */

        ret

LBL(.L__fsd_sincos_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fsd_sincos_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_pq1(%rip),%xmm3,%xmm3
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm3,%xmm0
	VFMA_231SD	(%xmm2,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,.L__dble_pq1+16(%rip),%xmm2,%xmm1
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#else
	vmulsd   %xmm2,%xmm3,%xmm3
        vaddsd   %xmm3,%xmm0,%xmm0
        vmulsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2
        vaddsd   %xmm2,%xmm1,%xmm1
#endif
        ret

LBL(.L__fsd_sincos_done1):
	rep
        ret

        ELF_FUNC(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_sincos_,TARGET_VEX_OR_FMA))


/* ------------------------------------------------------------------------- */

/*
 *	double complex __fvz_exp_evex_512(%zmm0-pd)
 *
 *	Allocate dcmplx_t structure on stack and then call four times:
 *	void __fsz_exp_vex(dcmplx_t *, creal(carg), cimag(carg))
 *
 *	dcmplx_t defined as:
 *	typedef struct {
 *		double	real;
 *		double	imag;
 *	} dcmplx_t;
 *
 *	Linux86-64/OSX64:
 *		Entry:
 *		(%zmm0[0:127])		CMPLX(REAL(carg[0]),IMAG(carg[0]))
 *		(%zmm0[128:255])	CMPLX(REAL(carg[1]),IMAG(carg[1]))
 *		(%zmm0[256:383])	CMPLX(REAL(carg[2]),IMAG(carg[2]))
 *		(%zmm0[384:511])	CMPLX(REAL(carg[3]),IMAG(carg[3]))
 *
 *		Exit:
 *		(%zmm0[0:127])		EXP(CMPLX(REAL(carg[0]),IMAG(carg[0])))
 *		(%zmm0[128:255])	EXP(CMPLX(REAL(carg[1]),IMAG(carg[1])))
 *		(%zmm0[256:383])	EXP(CMPLX(REAL(carg[2]),IMAG(carg[2])))
 *		(%zmm0[384:511])	EXP(CMPLX(REAL(carg[3]),IMAG(carg[3])))
 */

#if ! defined (TARGET_FMA)
        .text
        ALN_FUNC
        .globl ENT(__fvz_exp_evex_512)
ENT(__fvz_exp_evex_512):

#if	! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
	pushq	%rbp
	movq	%rsp,%rbp
	andq	$-64,%rsp
        subq    $128,%rsp		/* This assumes that the stack is already aligned on 8B */
        vmovapd %zmm0,(%rsp)		/* Save carg */

	leaq	64(%rsp),%rdi		/* Return struct for cexp(carg[0]) */
	vmovhlps %xmm0,%xmm1,%xmm1	/* (%xmm1) = imag */
        CALL(ENT(__fsz_exp_vex))

	vmovapd	16(%rsp),%xmm0
	leaq	16+64(%rsp),%rdi	/* Return struct for cexp(carg[1]) */
	vmovhlps %xmm0,%xmm1,%xmm1	/* (%xmm1) = imag */
        CALL(ENT(__fsz_exp_vex))

	vmovapd	32(%rsp),%xmm0
	leaq	32+64(%rsp),%rdi	/* Return struct for cexp(carg[2]) */
	vmovhlps %xmm0,%xmm1,%xmm1	/* (%xmm1) = imag */
        CALL(ENT(__fsz_exp_vex))

	vmovapd	48(%rsp),%xmm0
	leaq	48+64(%rsp),%rdi	/* Return struct for cexp(carg[3]) */
	vmovhlps %xmm0,%xmm1,%xmm1	/* (%xmm1) = imag */
        CALL(ENT(__fsz_exp_vex))

	vmovapd	64(%rsp),%zmm0		/* Real-lower, Imag-upper */
	movq	%rbp,%rsp
	popq	%rbp
        ret
#else		// ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)
	ud2	// No support for OSX & Windows yet - but entry points are needed
#endif		// ! defined(TARGET_OSX_X8664) && ! defined(TARGET_WIN_X8664)

        ELF_FUNC(__fvz_exp_evex_512)
        ELF_SIZE(__fvz_exp_evex_512)
#endif // ! defined (TARGET_FMA)

/*
 *	#ifdef TARGET_FMA
 *	double complex __fvz_exp_fma4(%ymm0-pd)
 *	#else
 *	double complex __fvz_exp_vex(%ymm0-pd)
 *	#endif
 *
 *	Allocate dcmplx_t structure on stack and then call twice:
 *	void __fsz_exp_vex(dcmplx_t *, creal(carg), cimag(carg))
 *
 *	dcmplx_t defined as:
 *	typedef struct {
 *		double	real;
 *		double	imag;
 *	} dcmplx_t;
 *
 *	Linux86-64/OSX64:
 *		Entry:
 *		(%ymm0[0:63])	REAL(carg[0])
 *		(%ymm0[64:127])	IMAG(carg[0])
 *		(%ymm0[128:191])REAL(carg[1])
 *		(%ymm0[192:255])IMAG(carg[1])
 *
 *		Exit:
 *		(%ymm0[0:63])	REAL(cexp(carg[0]))
 *		(%ymm0[64:127])	IMAG(cexp(carg[0]))
 *		(%ymm0[128:191])REAL(cexp(carg[1]))
 *		(%ymm0[192:255])IMAG(cexp(carg[1]))
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fvz_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvz_exp_,TARGET_VEX_OR_FMA)):

	pushq	%rbp
	movq	%rsp,%rbp
	andq	$-32,%rsp
        subq    $64,%rsp		/* This assumes that the stack is already aligned on 8B */

        vmovapd %xmm0,32(%rsp)		/* Save carg[0] */

	leaq	16(%rsp),%rdi		/* Return struct for cexp(carg[1]) */
	vmovapd	%xmm0,(%rsp)
	vextractf128 $1,%ymm0,%xmm0	/* (%xmm0) = carg[1] */
	vmovhlps %xmm0,%xmm1,%xmm1	/* (%xmm1) = imag */
        CALL(ENT(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA)))


	movq	%rsp,%rdi		/* Return struct for cexp(carg[0]) */
	vmovaps	(%rsp),%xmm0		/* (%xmm0) = carg[0] */
	vmovhlps %xmm0,%xmm1,%xmm1	/* (%xmm1) = cimag(carg[0]) */
        CALL(ENT(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA)))


	vmovapd	(%rsp),%ymm0		/* Real-lower, Imag-upper */
	movq	%rbp,%rsp
	popq	%rbp
        ret

        ELF_FUNC(ASM_CONCAT(__fvz_exp_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvz_exp_,TARGET_VEX_OR_FMA))


/*
 *	#ifdef TARGET_FMA
 *	double complex __fsz_exp_fma4_1v(%xmm0-pd)
 *	#else
 *	double complex __fsz_exp_vex_1v(%xmm0-pd)
 *	#endif
 *
 *
 *	Allocate dcmplx_t structure on stack and then call:
 *	void __fsz_exp_vex(dcmplx_t *, creal(carg), cimag(carg))
 *
 *	dcmplx_t defined as:
 *	typedef struct {
 *		double	real;
 *		double	imag;
 *	} dcmplx_t;
 *
 *	Linux86-64/OSX64:
 *		Entry:
 *		(%xmm0-lower)	REAL(carg)
 *		(%xmm0-upper)	IMAG(carg)
 *		Exit:
 *		(%xmm0-lower)	REAL(exp(carg))
 *		(%xmm0-upper)	IMAG(exp(carg))
 *	Windows:
 *		Entry:
 *		(%rcx):		pointer to return struct
 *		(%xmm1)		REAL(carg)
 *		(%xmm2)		IMAG(carg)
 *		Exit:
 *		0(%rcx)		REAL(exp(carg))
 *		8(%rcx)		IMAG(exp(carg))
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fsz_exp_1v_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsz_exp_1v_,TARGET_VEX_OR_FMA)):


#ifdef	_WIN64
	/*
	 * Return structure in (%rcx).
	 * Will be managed by macro I1.
	 */
	jmp	LBL(.L__fsz_exp_vex_win64)
#else
        subq    $24,%rsp		/* This assumes that the stack is already aligned on 8B */
        movq    %rsp,%rdi
	movhlps %xmm0,%xmm1
        CALL(ENT(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA)))


	vmovapd	0(%rsp), %xmm0		/* Real-lower, Imag-upper */
	addq	$24,%rsp
        ret
#endif

        ELF_FUNC(ASM_CONCAT(__fsz_exp_1v_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsz_exp_1v_,TARGET_VEX_OR_FMA))


/*
 *	#ifdef TARGET_FMA
 *	void __fsz_exp_fma4_c99(double real, double imag)
 *	#else
 *	void __fsz_exp_vex_c99(double real, double imag)
 *	#endif
 *
 *	Allocate dcmplx_t structure on stack and then call
 *	__fsz_exp_{vex,fma4}_c99(%rdi, double real, double imag)
 */
        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fsz_exp_,TARGET_VEX_OR_FMA,_c99))
ENT(ASM_CONCAT3(__fsz_exp_,TARGET_VEX_OR_FMA,_c99)):


	subq	$24,%rsp
	movq	%rsp,%rdi	/* Allocate valid return struct */

	CALL(ENT(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA)))


	vmovapd	(%rsp),%xmm0	/* Unpack results */
	vmovhlps	%xmm0,%xmm1,%xmm1
	addq	$24,%rsp
	ret

        ELF_FUNC(ASM_CONCAT3(__fsz_exp_,TARGET_VEX_OR_FMA,_c99))
        ELF_SIZE(ASM_CONCAT3(__fsz_exp_,TARGET_VEX_OR_FMA,_c99))


/*
 *	#ifdef TARGET_FMA
 *	void __fsz_exp_fma4(dcmplx_t *, double real, double imag)
 *	#else
 *	void __fsz_exp_vex(dcmplx_t *, double real, double imag)
 *	#endif
 *
 *	compute double precision complex EXP(real + I*imag)
 *
 *	Return:
 *	0(%rdi) = real
 *	8(%rdi) = imag
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA)):


#if defined(_WIN64)
	/*
	 *	WIN64 ONLY:
	 *	Jump entry point into routine from __fsz_exp_vex_v1.
	 */
LBL(.L__fsz_exp_vex_win64):
#endif

#if	defined(_WIN64)
	vmovapd	%xmm1,%xmm0
	vmovapd	%xmm2,%xmm1
#endif
        vcomisd .L__real_ln_max_doubleval1(%rip), %xmm0  /* compare to max */
        ja      LBL(.L_cdexp_inf)
        vmovd    %xmm1, %rax                             /* Move imag to gp */
        vshufpd $0, %xmm0, %xmm1, %xmm1                        /* pack real & imag */

        mov     $0x03fe921fb54442d18,%rdx
        vmovapd  .L__dble_cdexp_by_pi(%rip),%xmm4        /* For exp & sincos */
        andq    .L__real_mask_unsign(%rip), %rax       /* abs(imag) in gp */
        vcomisd .L__real_ln_min_doubleval1(%rip), %xmm0  /* compare to min */
        jbe     LBL(.L_cdexp_ninf)

        cmpq    %rdx,%rax
        jle     LBL(.L__fsz_exp_shortcuts)

        shrq    $52,%rax
        cmpq    $0x413,%rax
        jge     LBL(.L__fsz_exp_hard)

        /* Step 1. Reduce the argument x. */
        /* For sincos, the closest integer to 16x / pi */
        /* For exp, the closest integer to x * 32 / ln(2) */
        vmulpd   %xmm1,%xmm4,%xmm4                             /* Mpy to scale both */

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu  %ymm6, 128(%rsp)
        vmovdqu  %ymm7, 160(%rsp)
        vmovdqu  %ymm8, 192(%rsp)
        vmovdqu  %ymm9, 224(%rsp)
        movq    I1,24(%rsp)
#endif

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm4,%xmm5                          /* convert to integer */
        vmovsd  .L__dble_pi_by_16_ms(%rip), %xmm9
        vmovhpd  .L__cdexp_log2_by_32_lead(%rip), %xmm9, %xmm9

        vmovsd  .L__dble_pi_by_16_ls(%rip), %xmm2
        vmovhpd  .L__cdexp_log2_by_32_tail(%rip), %xmm2, %xmm2

        vmovsd  .L__dble_pi_by_16_us(%rip), %xmm3
        vcvtdq2pd %xmm5,%xmm4                          /* and back to double */

        vmovd    %xmm5, %rcx

        /* r = ((x - n*p1) - n*p2) - n*p3 (I wish it was this easy!) */
        vmulpd   %xmm4,%xmm9,%xmm9     /* n * p1 */
        vmulpd   %xmm4,%xmm2,%xmm2     /* n * p2 == rt */
        vmulsd   %xmm4,%xmm3,%xmm3     /* n * p3 */

        /* How to convert N into a table address */
        leaq    24(%rcx),%rax /* Add 24 for sine */
        andq    $0x1f,%rax    /* And lower 5 bits */
        movq    %rcx,%r8      /* Save in r8 */
        andq    $0x1f,%rcx    /* And lower 5 bits */
        rorq    $5,%rax       /* rotate right so bit 4 is sign bit */
        rorq    $5,%rcx       /* rotate right so bit 4 is sign bit */
        sarq    $4,%rax       /* Duplicate sign bit 4 times */
        sarq    $4,%rcx       /* Duplicate sign bit 4 times */
        rolq    $9,%rax       /* Shift back to original place */
        rolq    $9,%rcx       /* Shift back to original place */

        vmovapd   %xmm1,%xmm6     /* x in xmm6 */
        vsubpd   %xmm9,%xmm1,%xmm1     /* x - n * p1 == rh */
        vsubsd   %xmm9,%xmm6,%xmm6     /* x - n * p1 == rh == c */

        movq    %rax, %rdx    /* Duplicate it */
        sarq    $4,%rax       /* Sign bits moved down */
        xorq    %rax, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rax       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rax    /* Final tbl address */

        vsubpd   %xmm2,%xmm1,%xmm1     /* rh = rh - rt */

        movq    %rcx, %rdx    /* Duplicate it */
        sarq    $4,%rcx       /* Sign bits moved down */
        xorq    %rcx, %rdx    /* Xor bits, backwards over half the cycle */
        sarq    $4,%rcx       /* Sign bits moved down */
        andq    $0xf,%rdx     /* And lower 5 bits */
        addq    %rdx, %rcx    /* Final tbl address */

        vsubsd   %xmm1,%xmm6,%xmm6     /* (c - rh) */
        vmovapd   %xmm1,%xmm9     /* Move rh */
        vmovapd   %xmm1,%xmm4     /* Move rh */
        vmovapd  %xmm1,%xmm8
        vsubsd   %xmm2,%xmm6,%xmm6     /* ((c - rh) - rt) */

        vmovsd  .L__real_3FC5555555548F7C(%rip),%xmm0
        vmovhpd  .L__real_3f56c1728d739765(%rip),%xmm0,%xmm0

        vmovapd   %xmm1,%xmm5     /* Move rh */

        vmovsd  .L__real_3fe0000000000000(%rip),%xmm7
        vmovhpd  .L__real_3F811115B7AA905E(%rip),%xmm7,%xmm7

        vshufpd  $3, %xmm8, %xmm8, %xmm8

        vsubsd   %xmm6,%xmm3,%xmm3     /* rt = nx*dpiovr16u - ((c - rh) - rt) */
        vmovapd   %xmm9,%xmm2     /* Move rh */
#ifdef TARGET_FMA
#        VFMADDPD        %xmm7,%xmm8,%xmm0,%xmm0
	VFMA_213PD	(%xmm7,%xmm8,%xmm0)
#else
        vmulpd   %xmm8,%xmm0,%xmm0     /* r/720, r/6 */
        vaddpd   %xmm7,%xmm0,%xmm0      /* r/720 + 1/120, r/6 + 1/2 */
#endif

        vsubsd   %xmm3,%xmm1,%xmm1      /* c = rh - rt aka r */
        vsubsd   %xmm3,%xmm4,%xmm4      /* c = rh - rt aka r */
        vsubsd   %xmm3,%xmm5,%xmm5      /* c = rh - rt aka r */
        vmulsd   %xmm8,%xmm8,%xmm8      /* r, r^2 */
        vsubsd   %xmm1,%xmm9,%xmm9      /* (rh - c) */

        vmulpd   %xmm1,%xmm1,%xmm1     /* r^2 in both halves */
        vmovapd   %xmm4,%xmm6     /* r in xmm6 */
        vmulsd   %xmm4,%xmm4,%xmm4     /* r^2 in xmm4 */
        vmovapd   %xmm5,%xmm7     /* r in xmm7 */
        vmulsd   %xmm5,%xmm5,%xmm5     /* r^2 in xmm5 */

        /* xmm1, xmm4, xmm5 have r^2, xmm9, xmm2 has rr, xmm6, xmm7 has r */

        /* Step 2. Compute the polynomial. */
        /* p(r) = r + p1r^3 + p2r^5 + p3r^7 + p4r^9
           q(r) =     q1r^2 + q2r^4 + q3r^6 + q4r^8
           p(r) = (((p4 * r^2 + p3) * r^2 + p2) * r^2 + p1) * r^3 + r
           q(r) = (((q4 * r^2 + q3) * r^2 + q2) * r^2 + q1) * r^2
        */
        vmulsd   .L__dble_pq4(%rip), %xmm1, %xmm1     /* p4 * r^2 */
        vsubsd   %xmm6,%xmm2,%xmm2                   /* (rh - c) */
        vmulsd   .L__dble_pq4+16(%rip), %xmm4, %xmm4  /* q4 * r^2 */
        vsubsd   %xmm3,%xmm9,%xmm9                   /* (rh - c) - rt aka rr */
        vshufpd  $0, %xmm8, %xmm5, %xmm5              /* r^2 in both halves */

        vmulpd   %xmm8, %xmm0, %xmm0    /* r^2/720 + r/120, r^3/6 + r^2/2 */
        vshufpd  $1, %xmm8, %xmm8, %xmm8              /* r^2, r */

        vaddsd   .L__dble_pq3(%rip), %xmm1, %xmm1     /* + p3 */
        vaddsd   .L__dble_pq3+16(%rip), %xmm4, %xmm4  /* + q3 */
        vsubsd   %xmm3,%xmm2,%xmm2                   /* (rh - c) - rt aka rr */

        vmulpd   %xmm5,%xmm1,%xmm1                   /* r^4, (p4 * r^2 + p3) * r^2 */
#ifdef TARGET_FMA
#        VFMADDSD        .L__dble_pq2+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213SD	(.L__dble_pq2+16(%rip),%xmm5,%xmm4)
#else
        vmulsd   %xmm5,%xmm4,%xmm4                   /* (q4 * r^2 + q3) * r^2 */
        vaddsd   .L__dble_pq2+16(%rip), %xmm4, %xmm4  /* + q2 */
#endif
        vmulsd   %xmm5,%xmm7,%xmm7                   /* xmm7 = r^3 */

        vmovhpd  .L__real_3FA5555555545D4E(%rip),%xmm8,%xmm8
        vmovapd   %xmm9,%xmm3                   /* Move rr */
        vmulsd   %xmm5,%xmm9,%xmm9                   /* r * r * rr */

        vaddsd   .L__dble_pq2(%rip), %xmm1, %xmm1     /* + p2 */
/*...   vmulsd   .L__dble_pq1+16(%rip), %xmm9, %xmm9 ...*/ /* r * r * rr * 0.5 */
        vmulsd   %xmm6, %xmm3, %xmm3                  /* r * rr */
        vaddpd   %xmm8, %xmm0, %xmm0    /* r^2/720 + r/120 + 1/24, r^3/6 + r^2/2 + r */

        leaq    .L__dble_sincostbl(%rip), %rdx /* Move table base address */
        addq    %rcx,%rcx
        addq    %rax,%rax

/*...   vmulsd   %xmm5,%xmm1,%xmm1 ...*/                  /* * r^2 */
/*...   vmulsd   %xmm5,%xmm4,%xmm4 ...*/                  /* * r^2 */
#ifdef TARGET_FMA
#        VFMADDSD        %xmm2,.L__dble_pq1+16(%rip), %xmm9,%xmm2
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm9,%xmm2)
#else
        vmulsd   .L__dble_pq1+16(%rip), %xmm9, %xmm9  /* r * r * rr * 0.5 */
        vaddsd   %xmm9,%xmm2,%xmm2                   /* cs = rr - r * r * rt * 0.5 */
#endif
        vmovsd  8(%rdx,%rax,8),%xmm8          /* ds2 in xmm8 */
        vmovsd  8(%rdx,%rcx,8),%xmm9          /* dc2 in xmm9 */
        /* xmm1 has dp, xmm4 has dq,
           xmm9 is scratch
           xmm2 has cs, xmm3 has cc
           xmm5 has r^2, xmm6 has r, xmm7 has r^3
           xmm8 is ds2 */

#ifdef TARGET_FMA
/*..    VFMADDSD        .L__dble_pq1(%rip),%xmm5,%xmm1,%xmm1 ..*/
        vmulsd   %xmm5,%xmm1,%xmm1
        vaddsd   .L__dble_pq1(%rip), %xmm1, %xmm1
#        VFMADDSD        .L__dble_pq1+16(%rip),%xmm5,%xmm4,%xmm4
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm5,%xmm4)

/*..    VFMADDSD        %xmm2,%xmm7,%xmm1,%xmm1 ..*/
        vmulsd   %xmm7,%xmm1,%xmm1
        vaddsd   %xmm2,%xmm1,%xmm1
#        VFMSUBSD        %xmm3,%xmm4,%xmm5,%xmm4
	VFMS_213SD	(%xmm3,%xmm5,%xmm4)
#else
        vmulsd   %xmm5,%xmm1,%xmm1                   /* * r^2 */
        vaddsd   .L__dble_pq1(%rip), %xmm1, %xmm1     /* + p1 */
        vmulsd   %xmm5,%xmm4,%xmm4                   /* * r^2 */
        vaddsd   .L__dble_pq1+16(%rip), %xmm4, %xmm4  /* + q1 */
        vmulsd   %xmm7,%xmm1,%xmm1                   /* * r^3 */
        vaddsd   %xmm2,%xmm1,%xmm1                   /* + cs  == dp aka p(r) */
        vmulsd   %xmm5,%xmm4,%xmm4                   /* * r^2 == dq aka q(r) */
        vsubsd   %xmm3,%xmm4,%xmm4                   /* - cc  == dq aka q(r) */
#endif

        vmovapd   %xmm9,%xmm3                   /* dc2 in xmm3 */
        vmovsd   (%rdx,%rax,8),%xmm5          /* ds1 in xmm5 */
        vmovsd   (%rdx,%rcx,8),%xmm7          /* dc1 in xmm7 */
        vaddsd   %xmm6,%xmm1,%xmm1                   /* + r   == dp aka p(r) */
        vmovapd   %xmm8,%xmm2                   /* ds2 in xmm2 */

/*...   vmulsd   %xmm4,%xmm8,%xmm8 ...*/                  /* ds2 * dq */
/*...   vmulsd   %xmm4,%xmm9,%xmm9 ...*/                  /* dc2 * dq */
        vmovhpd  (%rdx),%xmm2,%xmm2                  /* high half is 1.0 */
        movq    %r8, %rcx

#ifdef TARGET_FMA
#        VFMADDSD        %xmm2,%xmm4,%xmm8,%xmm8
	VFMA_213SD	(%xmm2,%xmm4,%xmm8)
#        VFMADDSD        %xmm3,%xmm4,%xmm9,%xmm9
	VFMA_213SD	(%xmm3,%xmm4,%xmm9)
#else
        vmulsd   %xmm4,%xmm8,%xmm8                   /* ds2 * dq */
        vaddsd   %xmm2,%xmm8,%xmm8                   /* ds2 + ds2*dq */
        vmulsd   %xmm4,%xmm9,%xmm9                   /* dc2 * dq */
        vaddsd   %xmm3,%xmm9,%xmm9                   /* dc2 + dc2*dq */
#endif
        shrq    $32, %rcx
        leaq    .L__two_to_jby32_table(%rip),%rdx

#ifdef TARGET_FMA
#        VFMADDSD        %xmm8,%xmm1,%xmm3,%xmm8
	VFMA_231SD	(%xmm1,%xmm3,%xmm8)
#else
        vmulsd   %xmm1,%xmm3,%xmm3                   /* dc2 * dp */
        vaddsd   %xmm3,%xmm8,%xmm8                   /* (ds2 + ds2*dq) + dc2*dp */
#endif
        vmulsd   %xmm1,%xmm2,%xmm2                   /* ds2 * dp */
        movq    $0x1f,%rax
        vmovapd %xmm4,%xmm6                   /* xmm6 = dq */
        andl    %ecx,%eax

/*...   vaddsd   %xmm3,%xmm8,%xmm8 ...*/                  /* (ds2 + ds2*dq) + dc2*dp */
        vsubsd   %xmm2,%xmm9,%xmm9                   /* (dc2 + dc2*dq) - ds2*dp */

        subl    %eax,%ecx

        vmovapd %xmm5,%xmm3                   /* xmm3 = ds1 */
        vshufpd  $3, %xmm1,%xmm2,%xmm2               /* r^4, 1.0 */
        sarl    $5,%ecx

#ifdef TARGET_FMA
#        VFMADDSD        %xmm8,%xmm5,%xmm4,%xmm8
	VFMA_231SD	(%xmm5,%xmm4,%xmm8)
#        VFNMADDSD       %xmm9,%xmm5,%xmm1,%xmm9
	VFNMA_231SD	(%xmm5,%xmm1,%xmm9)
#        VFMADDSD        %xmm9,%xmm7,%xmm6,%xmm9
	VFMA_231SD	(%xmm7,%xmm6,%xmm9)
#else
        vmulsd   %xmm5,%xmm4,%xmm4                   /* ds1 * dq */
        vaddsd   %xmm4,%xmm8,%xmm8                   /* ((ds2...) + dc2*dp) + ds1*dq */
        vmulsd   %xmm1,%xmm5,%xmm5                   /* ds1 * dp */
        vsubsd   %xmm5,%xmm9,%xmm9                   /* (() - ds2*dp) - ds1*dp */
        vmulsd   %xmm7,%xmm6,%xmm6                   /* dc1 * dq */
        vaddsd   %xmm6,%xmm9,%xmm9                   /* + dc1*dq */
#endif

/*...   vmulsd   %xmm7,%xmm1,%xmm1 ...*/                  /* dc1 * dp */
/*...   vaddsd   %xmm4,%xmm8,%xmm8 ...*/                  /* ((ds2...) + dc2*dp) + ds1*dq */
/*...   vsubsd   %xmm5,%xmm9,%xmm9 ...*/                  /* (() - ds2*dp) - ds1*dp */

        vmulpd   %xmm0,%xmm2,%xmm2          /* r^6/720+r^5/120+r^4/24, r^3/6+r^2/2+r */
        vmovsd  (%rdx,%rax,8),%xmm5
        vaddsd   %xmm3,%xmm8,%xmm8                   /* + ds1 */
/*...   vaddsd   %xmm6,%xmm9,%xmm9 ...*/                  /* + dc1*dq */

        vshufpd  $1, %xmm2, %xmm2, %xmm2
#ifdef TARGET_FMA
#        VFMADDSD        %xmm8,%xmm7,%xmm1,%xmm1
	VFMA_213SD	(%xmm8,%xmm7,%xmm1)
#else
        vmulsd   %xmm7,%xmm1,%xmm1                   /* dc1 * dp */
        vaddsd   %xmm8,%xmm1,%xmm1                   /* sin(x) = Cp(r) + (S+Sq(r)) */
#endif
        vaddsd   %xmm7,%xmm9,%xmm9                   /* cos(x) = (C + Cq(r)) + Sq(r) */

        /* Now start exp */

        vaddsd   %xmm2, %xmm0, %xmm0

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
#ifdef TARGET_FMA
#        VFMADDSD        %xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213SD	(%xmm5,%xmm5,%xmm0)
#else
        vmulsd   %xmm5,%xmm0,%xmm0
        vaddsd   %xmm5,%xmm0,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

        /* deal with denormal results */
        movq    $1, %rdx
        movq    $1, %rax
        addq    $1022, %rcx     /* add bias */
        cmovleq %rcx, %rdx
        cmovleq %rax, %rcx
        shlq    $52,%rcx        /* build 2^n */
        addq    $1023, %rdx     /* add bias */
        shlq    $52,%rdx        /* build 2^n */
        movq    %rdx,104(%rsp)   /* get 2^n to memory */
        vmulsd   104(%rsp),%xmm0,%xmm0  /* result *= 2^n */

        /* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
        /* Step 3. Reconstitute. */
        movq    %rcx,104(%rsp)   /* get 2^n to memory */
        vmulsd   104(%rsp),%xmm0,%xmm0  /* result *= 2^n */
        vmulsd   %xmm0,%xmm1,%xmm1
        vmulsd   %xmm9,%xmm0,%xmm0

#if defined(_WIN64)
        movq    24(%rsp),I1
        vmovdqu  128(%rsp),%ymm6
        vmovdqu  160(%rsp),%ymm7
        vmovdqu  192(%rsp),%ymm8
        vmovdqu  224(%rsp),%ymm9
#endif
        vmovlpd  %xmm1,8(I1)
        vmovlpd  %xmm0,(I1)

        movq    %rbp, %rsp
        popq    %rbp
        ret


LBL(.L__fsz_exp_shortcuts):
        pushq   %rbp
        movq    %rsp, %rbp
        andq	$-16, %rsp
        subq    $32, %rsp

	vmovapd	%xmm1,(%rsp)
	movq	%rax, 16(%rsp)
	movq	I1, 24(%rsp)

	CALL(ENT(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA)))


	vmovapd	%xmm0,%xmm5
	vmovsd  (%rsp),%xmm0
        vmovsd  .L__dble_sincostbl(%rip), %xmm1  /* 1.0 */
	movq	16(%rsp),%rax
        vmovapd   %xmm0,%xmm2
        vmovapd   %xmm0,%xmm3
        shrq    $48,%rax
        cmpl    $0x03f20,%eax
        jl      LBL(.L__fsz_exp_small)
        vmovapd   %xmm0,%xmm4
        vmulsd   %xmm0,%xmm0,%xmm0
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   %xmm4,%xmm4,%xmm4

        vmulsd   .L__dble_dsin_c6(%rip),%xmm0,%xmm0    /* x2 * s6 */
        vmulsd   .L__dble_dcos_c6(%rip),%xmm2,%xmm2    /* x2 * c6 */
        vaddsd   .L__dble_dsin_c5(%rip),%xmm0,%xmm0    /* + s5 */
        vaddsd   .L__dble_dcos_c5(%rip),%xmm2,%xmm2    /* + c5 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__dble_dsin_c4(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c4(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c4(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c4(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_dsin_c3(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c3(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c3(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c3(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_dsin_c2(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_dsin_c2(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c2(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c2(%rip),%xmm4,%xmm2)
#	VFMADDSD	.L__dble_pq1(%rip),%xmm4,%xmm0,%xmm0
	VFMA_213SD	(.L__dble_pq1(%rip),%xmm4,%xmm0)
#	VFMADDSD	.L__dble_dcos_c1(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_dcos_c1(%rip),%xmm4,%xmm2)
	vmulsd		%xmm4,%xmm0,%xmm0
#	VFMADDSD	.L__dble_pq1+16(%rip),%xmm4,%xmm2,%xmm2
	VFMA_213SD	(.L__dble_pq1+16(%rip),%xmm4,%xmm2)
#	VFMADDSD	%xmm3,%xmm3,%xmm0,%xmm0
	VFMA_213SD	(%xmm3,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,%xmm4,%xmm2,%xmm1
	VFMA_231SD	(%xmm4,%xmm2,%xmm1)
#else
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s5 + ...) */
        vaddsd   .L__dble_dsin_c4(%rip),%xmm0,%xmm0    /* + s4 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c5 + ...) */
        vaddsd   .L__dble_dcos_c4(%rip),%xmm2,%xmm2    /* + c4 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s4 + ...) */
        vaddsd   .L__dble_dsin_c3(%rip),%xmm0,%xmm0    /* + s3 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c4 + ...) */
        vaddsd   .L__dble_dcos_c3(%rip),%xmm2,%xmm2    /* + c3 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s3 + ...) */
        vaddsd   .L__dble_dsin_c2(%rip),%xmm0,%xmm0    /* + s2 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c3 + ...) */
        vaddsd   .L__dble_dcos_c2(%rip),%xmm2,%xmm2    /* + c2 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x2 * (s2 + ...) */
        vaddsd   .L__dble_pq1(%rip),%xmm0,%xmm0        /* + s1 */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c2 + ...) */
        vaddsd   .L__dble_dcos_c1(%rip),%xmm2,%xmm2    /* + c1 */
        vmulsd   %xmm4,%xmm0,%xmm0                     /* x3 * (s1 + ...) */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (c1 + ...) */
        vaddsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2     /* - 0.5 */
        vmulsd   %xmm3,%xmm0,%xmm0                     /* x3 */
        vaddsd   %xmm3,%xmm0,%xmm0                     /* x + x3 * (...) done */
        vmulsd   %xmm4,%xmm2,%xmm2                     /* x2 * (0.5 + ...) */
        vaddsd   %xmm2,%xmm1,%xmm1                     /* 1.0 - 0.5x2 + (...) done */
#endif
	jmp	LBL(.L__fsz_exp_done1)

LBL(.L__fsz_exp_small):
        cmpl    $0x03e40,%eax
        jl      LBL(.L__fsz_exp_done1)
        /* return sin(x) = x - x * x * x * 1/3! */
        /* return cos(x) = 1.0 - x * x * 0.5 */
        vmulsd   %xmm2,%xmm2,%xmm2
        vmulsd   .L__dble_pq1(%rip),%xmm3,%xmm3
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm3,%xmm0
	VFMA_231SD	(%xmm2,%xmm3,%xmm0)
#	VFMADDSD	%xmm1,.L__dble_pq1+16(%rip),%xmm2,%xmm1
	VFMA_231SD	(.L__dble_pq1+16(%rip),%xmm2,%xmm1)
#else
        vmulsd   %xmm2,%xmm3,%xmm3
        vaddsd   %xmm3,%xmm0,%xmm0
        vmulsd   .L__dble_pq1+16(%rip),%xmm2,%xmm2
        vaddsd   %xmm2,%xmm1,%xmm1
#endif

LBL(.L__fsz_exp_done1):
	movq	24(%rsp),I1
	vmulsd	%xmm5,%xmm0,%xmm0
	vmulsd	%xmm5,%xmm1,%xmm1
	vmovlpd  %xmm0,8(I1)
	vmovlpd  %xmm1,(I1)
	movq	%rbp, %rsp
	popq	%rbp
        ret

LBL(.L_cdexp_inf):
	vmovsd	.L__real_infinity(%rip), %xmm0
	vmovlpd  %xmm0,8(I1)
	vmovlpd  %xmm0,(I1)
        ret

LBL(.L_cdexp_ninf):
	jp	LBL(.L_cdexp_cvt_nan)
	xorq	%rax, %rax
	movq	%rax, 8(I1)
	movq	%rax, (I1)
        ret

LBL(.L_cdexp_cvt_nan):
	movsd	.L__real_infinity+8(%rip), %xmm1
	vorpd	%xmm1, %xmm0, %xmm0
	vmovlpd  %xmm0,8(I1)
	vmovlpd  %xmm0,(I1)
        ret

LBL(.L__fsz_exp_hard):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $32, %rsp

	vmovlpd	%xmm1,(%rsp)
	movq	I1, 24(%rsp)

	CALL(ENT(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA)))


	vmovlpd	%xmm0,8(%rsp)
	vmovsd  (%rsp),%xmm0
	CALL(ENT(__mth_i_dsincos))

        vmovsd  8(%rsp), %xmm5
        jmp LBL(.L__fsz_exp_done1)

        ELF_FUNC(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsz_exp_,TARGET_VEX_OR_FMA))


/* ============================================================ */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA)):

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
	vmovaps	%xmm0, %xmm4
	vmovaps	%xmm1, %xmm5
        vmovaps  %xmm0, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3
	vandps	.L4_fvspow_infinity_mask(%rip), %xmm4, %xmm4
	vandps	.L4_fvspow_infinity_mask(%rip), %xmm5, %xmm5
	vcmpps	$2, %xmm3, %xmm2, %xmm2
	vcmpps	$0, .L4_fvspow_infinity_mask(%rip), %xmm4, %xmm4
	vcmpps	$0, .L4_fvspow_infinity_mask(%rip), %xmm5, %xmm5
	vorps	%xmm4, %xmm2, %xmm2
	/* Store input arguments onto stack */
        vmovaps  %xmm0, _SX0(%rsp)
	vorps	%xmm5, %xmm2, %xmm2
        vmovaps  %xmm1, _SY0(%rsp)
	vmovmskps %xmm2, %r8d
	test	$15, %r8d
	jnz	LBL(.L__Scalar_fvspow)

	/* Convert x0, x1 to dbl and call log */
	vcvtps2pd %xmm0, %xmm0
        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))


	/* dble(y) * dlog(x) */
        vmovlps  _SY0(%rsp), %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vmulpd	%xmm1, %xmm0, %xmm0
        vmovapd  %xmm0, _SR0(%rsp)

	/* Convert x2, x3 to dbl and call log */
        vmovlps  _SX2(%rsp), %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm0
        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))


	/* dble(y) * dlog(x) */
        vmovlps  _SY2(%rsp), %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vmulpd	%xmm0, %xmm1, %xmm1
	vmovapd	_SR0(%rsp), %xmm0
	CALL(ENT(ASM_CONCAT(__fvs_exp_dbl_,TARGET_VEX_OR_FMA)))


        movq    %rbp, %rsp
        popq    %rbp
	ret

LBL(.L__Scalar_fvspow):
        CALL(ENT(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovss   _SX1(%rsp), %xmm0
        vmovss   _SY1(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovss   _SX2(%rsp), %xmm0
        vmovss   _SY2(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovss   _SX3(%rsp), %xmm0
        vmovss   _SY3(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA))


/* ========================================================================= */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA)):


	vmovaps	%xmm1, %xmm2
	vmovaps	%xmm1, %xmm3
	vmovaps	%xmm1, %xmm4
	vshufps	$0, %xmm0, %xmm2, %xmm2
	vshufps	$0, %xmm0, %xmm3, %xmm3
	vshufps	$0, %xmm0, %xmm4, %xmm4
	vmovd	%xmm0, %eax
	vmovd	%xmm1, %ecx
	vcmpps	$0, .L4_100(%rip), %xmm2, %xmm2
	vcmpps	$0, .L4_101(%rip), %xmm3, %xmm3
	vandps	.L4_102(%rip), %xmm4, %xmm4
	vmovdqa	%xmm4, %xmm5
	vorps	%xmm3, %xmm2, %xmm2
	vpcmpeqd	.L4_103(%rip), %xmm5, %xmm5
	vorps	%xmm5, %xmm2, %xmm2
	vmovmskps %xmm2, %r8d
	test	$15, %r8d
	jnz	LBL(.L__Special_Pow_Cases)
	vcomiss	.L4_104(%rip), %xmm4
	ja	LBL(.L__Y_is_large)
	vcomiss	.L4_105(%rip), %xmm4
	jb	LBL(.L__Y_near_zero)
	vunpcklps %xmm1, %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm0
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$128, %rsp
	vmovsd	%xmm1, 0(%rsp)
	CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))

	vmulsd	0(%rsp), %xmm0, %xmm0
	CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))

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
        mov	.L4_104(%rip), %edx
        cmovne	.L4_100+12(%rip), %edx
        vmovd	%edx, %xmm0
        vdivss  %xmm0, %xmm1, %xmm0     /* Generate divide by zero op when y < 0 */
#endif
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_9b):
	/* x is +inf, test sign of y */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_100+12(%rip), %eax
	vmovd	%eax, %xmm0
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
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10b):
	test	.L4_102+8(%rip), %ecx
	cmovne 	.L4_108(%rip), %eax
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10c):
	/* x is -inf, test sign of y */
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10d)
	/* x is -inf, inty != 1 */
	mov	.L4_102+4(%rip), %eax
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_100+12(%rip), %eax
	vmovd	%eax, %xmm0
	ret
LBL(.L__Special_Case_10d):
	/* x is -inf, inty == 1 */
	test	.L4_102+8(%rip), %ecx
	cmovne	.L4_102+8(%rip), %eax
	vmovd	%eax, %xmm0
	ret

LBL(.L__Special_Case_10e):
	/* x is negative */
	vcomiss	.L4_104(%rip), %xmm4
	ja	LBL(.L__Y_is_large)
	test	$3, %r8d
	je	LBL(.L__Special_Case_10f)
	and	.L4_102(%rip), %eax
	vmovd	%eax, %xmm0
	vunpcklps %xmm1, %xmm1, %xmm1
	vcvtps2pd %xmm1, %xmm1
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm0
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$128, %rsp
	vmovsd	%xmm1, 0(%rsp)
	cmp	$1, %r8d
	je	LBL(.L__Special_Case_10g)

	CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))

	vmulsd	0(%rsp), %xmm0, %xmm0
	CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))

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
	vsqrtss	%xmm0, %xmm0, %xmm0 /* Generate an invalid op */
#endif
	vmovd	%eax, %xmm0
	ret

LBL(.L__Special_Case_10g):
	CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))

	vmulsd	0(%rsp), %xmm0, %xmm0
	CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))

	vmovaps	%xmm0, %xmm1
	vxorps	%xmm0, %xmm0, %xmm0
	vsubps	%xmm1, %xmm0, %xmm0
	movq	%rbp, %rsp
	popq	%rbp
	ret


LBL(.L__Special_Case_1):
LBL(.L__Special_Case_5):
	vmovss	.L4_101(%rip), %xmm0
	ret
LBL(.L__Special_Case_2):
	vsqrtss	%xmm0, %xmm1, %xmm1
	vmulss	%xmm1, %xmm0, %xmm0
	ret
LBL(.L__Special_Case_3):
	vsqrtss	%xmm0, %xmm0, %xmm0
	ret
LBL(.L__Special_Case_4):
	vsqrtss	%xmm0, %xmm0, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm0
	ret
LBL(.L__Special_Case_6):
	test	.L4_106(%rip), %eax
	je	LBL(.L__Special_Pow_Case_7)
	or	.L4_107(%rip), %eax
	vmovd	%eax, %xmm0
	ret

LBL(.L__Special_Case_7):
	test	.L4_106(%rip), %ecx
	je	LBL(.L__Y_is_large)
	or	.L4_107(%rip), %ecx
	vmovd	%ecx, %xmm0
	ret

/* This takes care of all the large Y cases */
LBL(.L__Y_is_large):
	vcomiss	.L4_103(%rip), %xmm1
	vandps	.L4_102(%rip), %xmm0, %xmm0
	jb	LBL(.L__Y_large_negative)
LBL(.L__Y_large_positive):
	/* If abs(x) < 1.0, return 0 */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return Inf */
	vcomiss	.L4_101(%rip), %xmm0
	jb	LBL(.L__Y_large_pos_0)
	je	LBL(.L__Y_large_pos_1)
LBL(.L__Y_large_pos_i):
	vmovss	.L4_102+4(%rip), %xmm0
	ret
LBL(.L__Y_large_pos_1):
	vmovss	.L4_101(%rip), %xmm0
	ret
/* */
LBL(.L__Y_large_negative):
	/* If abs(x) < 1.0, return Inf */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return 0 */
	vcomiss	.L4_101(%rip), %xmm0
	jb	LBL(.L__Y_large_pos_i)
	je	LBL(.L__Y_large_pos_1)
LBL(.L__Y_large_pos_0):
	vmovss	.L4_103(%rip), %xmm0
	ret

LBL(.L__Y_near_zero):
	vmovss	.L4_101(%rip), %xmm0
	ret

/* -------------------------------------------------------------------------- */

        ELF_FUNC(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fss_pow_,TARGET_VEX_OR_FMA))


/* ========================================================================= */
#define _DX0 0
#define _DX1 8

#define _DY0 16
#define _DY1 24

#define _DR0 32
#define _DR1 40

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA)):

        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
	vmovapd	%xmm0, %xmm4
	vmovapd	%xmm1, %xmm5
        vmovapd  %xmm0, %xmm2
	vxorpd	%xmm3, %xmm3, %xmm3
	vandpd	.L4_fvdpow_infinity_mask(%rip), %xmm4, %xmm4
	vandpd	.L4_fvdpow_infinity_mask(%rip), %xmm5, %xmm5
	vcmppd	$2, %xmm3, %xmm2, %xmm2
	vcmppd	$0, .L4_fvdpow_infinity_mask(%rip), %xmm4, %xmm4
	vcmppd	$0, .L4_fvdpow_infinity_mask(%rip), %xmm5, %xmm5
	vorpd	%xmm4, %xmm2, %xmm2
	/* Store input arguments onto stack */
        vmovapd  %xmm0, _DX0(%rsp)
	vorpd	%xmm5, %xmm2, %xmm2
        vmovapd  %xmm1, _DY0(%rsp)
	vmovmskpd %xmm2, %r8d
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdpow)

#if defined(_WIN64)
	vmovdqu	%ymm6, 96(%rsp)
#endif
	/* Call log long version */
        CALL(ENT(ASM_CONCAT(__fvd_log_long_,TARGET_VEX_OR_FMA)))


	/* Head in xmm0, tail in xmm1 */
	/* Carefully compute w = y * log(x) */

	/* Split y into hy (head) + ty (tail). */
        vmovapd  _DY0(%rsp), %xmm2  			/* xmm2 has copy y */
        vmovapd  _DY0(%rsp), %xmm5
        vmovapd	%xmm0, %xmm3

	vandpd   .L__real_fffffffff8000000(%rip), %xmm2, %xmm2	/* xmm2 = head(y) */

	vmulpd   _DY0(%rsp), %xmm3, %xmm3			/* y * hx */
	vsubpd   %xmm2, %xmm5, %xmm5				/* ty */

        vmovapd	%xmm0, %xmm4
        vmovapd	%xmm0, %xmm6
#ifdef TARGET_FMA
#	VFMSUBPD	%xmm3,%xmm2,%xmm4,%xmm4
	VFMS_213PD	(%xmm3,%xmm2,%xmm4)
#	VFMADDPD	%xmm4,%xmm5,%xmm6,%xmm4
	VFMA_231PD	(%xmm5,%xmm6,%xmm4)
#	VFMADDPD	%xmm4,%xmm1,%xmm2,%xmm4
	VFMA_231PD	(%xmm1,%xmm2,%xmm4)
#	VFMADDPD	%xmm4,%xmm5,%xmm1,%xmm1
	VFMA_213PD	(%xmm4,%xmm5,%xmm1)
#else
        vmulpd   %xmm2, %xmm4, %xmm4				/* hy*hx */
        vsubpd   %xmm3, %xmm4, %xmm4				/* hy*hx - y*hx */
        vmulpd   %xmm5, %xmm6, %xmm6				/* ty*hx */
        vaddpd   %xmm6, %xmm4, %xmm4				/* + ty*hx */
        vmulpd   %xmm1, %xmm2, %xmm2				/* hy*tx */
        vaddpd   %xmm2, %xmm4, %xmm4				/* + hy*tx */
        vmulpd   %xmm5, %xmm1, %xmm1				/* ty*tx */
        vaddpd   %xmm4, %xmm1, %xmm1				/* + ty*tx */
#endif


        vmovapd  %xmm3, %xmm0
        vaddpd   %xmm1, %xmm0, %xmm0
        vsubpd   %xmm0, %xmm3, %xmm3
        vaddpd   %xmm3, %xmm1, %xmm1

	CALL(ENT(ASM_CONCAT(__fvd_exp_long_,TARGET_VEX_OR_FMA)))


#if defined(_WIN64)
	vmovdqu	96(%rsp), %ymm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
	ret

LBL(.L__Scalar_fvdpow):
        CALL(ENT(ASM_CONCAT(__fsd_pow_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR0(%rsp)

        vmovsd   _DX1(%rsp), %xmm0
        vmovsd   _DY1(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__fsd_pow_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR1(%rsp)

        vmovapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA))


/* ========================================================================= */

	.text
	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fsd_pow_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_pow_,TARGET_VEX_OR_FMA)):


	pushq	%rbp
	movq	%rsp, %rbp
	subq	$64, %rsp

	/* Save y for mpy and broadcast for special case tests */
	vmovsd	%xmm1, 0(%rsp)
	vmovsd	%xmm0, 8(%rsp)

#if defined(_WIN64)
	vmovdqu	%ymm6, 32(%rsp)
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

	vmovsd	.L4_D105(%rip), %xmm4

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

	vandpd	%xmm1, %xmm4, %xmm4

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
	vcomisd	.L4_D107(%rip), %xmm4
	ja	LBL(.L__DY_is_large)
	vcomisd	.L4_D108(%rip), %xmm4
	jb	LBL(.L__DY_near_zero)

LBL(.L__D_algo_start):

	CALL(ENT(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA)))


	/* Head in xmm0, tail in xmm1 */
	/* Carefully compute w = y * log(x) */

	/* Split y into hy (head) + ty (tail). */

        vmovsd   0(%rsp), %xmm2  			/* xmm2 has copy y */
        vmovsd   0(%rsp), %xmm5
        vmovapd   %xmm0, %xmm3

	vandpd   .L__real_fffffffff8000000(%rip), %xmm2, %xmm2	/* xmm2 = head(y) */

	vmulsd   0(%rsp), %xmm3, %xmm3				/* y * hx */
	vsubsd   %xmm2, %xmm5, %xmm5				/* ty */

        vmovapd   %xmm0, %xmm4
#ifdef TARGET_FMA
#	VFMSUBSD	%xmm3,%xmm4,%xmm2,%xmm4
	VFMS_213SD	(%xmm3,%xmm2,%xmm4)
	vmovapd		%xmm0, %xmm6
#	VFMADDSD	%xmm4,%xmm5,%xmm6,%xmm4
	VFMA_231SD	(%xmm5,%xmm6,%xmm4)
#	VFMADDSD	%xmm4,%xmm2,%xmm1,%xmm4
	VFMA_231SD	(%xmm2,%xmm1,%xmm4)
#	VFMADDSD	%xmm4,%xmm1,%xmm5,%xmm1
	VFMA_213SD	(%xmm4,%xmm5,%xmm1)
#else
        vmulsd   %xmm2, %xmm4, %xmm4				/* hy*hx */
        vsubsd   %xmm3, %xmm4, %xmm4				/* hy*hx - y*hx */
        vmovapd   %xmm0, %xmm6
        vmulsd   %xmm5, %xmm6, %xmm6				/* ty*hx */
        vaddsd   %xmm6, %xmm4, %xmm4				/* + ty*hx */
        vmulsd   %xmm1, %xmm2, %xmm2				/* hy*tx */
        vaddsd   %xmm2, %xmm4, %xmm4				/* + hy*tx */

        vmulsd   %xmm5, %xmm1, %xmm1				/* ty*tx */
        vaddsd   %xmm4, %xmm1, %xmm1				/* + ty*tx */
#endif

        vmovapd   %xmm3, %xmm0
        vaddsd   %xmm1, %xmm0, %xmm0
        vsubsd   %xmm0, %xmm3, %xmm3
        vaddsd   %xmm3, %xmm1, %xmm1

        CALL(ENT(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA)))


LBL(.L__Dpop_and_return):
#if defined(_WIN64)
	vmovdqu	32(%rsp), %ymm6
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
	movq	.L4_D105(%rip), %rdx
	cmovneq	.L4_D106(%rip), %rdx
	movq	%rdx, %xmm0
	vdivsd	%xmm0, %xmm1, %xmm0	/* Generate divide by zero op when y < 0 */
#endif
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_9b):
	/* x is +inf, test sign of y */
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D101+8(%rip), %rax
	vmovd 	%rax, %xmm0
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
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10b):
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D109(%rip), %rax
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10c):
	/* x is -inf, test sign of y */
	cmpq	$1, %r8
	je	LBL(.L__DSpecial_Case_10d)
	/* x is -inf, inty != 1 */
	movq	.L4_D104(%rip), %rax
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D101+8(%rip), %rax
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10d):
	/* x is -inf, inty == 1 */
	testq	.L4_D105+8(%rip), %rcx
	cmovneq	.L4_D105+8(%rip), %rax
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_10e):
	/* x is negative */
	vcomisd	.L4_D107(%rip), %xmm4
	ja	LBL(.L__DY_is_large)
	testq	$3, %r8
	je	LBL(.L__DSpecial_Case_10f)
	andq	.L4_D105(%rip), %rax
	vmovd 	%rax, %xmm0

	/* Do the regular pow stuff */
	cmp 	$1, %r8d
	jne	LBL(.L__D_algo_start)

LBL(.L__DSpecial_Case_10g):
        CALL(ENT(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA)))


	/* Head in xmm0, tail in xmm1 */
	/* Carefully compute w = y * log(x) */

	/* Split y into hy (head) + ty (tail). */

        vmovsd   0(%rsp), %xmm2  			/* xmm2 has copy y */
        vmovsd   0(%rsp), %xmm5
        vmovapd   %xmm0, %xmm3

	vandpd   .L__real_fffffffff8000000(%rip), %xmm2, %xmm2	/* xmm2 = head(y) */

	vmulsd   0(%rsp), %xmm3, %xmm3				/* y * hx */
	vsubsd   %xmm2, %xmm5, %xmm5				/* ty */

        vmovapd   %xmm0, %xmm4
#ifdef TARGET_FMA
#	VFMSUBSD	%xmm3,%xmm2,%xmm4,%xmm4
	VFMS_213SD	(%xmm3,%xmm2,%xmm4)
	vmovapd		%xmm0,%xmm6
#	VFMADDSD	%xmm4,%xmm5,%xmm6,%xmm4
	VFMA_231SD	(%xmm5,%xmm6,%xmm4)
#	VFMADDSD	%xmm4,%xmm1,%xmm2,%xmm4
	VFMA_231SD	(%xmm1,%xmm2,%xmm4)
#	VFMADDSD	%xmm4,%xmm1,%xmm5,%xmm1
	VFMA_213SD	(%xmm4,%xmm5,%xmm1)
#else
        vmulsd   %xmm2, %xmm4, %xmm4				/* hy*hx */
        vsubsd   %xmm3, %xmm4, %xmm4				/* hy*hx - y*hx */
        vmovapd   %xmm0, %xmm6
        vmulsd   %xmm5, %xmm6, %xmm6				/* ty*hx */
        vaddsd   %xmm6, %xmm4, %xmm4				/* + ty*hx */
        vmulsd   %xmm1, %xmm2, %xmm2				/* hy*tx */
        vaddsd   %xmm2, %xmm4, %xmm4				/* + hy*tx */

        vmulsd   %xmm5, %xmm1, %xmm1				/* ty*tx */
        vaddsd   %xmm4, %xmm1, %xmm1				/* + ty*tx */
#endif

        vmovapd   %xmm3, %xmm0
        vaddsd   %xmm1, %xmm0, %xmm0
        vsubsd   %xmm0, %xmm3, %xmm3
        vaddsd   %xmm3, %xmm1, %xmm1

        CALL(ENT(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA)))


	vmulsd	.L4_D102+8(%rip), %xmm0, %xmm0
	jmp	LBL(.L__Dpop_and_return)

/* Changing this on Sept 13, 2005, to return 0xfff80000 for neg ** neg */
LBL(.L__DSpecial_Case_10f):
	movq 	.L4_D109(%rip), %rax
	orq	.L4_D10a(%rip), %rax
#ifdef FMATH_EXCEPTIONS
        vsqrtsd	%xmm0, %xmm0, %xmm0 /* Generate an invalid op */
#endif
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_1):
LBL(.L__DSpecial_Case_5):
	vmovsd	.L4_D102(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_2):
	vsqrtsd	%xmm0, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm0, %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_3):
	vsqrtsd	%xmm0, %xmm0, %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_4):
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vsqrtsd	%xmm0, %xmm0, %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DSpecial_Case_6):
	testq	.L4_D10b(%rip), %rax
	je	LBL(.L__DSpecial_Pow_Case_7)
	orq	.L4_D10a(%rip), %rax
	vmovd 	%rax, %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DSpecial_Case_7):
	testq	.L4_D10b(%rip), %rcx
	je	LBL(.L__DY_is_large)
	orq	.L4_D10a(%rip), %rcx
	vmovd 	%rcx, %xmm0
	jmp	LBL(.L__Dpop_and_return)

/* This takes care of all the large Y cases */
LBL(.L__DY_is_large):
	vcomisd	.L4_D106(%rip), %xmm1
	vandpd	.L4_D105(%rip), %xmm0, %xmm0
	jb	LBL(.L__DY_large_negative)
LBL(.L__DY_large_positive):
	/* If abs(x) < 1.0, return 0 */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return Inf */
	vcomisd	.L4_D102(%rip), %xmm0
	jb	LBL(.L__DY_large_pos_0)
	je	LBL(.L__DY_large_pos_1)
LBL(.L__DY_large_pos_i):
	vmovsd	.L4_D104(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)
LBL(.L__DY_large_pos_1):
	vmovsd	.L4_D102(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)
/* */
LBL(.L__DY_large_negative):
	/* If abs(x) < 1.0, return Inf */
	/* If abs(x) == 1.0, return 1.0 */
	/* If abs(x) > 1.0, return 0 */
	vcomisd	.L4_D102(%rip), %xmm0
	jb	LBL(.L__DY_large_pos_i)
	je	LBL(.L__DY_large_pos_1)
LBL(.L__DY_large_pos_0):
	vmovsd	.L4_D106(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)

LBL(.L__DY_near_zero):
	vmovsd	.L4_D102(%rip), %xmm0
	jmp	LBL(.L__Dpop_and_return)

/* -------------------------------------------------------------------------- */

        ELF_FUNC(ASM_CONCAT(__fsd_pow_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_pow_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        vcomisd         %xmm0, %xmm0
        jp     LBL(.LB_NZERO_SD_VEX)

	vcomisd	.L__np_ln_lead_table(%rip), %xmm0        /* Equal to 0.0? */
	jne	LBL(.LB_NZERO_SD_VEX)
	vmovsd	.L__two_to_jby32_table(%rip), %xmm0
	RZ_POP
	rep
	ret


LBL(.LB_NZERO_SD_VEX):


        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm0,%xmm3,%xmm3

	/* Set n = nearest integer to r */
	vcomisd	.L__real_ln_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	vcomisd	.L__real_ln_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_ninf)
	vcvtpd2dq %xmm3,%xmm4	/* convert to integer */
	vcvtdq2pd %xmm4,%xmm1	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
/*...	vmovsd	.L__real_log2_by_32_lead(%rip),%xmm2 ...*/

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,.L__real_log2_by_32_lead(%rip),%xmm1,%xmm0
	VFNMA_231SD	(.L__real_log2_by_32_lead(%rip),%xmm1,%xmm0)
#else
	vmulsd	.L__real_log2_by_32_lead(%rip),%xmm1,%xmm2
	vsubsd	%xmm2,%xmm0,%xmm0	/* r1 in xmm0, */
#endif
	vmovd	%xmm4,%ecx
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
/*...	vmulsd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1 ...*/

	/* j = n & 0x0000001f; */
	movq	$0x1f,%rax
	andl	%ecx,%eax
	vmovapd	%xmm0,%xmm2

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%eax,%ecx
	sarl	$5,%ecx

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2
	VFMA_231SD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2)
#else
	vmulsd  .L__real_log2_by_32_tail(%rip),%xmm1,%xmm1
	vaddsd	%xmm1,%xmm2,%xmm2    /* r = r1 + r2 */
#endif

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	vmovapd	%xmm2,%xmm1
	vmovsd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm0

	vmulsd	%xmm2,%xmm1,%xmm1
	vmovapd	%xmm1,%xmm4

#ifdef TARGET_FMA
#	VFMADDSD	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm2,%xmm3
	VFMA_213SD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFMADDSD	.L__real_3fe0000000000000(%rip),%xmm0,%xmm2,%xmm0
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
#else
	vmulsd	%xmm2,%xmm3,%xmm3
	vmulsd	%xmm2,%xmm0,%xmm0
	vaddsd	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm0,%xmm0
#endif

	vmulsd		%xmm1,%xmm4,%xmm4
#ifdef TARGET_FMA
#	VFMADDSD	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm2,%xmm3
	VFMA_213SD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMADDSD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#	VFMADDSD	%xmm0,%xmm3,%xmm4,%xmm0
	VFMA_231SD	(%xmm3,%xmm4,%xmm0)
#else
	vmulsd	%xmm2,%xmm3,%xmm3
	vmulsd	%xmm1,%xmm0,%xmm0
	vaddsd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3
	vaddsd	%xmm2,%xmm0,%xmm0
	vmulsd	%xmm4,%xmm3,%xmm3
	vaddsd	%xmm3,%xmm0,%xmm0
#endif

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%rdx,%rax,8),%xmm5
	/* deal with infinite results */
        movslq	%ecx,%rcx

#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213SD	(%xmm5,%xmm5,%xmm0)
#else
	vmulsd	%xmm5,%xmm0,%xmm0
	vaddsd	%xmm5,%xmm0,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

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
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

LBL(.Lfinal_check):
	RZ_POP
	rep
	ret

LBL(.L_inf):
	vmovsd	.L__real_infinity(%rip),%xmm0
	jmp	LBL(.Lfinal_check)

LBL(.L_ninf):
	jp	LBL(.L_cvt_nan)
	xorq	%rax, %rax
	vmovd 	%rax,%xmm0
	jmp	LBL(.Lfinal_check)

LBL(.L_sinh_ninf):
        jp      LBL(.L_cvt_nan)
	vmovsd   .L__real_ninfinity(%rip),%xmm0
        jmp     LBL(.Lfinal_check)

LBL(.L_cosh_ninf):
        jp      LBL(.L_cvt_nan)
        vmovsd   .L__real_infinity(%rip),%xmm0
        jmp     LBL(.Lfinal_check)

LBL(.L_cvt_nan):
	xorq	%rax, %rax
	vmovd 	%rax,%xmm1
	vmovsd	.L__real_infinity+8(%rip),%xmm1
	vorpd	%xmm1, %xmm0, %xmm0
	jmp	LBL(.Lfinal_check)


	ELF_FUNC(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA))


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
 *   copyright  notice,   this  list  of   conditions  and  the
 *   following disclaimer.
 *
 *  Redistributions  in binary  form must reproduce  the above
 *   copyright  notice,   this  list  of   conditions  and  the
 *   following  disclaimer in  the  documentation and/or  other
 *   materials provided with the distribution.
 *
 *  Neither the  name of Advanced Micro Devices,  Inc. nor the
 *   names  of  its contributors  may  be  used  to endorse  or
 *   promote  products  derived   from  this  software  without
 *   specific prior written permission.
 *
 * THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
 * COIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
 * MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
 * DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
 * INC.  OR COIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
 * INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
 * INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
 * GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
 * BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
 * LIABILITY,  WHETHER IN COACT,  STRICT LIABILITY,  OR TORT
 * INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
 * OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * It is  licensee's responsibility  to comply with  any export
 * regulations applicable in licensee's jurisdiction.
 *
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
	.globl ENT(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	%xmm0, %xmm2
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulpd	%xmm0,%xmm3,%xmm3

	/* save x for later. */
	vandpd	.L__real_mask_unsign(%rip), %xmm2,%xmm2

        /* Set n = nearest integer to r */
	vcvtpd2dq %xmm3,%xmm4
	vcmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2,%xmm2
	leaq	.L__two_to_jby32_table(%rip),%r11
	vcvtdq2pd %xmm4,%xmm1
	vmovmskpd %xmm2, %r8d

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovapd	.L__real_log2_by_32_lead(%rip),%xmm2
/*	vmulpd	%xmm1,%xmm2,%xmm2 */
	vmovq	 %xmm4,RZ_OFF(24)(%rsp)
	testl	$3, %r8d
	jnz	LBL(.L__Scalar_fvdexp)

	/* r2 =   - n * logbaseof2_by_32_trail; */

#ifdef TARGET_FMA
#	VFNMADDPD	%xmm0,%xmm1,%xmm2,%xmm0
	VFNMA_231PD	(%xmm1,%xmm2,%xmm0)
#else
	vmulpd	%xmm1,%xmm2,%xmm2
	vsubpd	%xmm2,%xmm0,%xmm0 	/* r1 in xmm0, */
#endif

/*	vmulpd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1 */  	/* r2 in xmm1 */

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	movl	RZ_OFF(24)(%rsp),%ecx
	andl	%ecx,%r9d

	movl	RZ_OFF(20)(%rsp),%edx
	andl	%edx,%r8d
	vmovapd	%xmm0,%xmm2

	/* f1 = two_to_jby32_lead_table[j]; */
	/* f2 = two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%r9d,%ecx
	sarl	$5,%ecx
	subl	%r8d,%edx
	sarl	$5,%edx
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2
	VFMA_231PD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2)
#else
	vmulpd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1 	/* r2 in xmm1 */
	vaddpd	%xmm1,%xmm2,%xmm2    /* r = r1 + r2 */
#endif

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +
	 * r*r*( 5.00000000000000008883e-01 +
	 * r*( 1.66666666665260878863e-01 +
	 * r*( 4.16666666662260795726e-02 +
	 * r*( 8.33336798434219616221e-03 +
	 * r*( 1.38889490863777199667e-03 ))))));
	 * q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */

	vmovapd	%xmm2,%xmm1
	vmovapd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovapd	.L__real_3FC5555555548F7C(%rip),%xmm0

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

#ifdef TARGET_FMA
#	VFMADDPD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFMADDPD	.L__real_3fe0000000000000(%rip),%xmm0,%xmm2,%xmm0
	VFMA_213PD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
#else
	vmulpd	%xmm2,%xmm3,%xmm3	/* *x */
	vaddpd	 .L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3
	vmulpd	%xmm2,%xmm0,%xmm0	/* *x */
	vaddpd	 .L__real_3fe0000000000000(%rip),%xmm0,%xmm0
#endif

	vmulpd	%xmm2,%xmm1,%xmm1	/* x*x */
	vmovapd	%xmm1,%xmm4
	vmulpd	%xmm1,%xmm4,%xmm4	/* x^4 */

#ifdef TARGET_FMA
#	VFMADDPD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMADDPD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm1,%xmm0)
#	VFMADDPD	%xmm0,%xmm4,%xmm3,%xmm0
	VFMA_231PD	(%xmm4,%xmm3,%xmm0)
#else
	vmulpd	%xmm2,%xmm3,%xmm3	/* *x */
	vaddpd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3

	vmulpd	%xmm1,%xmm0,%xmm0	/* *x^2 */
	vaddpd	%xmm2,%xmm0,%xmm0	/* + x  */
	vmulpd	%xmm4,%xmm3,%xmm3	/* *x^4 */
	vaddpd	%xmm3,%xmm0,%xmm0	/* q = final sum */
#endif

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1023,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

/*	vaddpd	%xmm3,%xmm0,%xmm0 */	/* q = final sum */

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	vmovhpd	(%r11,%r8,8),%xmm5,%xmm5 	/* f1 + f2 */

	/* shufpd	$0,%xmm4,%xmm5 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm5,%xmm0)
#else
	vmulpd	%xmm5,%xmm0,%xmm0
	vaddpd	%xmm5,%xmm0,%xmm0		/* z = z1 + z2 */
#endif

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
	vmulpd	RZ_OFF(40)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	vmulpd	RZ_OFF(24)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

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
        vmovapd  %xmm0, _DX0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR0(%rsp)

        vmovsd   _DX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fsd_exp_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR1(%rsp)

        vmovapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

	ELF_FUNC(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

	vcomiss		%xmm0, %xmm0
	jp     LBL(.LB_NZERO_SS_VEX)

        vcomiss .L__np_ln_lead_table(%rip), %xmm0        /* Equal to 0.0? */
        jne     LBL(.LB_NZERO_SS_VEX)
        vmovss .L4_386(%rip), %xmm0
        RZ_POP
        rep
        ret


LBL(.LB_NZERO_SS_VEX):


        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm2
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3

	/* Set n = nearest integer to r */
	vcomiss	.L__sp_ln_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_inf)
	vcomiss	.L_real_min_singleval(%rip), %xmm0
	jbe	LBL(.L_sp_ninf)

ASM_CONCAT(.L__fss_exp_dbl_entry_,TARGET_VEX_OR_FMA):
	vcvtpd2dq %xmm3,%xmm4	/* convert to integer */
	vcvtdq2pd %xmm4,%xmm1	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#	VFNMADDSD       %xmm2,.L__real_log2_by_32(%rip),%xmm1,%xmm2
	VFNMA_231SD	(.L__real_log2_by_32(%rip),%xmm1,%xmm2)
#else
	vmulsd	.L__real_log2_by_32(%rip),%xmm1,%xmm1
	vsubsd	%xmm1,%xmm2,%xmm2	/* r1 in xmm2, */
#endif
	vmovd	%xmm4,%ecx
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
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm1
	vmovapd	%xmm2,%xmm0

#ifdef TARGET_FMA
#	VFMADDSD        .L__real_3fe0000000000000(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm1)
        vmulsd          %xmm2,%xmm2,%xmm2
#        VFMADDSD        %xmm0,%xmm1,%xmm2,%xmm2
	VFMA_213SD	(%xmm0,%xmm1,%xmm2)
#else
	vmulsd	%xmm2,%xmm1,%xmm1
	vmulsd	%xmm2,%xmm2,%xmm2
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm1,%xmm1
	vmulsd	%xmm1,%xmm2,%xmm2
	vaddsd	%xmm0,%xmm2,%xmm2
#endif

	vmovsd	(%rdx,%rax,8),%xmm4

	/* *z2 = f2 + ((f1 + f2) * q); */
        add	$1023, %ecx	/* add bias */
	/* deal with infinite results */
	/* deal with denormal results */
        shlq	$52,%rcx        /* build 2^n */

#ifdef TARGET_FMA
#	VFMADDSD        %xmm4,%xmm2,%xmm4,%xmm2
	VFMA_213SD	(%xmm4,%xmm4,%xmm2)
#else
	vmulsd	%xmm4,%xmm2,%xmm2
	vaddsd	%xmm4,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(24)(%rsp),%xmm2,%xmm2	/* result *= 2^n */
	vunpcklpd %xmm2, %xmm2, %xmm2
	vcvtpd2ps %xmm2, %xmm0

LBL(.L_sp_final_check):
	RZ_POP
	rep
	ret

LBL(.L_sp_inf):
	vmovlps	.L_sp_real_infinity(%rip),%xmm0,%xmm0
	jmp	LBL(.L_sp_final_check)

LBL(.L_sp_ninf):
	jp	LBL(.L_sp_cvt_nan)
	xor	%eax, %eax
	vmovd	%eax,%xmm0
	jmp	LBL(.L_sp_final_check)

LBL(.L_sp_sinh_ninf):
        jp      LBL(.L_sp_cvt_nan)
        vmovlps  .L_sp_real_ninfinity(%rip),%xmm0,%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_cosh_ninf):
        jp      LBL(.L_sp_cvt_nan)
        vmovlps  .L_sp_real_infinity(%rip),%xmm0,%xmm0
        jmp     LBL(.L_sp_final_check)

LBL(.L_sp_cvt_nan):
	xor	%eax, %eax
	vmovd	%eax,%xmm1
	vmovsd	.L_real_cvt_nan(%rip),%xmm1
	vorps	%xmm1, %xmm0, %xmm0
	jmp	LBL(.L_sp_final_check)

	ELF_FUNC(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(104)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
#endif

	/* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovhlps  %xmm0, %xmm1, %xmm1
	vmovaps	 %xmm0, %xmm5
	vcvtps2pd %xmm0, %xmm2			/* xmm2 = dble(a(2)), dble(a(1)) */
	vcvtps2pd %xmm1, %xmm1			/* xmm1 = dble(a(4)), dble(a(3)) */
	vandps	 .L__ps_mask_unsign(%rip), %xmm5, %xmm5
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm4
	vcmpps	$6, .L__sp_ln_max_singleval(%rip), %xmm5, %xmm5
	vmulpd	%xmm2, %xmm3, %xmm3
	vmulpd	%xmm1, %xmm4, %xmm4
	vmovmskps %xmm5, %r8d

	/* Set n = nearest integer to r */
	vcvtpd2dq %xmm3,%xmm5	/* convert to integer */
	vcvtpd2dq %xmm4,%xmm6	/* convert to integer */
	test	 $15, %r8d
	vcvtdq2pd %xmm5,%xmm3	/* and back to float. */
	vcvtdq2pd %xmm6,%xmm4	/* and back to float. */
	jnz	LBL(.L__Scalar_fvsexp)

ASM_CONCAT(.L__fvs_exp_dbl_entry_,TARGET_VEX_OR_FMA):
	/* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm2,.L__real_log2_by_32(%rip),%xmm3,%xmm2
	VFNMA_231PD	(.L__real_log2_by_32(%rip),%xmm3,%xmm2)
#	VFNMADDPD	%xmm1,.L__real_log2_by_32(%rip),%xmm4,%xmm1
	VFNMA_231PD	(.L__real_log2_by_32(%rip),%xmm4,%xmm1)
#else
	vmulpd	.L__real_log2_by_32(%rip),%xmm3,%xmm3
	vsubpd	%xmm3,%xmm2,%xmm2	/* r1 in xmm2, */
	vmulpd	.L__real_log2_by_32(%rip),%xmm4,%xmm4
	vsubpd	%xmm4,%xmm1,%xmm1	/* r1 in xmm1, */
#endif
	vmovq	%xmm5,RZ_OFF(16)(%rsp)
	vmovq	%xmm6,RZ_OFF(24)(%rsp)
/*	vsubpd	%xmm3,%xmm2,%xmm2 */	/* r1 in xmm2, */
/*	vsubpd	%xmm4,%xmm1,%xmm1 */	/* r1 in xmm1, */
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
	vmovapd	%xmm2,%xmm0
	vmovapd	%xmm1,%xmm3
	vmovapd	%xmm2,%xmm4
	vmovapd	%xmm1,%xmm5

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
	vmulpd	.L__real_3FC5555555548F7C(%rip),%xmm0,%xmm0
	vmulpd	.L__real_3FC5555555548F7C(%rip),%xmm1,%xmm1

	sub 	%esi,%r10d
	sar 	$5,%r10d
	sub 	%edi,%r11d
	sar 	$5,%r11d

	vmulpd	%xmm2,%xmm2,%xmm2
	vmulpd	%xmm3,%xmm3,%xmm3
	vaddpd	.L__real_3fe0000000000000(%rip),%xmm0,%xmm0
	vaddpd	.L__real_3fe0000000000000(%rip),%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,%xmm0,%xmm2,%xmm2
	VFMA_213PD	(%xmm4,%xmm0,%xmm2)
#	VFMADDPD	%xmm5,%xmm1,%xmm3,%xmm3
	VFMA_213PD	(%xmm5,%xmm1,%xmm3)
#else
	vmulpd	%xmm0,%xmm2,%xmm2
	vaddpd	%xmm4,%xmm2,%xmm2
	vmulpd	%xmm1,%xmm3,%xmm3
	vaddpd	%xmm5,%xmm3,%xmm3
#endif
	vmovsd	(%rax,%rdx,8),%xmm0
	vmovhpd	(%rax,%rcx,8),%xmm0,%xmm0

	vmovsd	(%rax,%rdi,8),%xmm1
	vmovhpd	(%rax,%rsi,8),%xmm1,%xmm1

/*	vaddpd	%xmm4,%xmm2,%xmm2 */
/*	vaddpd	%xmm5,%xmm3,%xmm3 */

	/* *z2 = f2 + ((f1 + f2) * q); */
        add 	$1023, %r8d	/* add bias */
        add 	$1023, %r9d	/* add bias */
        add 	$1023, %r10d	/* add bias */
        add 	$1023, %r11d	/* add bias */

	/* deal with infinite and denormal results */
/*	vmulpd	%xmm0,%xmm2,%xmm2 */
/*	vmulpd	%xmm1,%xmm3,%xmm3 */
        shlq	$52,%r8
        shlq	$52,%r9
        shlq	$52,%r10
        shlq	$52,%r11
#ifdef TARGET_FMA
#	VFMADDPD	%xmm0,%xmm0,%xmm2,%xmm2
	VFMA_213PD	(%xmm0,%xmm0,%xmm2)
#	VFMADDPD	%xmm1,%xmm1,%xmm3,%xmm3
	VFMA_213PD	(%xmm1,%xmm1,%xmm3)
#else
	vmulpd	%xmm0,%xmm2,%xmm2
	vaddpd	%xmm0,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
	vmulpd	%xmm1,%xmm3,%xmm3
	vaddpd	%xmm1,%xmm3,%xmm3  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%r8,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	vmulpd	RZ_OFF(24)(%rsp),%xmm2,%xmm2	/* result *= 2^n */

	movq	%r11,RZ_OFF(40)(%rsp) 	/* get 2^n to memory */
	movq	%r10,RZ_OFF(32)(%rsp) 	/* get 2^n to memory */
	vmulpd	RZ_OFF(40)(%rsp),%xmm3,%xmm3	/* result *= 2^n */

	vcvtpd2ps %xmm2,%xmm0
	vcvtpd2ps %xmm3,%xmm1
	vshufps	$68,%xmm1,%xmm0,%xmm0

LBL(.L_vsp_final_check):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(104)(%rsp), %ymm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif

	RZ_POP
	rep
	ret

LBL(.L__Scalar_fvsexp):
#if defined(_WIN64)
	/* Need to restore callee-saved regs can do here for this path
	 * because entry was only thru fvs_exp_fma4/fvs_exp_vex
	 */
	vmovdqu	RZ_OFF(104)(%rsp), %ymm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif
        pushq   %rbp			/* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovaps  %xmm0, _SX0(%rsp)

        CALL(ENT(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovss   _SX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovss   _SX2(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovss   _SX3(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_exp_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

        ELF_FUNC(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA))


/* ============================================================ */
/* This routine takes a 4 doubles input, and produces 4
   single precision outputs

** MAKE SURE THE STACK HERE MATCHES THE STACK IN __fvsexp

*/
	.text
        ALN_FUNC

#ifdef TARGET_FMA
ENT(ASM_CONCAT(__fvs_exp_dbl_,TARGET_VEX_OR_FMA)):
#else
ENT(__fvs_exp_dbl_vex):
#endif
	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(104)(%rsp)
	movq	%rsi, RZ_OFF(64)(%rsp)
	movq	%rdi, RZ_OFF(72)(%rsp)
#endif

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm4
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm1, %xmm6
	vmulpd	%xmm0, %xmm3, %xmm3
	vmulpd	%xmm1, %xmm4, %xmm4
	vandpd	.L__real_mask_unsign(%rip), %xmm2, %xmm2
	vandpd	.L__real_mask_unsign(%rip), %xmm6, %xmm6

	/* Compare input with max */
	vcmppd	 $6, .L__dp_max_singleval(%rip), %xmm2, %xmm2
	vcmppd	 $6, .L__dp_max_singleval(%rip), %xmm6, %xmm6

	/* Set n = nearest integer to r */
	vcvtpd2dq %xmm3,%xmm5	/* convert to integer */
	vorpd	%xmm6, %xmm2, %xmm2	/* Or masks together */
	vcvtpd2dq %xmm4,%xmm6	/* convert to integer */
	vmovmskps %xmm2, %r8d
	vcvtdq2pd %xmm5,%xmm3	/* and back to float. */
	vcvtdq2pd %xmm6,%xmm4	/* and back to float. */
	vmovapd	%xmm0, %xmm2	/* Move input */
	test	 $3, %r8d
	jz	ASM_CONCAT(.L__fvs_exp_dbl_entry_,TARGET_VEX_OR_FMA)

LBL(.L__Scalar_fvs_exp_dbl):
        pushq   %rbp			/* This works because -8(rsp) not used! */
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovapd  %xmm0, _DX0(%rsp)
        vmovapd  %xmm1, _DX2(%rsp)

        CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovsd   _DX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovsd   _DX2(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovsd   _DX3(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)))


        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp

	/* Done */
#if defined(_WIN64)
	vmovdqu	RZ_OFF(104)(%rsp), %ymm6
	movq	RZ_OFF(64)(%rsp), %rsi
	movq	RZ_OFF(72)(%rsp), %rdi
#endif

	RZ_POP
	ret

        ELF_FUNC(ASM_CONCAT(__fvs_exp_dbl_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_exp_dbl_,TARGET_VEX_OR_FMA))


/* ============================================================ */
/* This routine takes a double input, and produces a
   single precision output

** MAKE SURE THE STACK HERE MATCHES THE STACK IN __fss_exp

*/
	.text
        ALN_FUNC
#ifdef TARGET_FMA
ENT(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA)):
#else
ENT(__fss_exp_dbl_vex):
#endif

	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmovapd	%xmm0, %xmm2
	vmulsd	%xmm0, %xmm3, %xmm3

	/* Set n = nearest integer to r */
	vcomisd	.L__dp_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_dp_inf)
	vcomisd	.L__dp_min_singleval(%rip), %xmm0
	jnbe	ASM_CONCAT(.L__fss_exp_dbl_entry_,TARGET_VEX_OR_FMA)

LBL(.L_sp_dp_ninf):
	xor	%eax, %eax
	vmovd	%eax,%xmm0
	jmp	LBL(.L_sp_dp_final_check)

LBL(.L_sp_dp_inf):
	vmovlps	.L_sp_real_infinity(%rip),%xmm0,%xmm0
LBL(.L_sp_dp_final_check):
	RZ_POP
	rep
	ret

	ELF_FUNC(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_exp_dbl_,TARGET_VEX_OR_FMA))


/* ============================================================ */
/* This routine takes two doubles input, and produces a
   double precision output

** MAKE SURE THE STACK HERE MATCHES THE STACK IN __fsd_exp

*/
	.text
        ALN_FUNC

#ifdef TARGET_FMA
ENT(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA)):
#else
ENT(__fsd_exp_long_vex):
#endif
	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm0,%xmm3,%xmm3

	/* Set n = nearest integer to r */
	vcomisd	.L__real_ln_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	vcomisd	.L__real_ln_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_ninf)
	vcvtpd2dq %xmm3,%xmm3	/* convert to integer */
	vcvtdq2pd %xmm3,%xmm5	/* and back to float. */

	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovsd	.L__real_log2_by_32_lead(%rip),%xmm2
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm2,%xmm5,%xmm0
	VFNMA_231SD	(%xmm2,%xmm5,%xmm0)
#else
	vmulsd	%xmm5,%xmm2,%xmm2
	vsubsd	%xmm2,%xmm0,%xmm0	/* r1 in xmm0, */
#endif
	vmovd	%xmm3,%ecx
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,.L__real_log2_by_32_tail(%rip),%xmm5,%xmm5
	VFMA_132SD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm5)
#else
	vmulsd	.L__real_log2_by_32_tail(%rip),%xmm5,%xmm5
	vaddsd	%xmm1, %xmm5,%xmm5
#endif

	/* j = n & 0x0000001f; */
	movq	$0x1f,%rax
	and	%ecx,%eax
	vmovapd	%xmm0,%xmm2	/* r1 */

	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%eax,%ecx
	sar	$5,%ecx
	vaddsd	%xmm5,%xmm2,%xmm2    /* r = r1 + r2 */
	vshufpd	$0, %xmm0, %xmm5, %xmm5	/* Store r1 and r2 */

	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +
	   r*r*( 5.00000000000000008883e-01 +
	   r*( 1.66666666665260878863e-01 +
	   r*( 4.16666666662260795726e-02 +
	   r*( 8.33336798434219616221e-03 +
	   r*( 1.38889490863777199667e-03 ))))));
	   q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */

	/* r in %xmm2, r1, r2 in %xmm5 */

	vmovapd	%xmm2,%xmm1
	vmovsd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm0
#ifdef TARGET_FMA
#	VFMADDSD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFMADDSD	.L__real_3fe0000000000000(%rip),%xmm2,%xmm0,%xmm0
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
#else
	vmulsd	%xmm2,%xmm3,%xmm3
	vaddsd	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3
	vmulsd	%xmm2,%xmm0,%xmm0
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm0,%xmm0
#endif
	vmulsd	%xmm2,%xmm1,%xmm1
	vmovapd	%xmm1,%xmm4
	vmulsd	%xmm1,%xmm4,%xmm4
#ifdef TARGET_FMA
#	VFMADDSD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMADDSD	%xmm5,%xmm1,%xmm0,%xmm0
	VFMA_213SD	(%xmm5,%xmm1,%xmm0)
#	VFMADDSD	%xmm0,%xmm4,%xmm3,%xmm0
	VFMA_231SD	(%xmm4,%xmm3,%xmm0)
#else
	vmulsd	%xmm2,%xmm3,%xmm3
	vaddsd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3
	vmulsd	%xmm1,%xmm0,%xmm0
	vaddsd	%xmm5,%xmm0,%xmm0
	vmulsd	%xmm4,%xmm3,%xmm3
	vaddsd	%xmm3,%xmm0,%xmm0
#endif
	vshufpd	$1, %xmm5, %xmm5, %xmm5	/* Store r1 and r2 */
	vaddsd	%xmm5,%xmm0,%xmm0

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%rdx,%rax,8),%xmm5
	/* deal with infinite results */
        movslq	%ecx,%rcx
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213SD	(%xmm5,%xmm5,%xmm0)
#else
	vmulsd	%xmm5,%xmm0,%xmm0
	vaddsd	%xmm5,%xmm0,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

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
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

	RZ_POP
	ret

	ELF_FUNC(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA))


/* ============================================================
 *
 *  Prototype:
 *
 *      __fvd_exp_long(__m128d x);
 *
 *    Computes e raised to the x power.
 *  Does not perform error checking.   Denormal results are truncated to 0.
 *
 */
        .text
        ALN_FUNC
#ifdef TARGET_FMA
ENT(ASM_CONCAT(__fvd_exp_long_,TARGET_VEX_OR_FMA)):
#else
ENT(__fvd_exp_long_vex):
#endif

	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	%xmm0, %xmm2
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulpd	%xmm0,%xmm3,%xmm3

	/* save x for later. */
	vandpd	.L__real_mask_unsign(%rip), %xmm2,%xmm2

        /* Set n = nearest integer to r */
	vcvtpd2dq %xmm3,%xmm3
	vcmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2,%xmm2
	leaq	.L__two_to_jby32_table(%rip),%r11
	vcvtdq2pd %xmm3,%xmm5
	vmovmskpd %xmm2, %r8d

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovapd	.L__real_log2_by_32_lead(%rip),%xmm2
/*	vmulpd	%xmm5,%xmm2,%xmm2 */
	vmovq	 %xmm3,RZ_OFF(24)(%rsp)
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvd_exp_long)

	/* r2 =   - n * logbaseof2_by_32_trail; */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm0,%xmm5,%xmm2,%xmm0
	VFNMA_231PD	(%xmm5,%xmm2,%xmm0)
#	VFMADDPD	%xmm1,.L__real_log2_by_32_tail(%rip),%xmm5,%xmm5
	VFMA_132PD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm5)
#else
	vmulpd	%xmm5,%xmm2,%xmm2
	vsubpd	%xmm2,%xmm0,%xmm0	/* r1 in xmm0, */
	vmulpd	.L__real_log2_by_32_tail(%rip),%xmm5,%xmm5 	/* r2 in xmm5 */
	vaddpd	%xmm1, %xmm5,%xmm5
#endif

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	mov	RZ_OFF(24)(%rsp),%ecx
	and	%ecx,%r9d

	mov	RZ_OFF(20)(%rsp),%edx
	and	%edx,%r8d
	vmovapd	%xmm0,%xmm2	/* xmm2 = r1 for now */
	                   	/* xmm0 = r1 for good */

	/* f1 = two_to_jby32_lead_table[j]; */
	/* f2 = two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	sub	%r9d,%ecx
	sar	$5,%ecx
	sub	%r8d,%edx
	sar	$5,%edx
	vaddpd	%xmm5,%xmm2,%xmm2    /* xmm2 = r = r1 + r2 */

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(72)(%rsp)
#endif
	/* Step 2. Compute the polynomial. */
	/* q = r1 + (r2 +
	 * r*r*( 5.00000000000000008883e-01 +
	 * r*( 1.66666666665260878863e-01 +
	 * r*( 4.16666666662260795726e-02 +
	 * r*( 8.33336798434219616221e-03 +
	 * r*( 1.38889490863777199667e-03 ))))));
	 * q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
	vmovapd	%xmm2,%xmm1
	vmovapd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovapd	.L__real_3FC5555555548F7C(%rip),%xmm6

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

#ifdef TARGET_FMA
#	VFMADDPD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFMADDPD	.L__real_3fe0000000000000(%rip),%xmm2,%xmm6,%xmm6
	VFMA_213PD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm6)
#else
	vmulpd	%xmm2,%xmm3,%xmm3	/* *x */
	vaddpd	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3
	vmulpd	%xmm2,%xmm6,%xmm6	/* *x */
	vaddpd	.L__real_3fe0000000000000(%rip),%xmm6,%xmm6
#endif
	vmulpd	%xmm2,%xmm1,%xmm1	/* x*x */
	vmovapd	%xmm1,%xmm4

/*	vaddpd	 .L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3 */
/*	vaddpd	 .L__real_3fe0000000000000(%rip),%xmm6,%xmm6 */
	vmulpd	%xmm1,%xmm4,%xmm4	/* x^4 */

#ifdef TARGET_FMA
#	VFMADDPD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMADDPD	%xmm5,%xmm1,%xmm6,%xmm6
	VFMA_213PD	(%xmm5,%xmm1,%xmm6)
#	VFMADDPD	%xmm6,%xmm4,%xmm3,%xmm6
	VFMA_231PD	(%xmm4,%xmm3,%xmm6)
#else
	vmulpd	%xmm2,%xmm3,%xmm3	/* *x */
	vaddpd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3
	vmulpd	%xmm1,%xmm6,%xmm6	/* *x^2 */
	vaddpd	%xmm5,%xmm6,%xmm6	/* + x  */
	vmulpd	%xmm4,%xmm3,%xmm3	/* *x^4 */
	vaddpd	%xmm3,%xmm6,%xmm6
#endif

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1023,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

/*	vaddpd	%xmm3,%xmm6,%xmm6 */	/* q = final sum */
	vaddpd	%xmm6,%xmm0,%xmm0	/* final sum */

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	vmovhpd	(%r11,%r8,8),%xmm5,%xmm5 	/* f1 + f2 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm5,%xmm0)
#else
	vmulpd	%xmm5,%xmm0,%xmm0
	vaddpd	%xmm5,%xmm0,%xmm0		/* z = z1 + z2 */
#endif

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
	vmulpd	RZ_OFF(40)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	vmulpd	RZ_OFF(24)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

#if defined(_WIN64)
	vmovdqu	RZ_OFF(72)(%rsp), %ymm6
#endif

LBL(.L__final_check_long):
	RZ_POP
	rep
	ret

#define _DT0 48
#define _DT1 56

LBL(.L__Scalar_fvd_exp_long):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovapd  %xmm0, _DX0(%rsp)
        vmovapd  %xmm1, _DT0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR0(%rsp)

        vmovsd   _DX1(%rsp), %xmm0
        vmovsd   _DT1(%rsp), %xmm1
        CALL(ENT(ASM_CONCAT(__fsd_exp_long_,TARGET_VEX_OR_FMA)))


	vmovsd   %xmm0, _DR1(%rsp)

        vmovapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check_long)

	ELF_FUNC(ASM_CONCAT(__fvd_exp_long_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fvd_exp_long_,TARGET_VEX_OR_FMA))


/*============================================================
#Copyright (c) 2004 Advanced Micro Devices, Inc.
#
#All rights reserved.
#
#Redistribution and  use in source and binary  forms, with or
#without  modification,  are   permitted  provided  that  the
#following conditions are met:
#
#+ Redistributions  of source  code  must  retain  the  above
#  copyright  notice,   this  list  of   conditions  and  the
#  following disclaimer.
#
#+ Redistributions  in binary  form must reproduce  the above
#  copyright  notice,   this  list  of   conditions  and  the
#  following  disclaimer in  the  documentation and/or  other
#  materials provided with the distribution.
#
#+ Neither the  name of Advanced Micro Devices,  Inc. nor the
#  names  of  its contributors  may  be  used  to endorse  or
#  promote  products  derived   from  this  software  without
#  specific prior written permission.
#
#THIS  SOFTWARE  IS PROVIDED  BY  THE  COPYRIGHT HOLDERS  AND
#CONTRIBUTORS "AS IS" AND  ANY EXPRESS OR IMPLIED WARRANTIES,
#INCLUDING,  BUT NOT  LIMITED TO,  THE IMPLIED  WARRANTIES OF
#MERCHANTABILITY  AND FITNESS  FOR A  PARTICULAR  PURPOSE ARE
#DISCLAIMED.  IN  NO  EVENT  SHALL  ADVANCED  MICRO  DEVICES,
#INC.  OR CONTRIBUTORS  BE LIABLE  FOR ANY  DIRECT, INDIRECT,
#INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL  DAMAGES
#(INCLUDING,  BUT NOT LIMITED  TO, PROCUREMENT  OF SUBSTITUTE
#GOODS  OR  SERVICES;  LOSS  OF  USE, DATA,  OR  PROFITS;  OR
#BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON  ANY THEORY OF
#LIABILITY,  WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
#(INCLUDING NEGLIGENCE  OR OTHERWISE) ARISING IN  ANY WAY OUT
#OF  THE  USE  OF  THIS  SOFTWARE, EVEN  IF  ADVISED  OF  THE
#POSSIBILITY OF SUCH DAMAGE.
#
#It is  licensee's responsibility  to comply with  any export
#regulations applicable in licensee's jurisdiction.
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
        .globl  ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)):


	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(96)(%rsp)
#endif
	/* Get input x into the range [0.5,1) */
	/* compute the index into the log tables */

	vcomisd	.L__real_mindp(%rip), %xmm0
	vmovdqa	%xmm0,%xmm3
	vmovapd	%xmm0,%xmm1
	jb	LBL(.L__z_or_n)

	vpsrlq	$52,%xmm3,%xmm3
	vsubsd	.L__real_one(%rip),%xmm1,%xmm1
	vpsubq	.L__mask_1023(%rip),%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm6	/* xexp */

LBL(.L__100):
	vmovdqa	%xmm0,%xmm3
	vpand	.L__real_mant(%rip),%xmm3,%xmm3
	xorq	%r8,%r8
	vmovdqa	%xmm3,%xmm4
	vmovsd	.L__real_half(%rip),%xmm5	/* .5 */
	/* Now  x = 2**xexp  * f,  1/2 <= f < 1. */
	vpsrlq	$45,%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm2
	vpsrlq	$1,%xmm3,%xmm3
	vpaddq	.L__mask_040(%rip),%xmm3,%xmm3
	vpand	.L__mask_001(%rip),%xmm2,%xmm2
	vpaddq	%xmm2,%xmm3,%xmm3

	vandpd	.L__real_notsign(%rip),%xmm1,%xmm1
	vcomisd	.L__real_threshold(%rip),%xmm1
	vcvtdq2pd %xmm3,%xmm1
	jb	LBL(.L__near_one)
	vmovd	%xmm3,%r8d

	/* reduce and get u */
	vpor	.L__real_half(%rip),%xmm4,%xmm4
	vmovdqa	%xmm4,%xmm2

	leaq	.L__np_ln_lead_table(%rip),%r9

	vmulsd	.L__real_3f80000000000000(%rip),%xmm1,%xmm1	/* f1 = index/128 */
	vsubsd	%xmm1,%xmm2,%xmm2				/* f2 = f - f1 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,%xmm5,%xmm2,%xmm1
	VFMA_231SD	(%xmm5,%xmm2,%xmm1)
#else
	vmulsd	%xmm2,%xmm5,%xmm5
	vaddsd	%xmm5,%xmm1,%xmm1
#endif

	vdivsd	%xmm1,%xmm2,%xmm2				/* u */

	/* Check for +inf */
	vcomisd	.L__real_inf(%rip),%xmm0
	je	LBL(.L__finish)

	vmovsd	-512(%r9,%r8,8),%xmm0 			/* z1 */
	/* solve for ln(1+u) */
	vmovapd	%xmm2,%xmm1				/* u */
	vmulsd	%xmm2,%xmm2,%xmm2				/* u^2 */
	vmovapd	%xmm2,%xmm5
	vmovapd	.L__real_cb3(%rip),%xmm3
	vmulsd	%xmm1,%xmm5,%xmm5				/* u^3 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__real_cb2(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_cb2(%rip),%xmm2,%xmm3)
#else
	vmulsd	%xmm2,%xmm3,%xmm3				/* Cu2 */
	vaddsd	.L__real_cb2(%rip),%xmm3,%xmm3 		/* B+Cu2 */
#endif
	vmovapd	%xmm2,%xmm4
	vmulsd	%xmm5,%xmm4,%xmm4				/* u^5 */
	vmovsd	.L__real_log2_lead(%rip),%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,.L__real_cb1(%rip),%xmm5,%xmm1
	VFMA_231SD	(.L__real_cb1(%rip),%xmm5,%xmm1)
#	VFMADDSD	%xmm1,%xmm3,%xmm4,%xmm1
	VFMA_231SD	(%xmm3,%xmm4,%xmm1)
#else
	vmulsd	.L__real_cb1(%rip),%xmm5,%xmm5 		/* Au3 */
	vaddsd	%xmm5,%xmm1,%xmm1				/* u+Au3 */
	vmulsd	%xmm3,%xmm4,%xmm4				/* u5(B+Cu2) */
	vaddsd	%xmm4,%xmm1,%xmm1				/* poly */
#endif

	/* recombine */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	vaddsd	-512(%rdx,%r8,8),%xmm1,%xmm1 			/* z2	+=q */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm6,%xmm0
	VFMA_231SD	(%xmm2,%xmm6,%xmm0)
#	VFMADDSD	%xmm1,.L__real_log2_tail(%rip),%xmm6,%xmm1
	VFMA_231SD	(.L__real_log2_tail(%rip),%xmm6,%xmm1)
#else
	vmulsd	%xmm6,%xmm2,%xmm2				/* npi2 * log2_lead */
	vaddsd	%xmm2,%xmm0,%xmm0				/* r1 */
	vmulsd	.L__real_log2_tail(%rip),%xmm6,%xmm6
	vaddsd	%xmm6,%xmm1,%xmm1				/* r2 */
#endif
	vaddsd	%xmm1,%xmm0,%xmm0

LBL(.L__finish):
#if defined(_WIN64)
	vmovdqu	RZ_OFF(96)(%rsp), %ymm6
#endif

	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.L__near_one):
	/* saves 10 cycles */
	/* r = x - 1.0; */
	vmovsd	.L__real_two(%rip),%xmm2
	vsubsd	.L__real_one(%rip),%xmm0,%xmm0

	/* u = r / (2.0 + r); */
	vaddsd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivsd	%xmm2,%xmm1,%xmm1
	vmovsd	.L__real_ca4(%rip),%xmm4
	vmovsd	.L__real_ca3(%rip),%xmm5
	/* correction = r * u; */
	vmovapd	%xmm0,%xmm6
	vmulsd	%xmm1,%xmm6,%xmm6

	/* u = u + u; */
	vaddsd	%xmm1,%xmm1,%xmm1
	vmovapd	%xmm1,%xmm2
	vmulsd	%xmm2,%xmm2,%xmm2
	/* r2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction); */
	vmulsd	%xmm1,%xmm5,%xmm5
	vmovapd	%xmm1,%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3
	vmulsd	.L__real_ca2(%rip),%xmm2,%xmm2

	vaddsd	.L__real_ca1(%rip),%xmm2,%xmm2
	vmovapd	%xmm3,%xmm1
	vmulsd	%xmm1,%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm3,%xmm4,%xmm5
	VFMA_231SD	(%xmm3,%xmm4,%xmm5)
	vmulsd		%xmm5,%xmm1,%xmm1
#	VFMADDSD	%xmm1,%xmm2,%xmm3,%xmm2
	VFMA_213SD	(%xmm1,%xmm3,%xmm2)
#else
	vmulsd	%xmm3,%xmm4,%xmm4
	vaddsd	%xmm4,%xmm5,%xmm5

	vmulsd	%xmm5,%xmm1,%xmm1
	vmulsd	%xmm3,%xmm2,%xmm2
	vaddsd	%xmm1,%xmm2,%xmm2
#endif
	vsubsd	%xmm6,%xmm2,%xmm2

	/* return r + r2; */
	vaddsd	%xmm2,%xmm0,%xmm0
	jmp	LBL(.L__finish)

	/* Start here for all the conditional cases */
	/* we have a zero, a negative number, denorm, or nan. */
LBL(.L__z_or_n):
	jp	LBL(.L__lnan)
	vxorpd	%xmm1, %xmm1, %xmm1
	vcomisd	%xmm1, %xmm0
	je	LBL(.L__zero)
	jbe	LBL(.L__negative_x)

	/* A Denormal input, scale appropriately */
	vmulsd	.L__real_scale(%rip), %xmm0, %xmm0
	vmovdqa	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm1

	vpsrlq	$52,%xmm3,%xmm3
	vsubsd	.L__real_one(%rip),%xmm1,%xmm1
	vpsubq	.L__mask_1075(%rip),%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm6
	jmp	LBL(.L__100)

	/* x == +/-0.0 */
LBL(.L__zero):
#ifdef FMATH_EXCEPTIONS
        vmovsd	.L__real_one(%rip), %xmm1
	vdivsd	%xmm0, %xmm1, %xmm0 /* Generate divide-by-zero op */
#endif
	vmovsd	.L__real_ninf(%rip),%xmm0  /* C99 specs -inf for +-0 */
	jmp	LBL(.L__finish)

	/* x < 0.0 */
LBL(.L__negative_x):
#ifdef FMATH_EXCEPTIONS
	vsqrtsd	%xmm0, %xmm0, %xmm0
#endif
	vmovsd	.L__real_nan(%rip),%xmm0
	jmp	LBL(.L__finish)

	/* NaN */
LBL(.L__lnan):
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	.L__real_qnanbit(%rip), %xmm1	/* convert to quiet */
	vorpd	%xmm1, %xmm0, %xmm0
	jmp	LBL(.L__finish)


        ELF_FUNC(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA))


/* ======================================================================== */
    	.text
    	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(72)(%rsp)
#endif

	vmovdqa	%xmm0, RZ_OFF(40)(%rsp)	/* save the input values */
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm4
	vpxor	%xmm1, %xmm1, %xmm1
	vcmppd	$6, .L__real_maxfp(%rip), %xmm2, %xmm2
	vcmppd 	$1, .L__real_mindp(%rip), %xmm4, %xmm4
	vmovdqa	%xmm0, %xmm3
	vpsrlq	$52, %xmm3, %xmm3
	vorpd	%xmm2, %xmm4, %xmm4
	vpsubq	.L__mask_1023(%rip),%xmm3,%xmm3
	vmovmskpd %xmm4, %r8d
	vpackssdw %xmm1, %xmm3, %xmm3
	vcvtdq2pd %xmm3, %xmm6		/* xexp */
	vmovdqa	%xmm0, %xmm2
	xorq	%rax, %rax
	vsubpd	.L__real_one(%rip), %xmm2, %xmm2
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdlog)

	vmovdqa	%xmm0,%xmm3
	vandpd	.L__real_notsign(%rip),%xmm2, %xmm2
	vpand	.L__real_mant(%rip),%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm4
	vmovapd	.L__real_half(%rip),%xmm5	/* .5 */

	vcmppd	$1,.L__real_threshold(%rip),%xmm2,%xmm2
	vmovmskpd %xmm2,%r10d
	cmp	$3,%r10d
	jz	LBL(.Lall_nearone)

	vpsrlq	$45,%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm2
	vpsrlq	$1,%xmm3,%xmm3
	vpaddq	.L__mask_040(%rip),%xmm3,%xmm3
	vpand	.L__mask_001(%rip),%xmm2,%xmm2
	vpaddq	%xmm2,%xmm3,%xmm3

	vpackssdw %xmm1,%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm1
	xorq	 %rcx,%rcx
	vmovq	 %xmm3,RZ_OFF(24)(%rsp)

	vpor	.L__real_half(%rip),%xmm4,%xmm4
	vmovdqa	%xmm4,%xmm2
	vmulpd	.L__real_3f80000000000000(%rip),%xmm1,%xmm1	/* f1 = index/128 */

	leaq	.L__np_ln_lead_table(%rip),%rdx
	mov	RZ_OFF(24)(%rsp),%eax

	vsubpd	%xmm1,%xmm2,%xmm2				/* f2 = f - f1 */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,%xmm2,%xmm5,%xmm1
	VFMA_231PD	(%xmm2,%xmm5,%xmm1)
#else
	vmulpd	%xmm2,%xmm5,%xmm5
	vaddpd	%xmm5,%xmm1,%xmm1
#endif
	vdivpd	%xmm1,%xmm2,%xmm2				/* u */

	vmovsd	 -512(%rdx,%rax,8),%xmm0		/* z1 */
	mov	RZ_OFF(20)(%rsp),%ecx
	vmovhpd	 -512(%rdx,%rcx,8),%xmm0,%xmm0		/* z1 */
	vmovapd	%xmm2,%xmm1				/* u */
	vmulpd	%xmm2,%xmm2,%xmm2				/* u^2 */
	vmovapd	%xmm2,%xmm5
	vmovapd	.L__real_cb3(%rip),%xmm3
	vmulpd	%xmm1,%xmm5,%xmm5				/* u^3 */
#ifdef TARGET_FMA
#	VFMADDPD	.L__real_cb2(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_cb2(%rip),%xmm2,%xmm3)
#else
	vmulpd	%xmm2,%xmm3,%xmm3				/* Cu2 */
	vaddpd	.L__real_cb2(%rip),%xmm3,%xmm3 		/* B+Cu2 */
#endif
	vmulpd	%xmm5,%xmm2,%xmm2				/* u^5 */
	vmovapd	.L__real_log2_lead(%rip),%xmm4

#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,.L__real_cb1(%rip),%xmm5,%xmm1
	VFMA_231PD	(.L__real_cb1(%rip),%xmm5,%xmm1)
#	VFMADDPD	%xmm1,%xmm3,%xmm2,%xmm1
	VFMA_231PD	(%xmm3,%xmm2,%xmm1)
#	VFMADDPD	%xmm0,%xmm6,%xmm4,%xmm0
	VFMA_231PD	(%xmm6,%xmm4,%xmm0)
#else
	vmulpd	.L__real_cb1(%rip),%xmm5,%xmm5 		/* Au3 */
	vaddpd	%xmm5,%xmm1,%xmm1				/* u+Au3 */
	vmulpd	%xmm3,%xmm2,%xmm2				/* u5(B+Cu2) */
	vaddpd	%xmm2,%xmm1,%xmm1				/* poly */
	vmulpd	%xmm6,%xmm4,%xmm4				/* xexp * log2_lead */
	vaddpd	%xmm4,%xmm0,%xmm0				/* r1 */
#endif
	leaq	.L__np_ln_tail_table(%rip),%rdx
	vmovsd	 -512(%rdx,%rax,8),%xmm4		/* z2+=q */
	vmovhpd	 -512(%rdx,%rcx,8),%xmm4,%xmm4		/* z2+=q */

	vaddpd	%xmm4,%xmm1,%xmm1

#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,.L__real_log2_tail(%rip),%xmm6,%xmm1
	VFMA_231PD	(.L__real_log2_tail(%rip),%xmm6,%xmm1)
#else
	vmulpd	.L__real_log2_tail(%rip),%xmm6,%xmm6
	vaddpd	%xmm6,%xmm1,%xmm1				/* r2 */
#endif

	vaddpd	%xmm1,%xmm0,%xmm0

LBL(.Lfinish):
	test		 $3,%r10d
	jnz		LBL(.Lnear_one)
LBL(.Lfinishn1):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(72)(%rsp), %ymm6
#endif
	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.Lall_nearone):
	vmovapd	.L__real_two(%rip),%xmm2
	vsubpd	.L__real_one(%rip),%xmm0,%xmm0	/* r */
	vaddpd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivpd	%xmm2,%xmm1,%xmm1			/* u */
	vmovapd	.L__real_ca4(%rip),%xmm4  	/* D */
	vmovapd	.L__real_ca3(%rip),%xmm5 	/* C */
	vmovapd	%xmm0,%xmm6
	vmulpd	%xmm1,%xmm6,%xmm6			/* correction */
	vaddpd	%xmm1,%xmm1,%xmm1			/* u */
	vmovapd	%xmm1,%xmm2
	vmulpd	%xmm2,%xmm2,%xmm2			/* v =u^2 */
	vmulpd	%xmm1,%xmm5,%xmm5 			/* Cu */
	vmovapd	%xmm1,%xmm3
	vmulpd	%xmm2,%xmm3,%xmm3			/* u^3 */
	vmulpd	.L__real_ca2(%rip),%xmm2,%xmm2	/* Bu^2 */
	vmulpd	%xmm3,%xmm4,%xmm4			/* Du^3 */

	vaddpd	.L__real_ca1(%rip),%xmm2,%xmm2	/* +A */
	vmovapd	%xmm3,%xmm1
	vmulpd	%xmm1,%xmm1,%xmm1			/* u^6 */
	vaddpd	%xmm4,%xmm5,%xmm5			/* Cu+Du3 */
	vmulpd	%xmm5,%xmm1,%xmm1			/* u6(Cu+Du3) */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,%xmm3,%xmm2,%xmm2
	VFMA_213PD	(%xmm1,%xmm3,%xmm2)
#else
	vmulpd	%xmm3,%xmm2,%xmm2			/* u3(A+Bu2) */
	vaddpd	%xmm1,%xmm2,%xmm2
#endif
	vsubpd	%xmm6,%xmm2,%xmm2			/*  -correction */

	vaddpd	%xmm2,%xmm0,%xmm0
	jmp	LBL(.Lfinishn1)

	ALN_QUAD
LBL(.Lnear_one):
	test	$1,%r10d
	jz	LBL(.Llnn12)

	vmovlpd	RZ_OFF(40)(%rsp),%xmm0,%xmm0          /* Don't mess with this one */
                                                /* Need the high half live */
	call	LBL(.Lln1)

LBL(.Llnn12):
	test	$2,%r10d			/* second number? */
	jz	LBL(.Llnn1e)
	vmovlpd	%xmm0,RZ_OFF(40)(%rsp)
	vmovsd	RZ_OFF(32)(%rsp),%xmm0
	call	LBL(.Lln1)
	vmovlpd	%xmm0,RZ_OFF(32)(%rsp)
	vmovapd	RZ_OFF(40)(%rsp),%xmm0

LBL(.Llnn1e):
	jmp	LBL(.Lfinishn1)

LBL(.Lln1):
	vmovsd	.L__real_two(%rip),%xmm2
	vsubsd	.L__real_one(%rip),%xmm0,%xmm0	/* r */
	vaddsd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivsd	%xmm2,%xmm1,%xmm1			/* u */
	vmovsd	.L__real_ca4(%rip),%xmm4	/* D */
	vmovsd	.L__real_ca3(%rip),%xmm5	/* C */
	vmovapd	%xmm0,%xmm6
	vmulsd	%xmm1,%xmm6,%xmm6			/* correction */
	vaddsd	%xmm1,%xmm1,%xmm1			/* u */
	vmovapd	%xmm1,%xmm2
	vmulsd	%xmm2,%xmm2,%xmm2			/* v =u^2 */
	vmulsd	%xmm1,%xmm5,%xmm5			/* Cu */
	vmovapd	%xmm1,%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3			/* u^3 */
	vmulsd	.L__real_ca2(%rip),%xmm2,%xmm2	/*Bu^2 */
/*	vmulsd	%xmm3,%xmm4,%xmm4 */			/*Du^3 */

	vaddsd	.L__real_ca1(%rip),%xmm2,%xmm2	/* +A */
	vmovapd	%xmm3,%xmm1
	vmulsd	%xmm1,%xmm1,%xmm1			/* u^6 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm3,%xmm4,%xmm5
	VFMA_231SD	(%xmm3,%xmm4,%xmm5)
#else
	vmulsd	%xmm3,%xmm4,%xmm4			/*Du^3 */
	vaddsd	%xmm4,%xmm5,%xmm5			/* Cu+Du3 */
#endif

	vmulsd	%xmm3,%xmm2,%xmm2			/* u3(A+Bu2) */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm5,%xmm1,%xmm2
	VFMA_231SD	(%xmm5,%xmm1,%xmm2)
#else
	vmulsd	%xmm5,%xmm1,%xmm1			/* u6(Cu+Du3) */
	vaddsd	%xmm1,%xmm2,%xmm2
#endif
	vsubsd	%xmm6,%xmm2,%xmm2			/* -correction */

	vaddsd	%xmm2,%xmm0,%xmm0
	ret

#define _X0 0
#define _X1 8

#define _R0 32
#define _R1 40

LBL(.L__Scalar_fvdlog):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovapd  %xmm0, _X0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _R0(%rsp)

        vmovsd   _X1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fsd_log_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _R1(%rsp)

        vmovapd  _R0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.Lfinishn1)

        ELF_FUNC(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA))



/**** Here starts the main calculations  ****/
/* This is the extra precision log(x) calculation, for use in dpow */

/* Input argument comes in xmm0
 * Output arguments go in xmm0, xmm1
 */
	.text
	ALN_FUNC
#ifdef TARGET_FMA
ENT(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA)):
#else
ENT(__fsd_log_long_vex):
#endif

	vcomisd  .L__real_mindp(%rip), %xmm0
	vmovd 	%xmm0, %rdx
	movq	$0x07fffffffffffffff,%rcx
	jb      LBL(.L__z_or_n_long)

	andq	%rdx, %rcx       /* rcx is ax */
	shrq	$52,%rcx
	sub	$1023,%ecx

LBL(.L__100_long):
	/* log_thresh1 = 9.39412117004394531250e-1 = 0x3fee0faa00000000
	   log_thresh2 = 1.06449508666992187500 = 0x3ff1082c00000000 */
	/* if (ux >= log_thresh1 && ux <= log_thresh2) */
	vmovd %ecx, %xmm6
	vcvtdq2pd %xmm6, %xmm6
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
	vcvtsi2sd %r8d,%xmm1,%xmm1	/* convert index to float */
	movq	$0x03fe0000000000000,%rdx
	orq	%rdx,%rax
	vmovd 	%rax,%xmm2	/* f */

	vmovsd	.L__real_half(%rip),%xmm5	/* .5 */
	vmulsd	.L__real_3f80000000000000(%rip),%xmm1,%xmm1	/* f1 = index/128 */
	leaq	.L__np_ln_lead_table(%rip),%r9
	vmovsd	 -512(%r9,%r8,8),%xmm0			/* z1 */
	vsubsd	%xmm1,%xmm2,%xmm2				/* f2 = f - f1 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,%xmm2,%xmm5,%xmm1
	VFMA_231SD	(%xmm2,%xmm5,%xmm1)
#else
	vmulsd	%xmm2,%xmm5,%xmm5
	vaddsd	%xmm5,%xmm1,%xmm1				/* denominator */
#endif
	vdivsd	%xmm1,%xmm2,%xmm2				/* u */

	/* solve for ln(1+u) */
	vmovapd	%xmm2,%xmm1				/* u */
	vmulsd	%xmm2,%xmm2,%xmm2				/* u^2 */
	vmovsd	.L__real_cb3(%rip),%xmm3
#ifdef TARGET_FMA
#	VFMADDSD	.L__real_cb2(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_cb2(%rip),%xmm2,%xmm3)
#else
	vmulsd	%xmm2,%xmm3,%xmm3				/* Cu2 */
	vaddsd	.L__real_cb2(%rip),%xmm3,%xmm3 		/* B+Cu2 */
#endif
	vmovapd	%xmm2,%xmm5
	vmulsd	%xmm1,%xmm5,%xmm5				/* u^3 */
	vmovapd	%xmm2,%xmm4
	vmulsd	%xmm5,%xmm4,%xmm4				/* u^5 */
	vmovsd	.L__real_log2_lead(%rip),%xmm2
	vmulsd	.L__real_cb1(%rip),%xmm5,%xmm5 		/* Au3 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm3,%xmm4,%xmm4
	VFMA_213SD	(%xmm5,%xmm3,%xmm4)
#else
	vmulsd	%xmm3,%xmm4,%xmm4				/* u5(B+Cu2) */
	vaddsd	%xmm5,%xmm4,%xmm4				/* Au3+u5(B+Cu2) */
#endif

	/* recombine */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm6,%xmm2,%xmm0
	VFMA_231SD	(%xmm6,%xmm2,%xmm0)
#else
	vmulsd	%xmm6,%xmm2,%xmm2				/* xexp * log2_lead */
	vaddsd	%xmm2,%xmm0,%xmm0				/* r1,A */
#endif
	leaq	.L__np_ln_tail_table(%rip),%r9
        vaddsd   -512(%r9,%r8,8),%xmm4,%xmm4                   /* z2+=q ,C */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm4,.L__real_log2_tail(%rip),%xmm6,%xmm6
	VFMA_132SD	(.L__real_log2_tail(%rip),%xmm4,%xmm6)
#else
	vmulsd	.L__real_log2_tail(%rip),%xmm6,%xmm6	/* xexp * log2_tail */
	vaddsd	%xmm4,%xmm6,%xmm6				/* C */
#endif

	/* redistribute the result */
	vmovapd	%xmm1,%xmm2		/* B */
	vaddsd	%xmm6,%xmm1,%xmm1		/* 0xB = B+C */
	vsubsd	%xmm1,%xmm2,%xmm2		/* -0xC = B-Bh */
	vaddsd	%xmm2,%xmm6,%xmm6		/* Ct = C-0xC */

	vmovapd	%xmm0,%xmm3
	vaddsd	%xmm1,%xmm0,%xmm0		/* H = A+0xB */
	vsubsd	%xmm0,%xmm3,%xmm3		/* -Bhead = A-H */
	vaddsd	%xmm3,%xmm1,%xmm1		/* +Btail = 0xB-Bhead */

	vmovapd	%xmm0,%xmm4

	vandpd	.L__real_fffffffff8000000(%rip),%xmm0,%xmm0	/* Head */
	vsubsd	%xmm0,%xmm4,%xmm4		/* Ht = H - Head */
	vaddsd	%xmm4,%xmm1,%xmm1		/* tail = Btail +Ht */


	vaddsd	%xmm6,%xmm1,%xmm1		/* Tail = tail + ct */
	ret

LBL(.L__near_one_long):
	/* saves 10 cycles */
	/* r = x - 1.0; */
	vmovsd	.L__real_two(%rip),%xmm2
	vsubsd	.L__real_one(%rip),%xmm0,%xmm0	/* r */

	/* u = r / (2.0 + r); */
	vaddsd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivsd	%xmm2,%xmm1,%xmm1			/* u */
	vmovsd	.L__real_ca4(%rip),%xmm4	/* D */
	vmovsd	.L__real_ca3(%rip),%xmm5	/* C */
	/* correction = r * u; */
	vmovapd	%xmm0,%xmm6
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm6,%xmm0
	VFNMA_231SD	(%xmm1,%xmm6,%xmm0)
#else
	vmulsd	%xmm1,%xmm6,%xmm6			/* correction */
	vsubsd	%xmm6,%xmm0,%xmm0			/*  -correction	part A */
#endif

	/* u = u + u; */
	vaddsd	%xmm1,%xmm1,%xmm1			/* u */
	vmovapd	%xmm1,%xmm2
	vmulsd	%xmm2,%xmm2,%xmm2			/* v =u^2 */

	/* r2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction); */
	vmulsd	%xmm1,%xmm5,%xmm5			/* Cu */
	vmovapd	%xmm1,%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3			/* u^3 */
	vmulsd	.L__real_ca2(%rip),%xmm2,%xmm2	/* Bu^2 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm3,%xmm4,%xmm5
	VFMA_231SD	(%xmm3,%xmm4,%xmm5)
#else
	vmulsd	%xmm3,%xmm4,%xmm4			/* Du^3 */
	vaddsd	%xmm4,%xmm5,%xmm5			/* Cu+Du3 */
#endif

	vaddsd	.L__real_ca1(%rip),%xmm2,%xmm2	/* +A */
	vmovapd	%xmm3,%xmm1
	vmulsd	%xmm1,%xmm1,%xmm1			/* u^6 */

	vmulsd	%xmm3,%xmm2,%xmm2			/* u3(A+Bu2)	part B */
	vmovapd	%xmm0,%xmm4

	/* we now have 3 terms, develop a head and tail for the sum */

	vmovapd	%xmm2,%xmm3			/* B */
	vaddsd	%xmm3,%xmm0,%xmm0			/* H = A+B */
	vsubsd	%xmm0,%xmm4,%xmm4			/* 0xB = A - H */
	vaddsd	%xmm4,%xmm2,%xmm2			/* Bt = B-0xB */

	vmovapd	%xmm0,%xmm3			/* split the top term */
	vandpd	.L__real_fffffffff8000000(%rip),%xmm0,%xmm0		/* Head */
	vsubsd	%xmm0,%xmm3,%xmm3			/* Ht = H - Head */
	vaddsd	%xmm3,%xmm2,%xmm2			/* Tail = Bt +Ht */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm5,%xmm1,%xmm1
	VFMA_213SD	(%xmm2,%xmm5,%xmm1)
#else
	vmulsd	%xmm5,%xmm1,%xmm1			/* u6(Cu+Du3)	part C */
	vaddsd	%xmm2,%xmm1,%xmm1			/* Tail = tail + C */
#endif
	ret

	/* Start here for all the conditional cases */
	/* we have a zero, a negative number, denorm, or nan. */
LBL(.L__z_or_n_long):
	jp      LBL(.L__lnan)
	vxorpd   %xmm1, %xmm1, %xmm1
	vcomisd  %xmm1, %xmm0
	je      LBL(.L__zero)
	jbe     LBL(.L__negative_x)

	/* A Denormal input, scale appropriately */
	vmulsd   .L__real_scale(%rip), %xmm0, %xmm0
	vmovd 	%xmm0, %rdx
	movq	$0x07fffffffffffffff,%rcx
	andq	%rdx, %rcx       /* rcx is ax */
	shrq	$52,%rcx
	sub	$1075,%ecx
	jmp     LBL(.L__100_long)

        ELF_FUNC(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA))


/* ======================================================================== */

    	.text
    	ALN_FUNC
#ifdef TARGET_FMA
ENT(ASM_CONCAT(__fvd_log_long_,TARGET_VEX_OR_FMA)):
#else
ENT(__fvd_log_long_vex):
#endif
	RZ_PUSH

	vmovdqa	%xmm0, RZ_OFF(40)(%rsp)	/* save the input values */
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm4
	vpxor	%xmm1, %xmm1, %xmm1
	vcmppd	$6, .L__real_maxfp(%rip), %xmm2, %xmm2
	vcmppd 	$1, .L__real_mindp(%rip), %xmm4, %xmm4
	vmovdqa	%xmm0, %xmm3
	vpsrlq	$52, %xmm3, %xmm3
	vorpd	%xmm2, %xmm4, %xmm4
	vpsubq	.L__mask_1023(%rip),%xmm3, %xmm3
	vmovmskpd %xmm4, %r8d
	vpackssdw %xmm1, %xmm3, %xmm3
	vcvtdq2pd %xmm3, %xmm6		/* xexp */
	vmovdqa	%xmm0, %xmm2
	xorq	%rax, %rax
	vsubpd	.L__real_one(%rip), %xmm2, %xmm2
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvd_log_long)

	vmovdqa	%xmm0,%xmm3
	vandpd	.L__real_notsign(%rip),%xmm2, %xmm2
	vpand	.L__real_mant(%rip),%xmm3, %xmm3
	vmovdqa	%xmm3,%xmm4
	vmovapd	.L__real_half(%rip),%xmm5	/* .5 */

	vcmppd	$1,.L__real_threshold(%rip),%xmm2,%xmm2
	vmovmskpd %xmm2,%r10d
	cmp	$3,%r10d
	jz	LBL(.Lall_nearone_long)

	test	$3,%r10d
	jnz	LBL(.L__Scalar_fvd_log_long)

	vpsrlq	$45,%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm2
	vpsrlq	$1,%xmm3,%xmm3
	vpaddq	.L__mask_040(%rip),%xmm3,%xmm3
	vpand	.L__mask_001(%rip),%xmm2,%xmm2
	vpaddq	%xmm2,%xmm3,%xmm3

	vpackssdw %xmm1,%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm1
	xorq	 %rcx,%rcx
	vmovq	 %xmm3,RZ_OFF(24)(%rsp)

	vpor	.L__real_half(%rip),%xmm4,%xmm4
	vmovdqa	%xmm4,%xmm2
	vmulpd	.L__real_3f80000000000000(%rip),%xmm1,%xmm1	/* f1 = index/128 */

	leaq	.L__np_ln_lead_table(%rip),%rdx
	mov	RZ_OFF(24)(%rsp),%eax
	mov	RZ_OFF(20)(%rsp),%ecx

	vsubpd	%xmm1,%xmm2,%xmm2				/* f2 = f - f1 */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,%xmm2,%xmm5,%xmm1
	VFMA_231PD	(%xmm2,%xmm5,%xmm1)
#else
	vmulpd	%xmm2,%xmm5,%xmm5
	vaddpd	%xmm5,%xmm1,%xmm1
#endif

	vdivpd	%xmm1,%xmm2,%xmm2				/* u */

	vmovsd	 -512(%rdx,%rax,8),%xmm0		/* z1 */
	vmovhpd	 -512(%rdx,%rcx,8),%xmm0,%xmm0		/* z1 */


	vmovapd	%xmm2,%xmm1				/* u */
	vmulpd	%xmm2,%xmm2,%xmm2				/* u^2 */
	vmovapd	%xmm2,%xmm5
	vmovapd	.L__real_cb3(%rip),%xmm3
	vmulpd	%xmm1,%xmm5,%xmm5				/* u^3 */
#ifdef TARGET_FMA
#	VFMADDPD	.L__real_cb2(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_cb2(%rip),%xmm2,%xmm3)
#else
	vmulpd	%xmm2,%xmm3,%xmm3				/* Cu2 */
	vaddpd	.L__real_cb2(%rip),%xmm3,%xmm3 		/* B+Cu2 */
#endif

	vmulpd	%xmm5,%xmm2,%xmm2				/* u^5 */
	vmovapd	.L__real_log2_lead(%rip),%xmm4
	vmulpd	.L__real_cb1(%rip),%xmm5,%xmm5 		/* Au3 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm3,%xmm2,%xmm2
	VFMA_213PD	(%xmm5,%xmm3,%xmm2)
#else
	vmulpd	%xmm3,%xmm2,%xmm2				/* u5(B+Cu2) */
	vaddpd	%xmm5,%xmm2,%xmm2				/* u+Au3 */
#endif

	/* table lookup */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	vmovsd	 -512(%rdx,%rax,8),%xmm3		/* z2+=q */
	vmovhpd	 -512(%rdx,%rcx,8),%xmm3,%xmm3		/* z2+=q */


	/* recombine */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm0,%xmm6,%xmm4,%xmm0
	VFMA_231PD	(%xmm6,%xmm4,%xmm0)
	vaddpd		%xmm3, %xmm2,%xmm2
#	VFMADDPD	%xmm2,.L__real_log2_tail(%rip),%xmm6,%xmm6
	VFMA_132PD	(.L__real_log2_tail(%rip),%xmm2,%xmm6)
#else
	vmulpd	%xmm6,%xmm4,%xmm4				/* xexp * log2_lead */
	vaddpd	%xmm4,%xmm0,%xmm0				/* r1 */
	vaddpd	%xmm3, %xmm2,%xmm2
	vmulpd	.L__real_log2_tail(%rip),%xmm6,%xmm6
	vaddpd	%xmm2, %xmm6,%xmm6
#endif

	/* redistribute the result */
	vmovapd	%xmm1,%xmm2		/* B */
	vaddpd	%xmm6,%xmm1,%xmm1		/* 0xB = B+C */
	vsubpd	%xmm1,%xmm2,%xmm2		/* -0xC = B-Bh */
	vaddpd	%xmm2,%xmm6,%xmm6		/* Ct = C-0xC */

	vmovapd	%xmm0,%xmm3
	vaddpd	%xmm1,%xmm0,%xmm0		/* H = A+0xB */
	vsubpd	%xmm0,%xmm3,%xmm3		/* -Bhead = A-H */
	vaddpd	%xmm3,%xmm1,%xmm1		/* +Btail = 0xB-Bhead */

	vmovapd	%xmm0,%xmm4

	vandpd	.L__real_fffffffff8000000(%rip),%xmm0,%xmm0	/* Head */
	vsubpd	%xmm0,%xmm4,%xmm4		/* Ht = H - Head */
	vaddpd	%xmm4,%xmm1,%xmm1		/* tail = Btail +Ht */
	vaddpd	%xmm6,%xmm1,%xmm1		/* Tail = tail + ct */

LBL(.Lfinishn1_long):
	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.Lall_nearone_long):
	vmovapd	.L__real_two(%rip),%xmm2
	vsubpd	.L__real_one(%rip),%xmm0,%xmm0	/* r */

	vaddpd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivpd	%xmm2,%xmm1,%xmm1			/* u */
	vmovapd	.L__real_ca4(%rip),%xmm4  	/* D */
	vmovapd	.L__real_ca3(%rip),%xmm5 	/* C */

	vmovapd	%xmm0,%xmm6
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm0,%xmm1,%xmm6,%xmm0
	VFNMA_231PD	(%xmm1,%xmm6,%xmm0)
#else
	vmulpd	%xmm1,%xmm6,%xmm6			/* correction */
	vsubpd	%xmm6,%xmm0,%xmm0			/*  -correction	part A */
#endif

	vaddpd	%xmm1,%xmm1,%xmm1			/* u */
	vmovapd	%xmm1,%xmm2
	vmulpd	%xmm2,%xmm2,%xmm2			/* v =u^2 */

	vmulpd	%xmm1,%xmm5,%xmm5			/* Cu */
	vmovapd	%xmm1,%xmm3
	vmulpd	%xmm2,%xmm3,%xmm3			/* u^3 */
	vmulpd	.L__real_ca2(%rip),%xmm2,%xmm2	/* Bu^2 */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm3,%xmm4,%xmm5
	VFMA_231PD	(%xmm3,%xmm4,%xmm5)
#else
	vmulpd	%xmm3,%xmm4,%xmm4			/* Du^3 */
	vaddpd	%xmm4,%xmm5,%xmm5			/* Cu+Du3 */
#endif

	vaddpd	.L__real_ca1(%rip),%xmm2,%xmm2	/* +A */
	vmovapd	%xmm3,%xmm1
	vmulpd	%xmm1,%xmm1,%xmm1			/* u^6 */
/*	vaddpd	%xmm4,%xmm5,%xmm5 */			/* Cu+Du3 */
/*	vsubpd	%xmm6,%xmm0,%xmm0 */			/*  -correction	part A */

	vmulpd	%xmm3,%xmm2,%xmm2			/* u3(A+Bu2)	part B */
	vmovapd	%xmm0,%xmm4
	vmulpd	%xmm5,%xmm1,%xmm1			/* u6(Cu+Du3)	part C */

	/* we now have 3 terms, develop a head and tail for the sum */

	vmovapd	%xmm2,%xmm3			/* B */
	vaddpd	%xmm3,%xmm0,%xmm0			/* H = A+B */
	vsubpd	%xmm0,%xmm4,%xmm4			/* 0xB = A - H */
	vaddpd	%xmm4,%xmm2,%xmm2			/* Bt = B-0xB */

	vmovapd	%xmm0,%xmm3			/* split the top term */
	vandpd	.L__real_fffffffff8000000(%rip),%xmm0,%xmm0		/* Head */
	vsubpd	%xmm0,%xmm3,%xmm3			/* Ht = H - Head */
	vaddpd	%xmm3,%xmm2,%xmm2			/* Tail = Bt +Ht */
	vaddpd	%xmm2,%xmm1,%xmm1			/* Tail = tail + C */
	jmp	LBL(.Lfinishn1_long)

#define _T0 48
#define _T1 56

LBL(.L__Scalar_fvd_log_long):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovapd  %xmm0, _X0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _R0(%rsp)
        vmovsd   %xmm1, _T0(%rsp)

        vmovsd   _X1(%rsp), %xmm0

        CALL(ENT(ASM_CONCAT(__fsd_log_long_,TARGET_VEX_OR_FMA)))


        vmovsd   %xmm0, _R1(%rsp)
        vmovsd   %xmm1, _T1(%rsp)

        vmovapd  _R0(%rsp), %xmm0
        vmovapd  _T0(%rsp), %xmm1
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.Lfinishn1_long)

        ELF_FUNC(ASM_CONCAT(__fvd_log_long_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_log_long_,TARGET_VEX_OR_FMA))



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
	.globl ENT(ASM_CONCAT(__fvs_log_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_log_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(72)(%rsp)
	vmovdqu	%ymm7, RZ_OFF(104)(%rsp)
#endif

/* Fast vector natural logarithm code goes here... */
        /* First check for valid input:
         * if (a .gt. 0.0) then */
	vmovaps  .L4_384(%rip), %xmm4	/* Move min arg to xmm4 */
	vxorps	%xmm7, %xmm7, %xmm7		/* Still need 0.0 */
	vmovaps	%xmm0, %xmm2		/* Move for nx */
	vmovaps	%xmm0, %xmm1		/* Move to xmm1 for later ma */

	/* Check exceptions and valid range */
	vcmpleps	%xmm0, %xmm4, %xmm4		/* '00800000'x <= a, xmm4 1 where true */
	vcmpltps	%xmm0, %xmm7, %xmm7		/* Test for 0.0 < a, xmm7 1 where true */
	vcmpneqps	.L4_387(%rip), %xmm0, %xmm0	/* Test for == +inf */
	vxorps		%xmm7, %xmm4, %xmm4		/* xor to find just denormal inputs */
	vmovmskps	%xmm4, %eax		/* Move denormal mask to gp ref */
	vmovaps		%xmm2, RZ_OFF(24)(%rsp)	/* Move for exception processing */
	vmovaps		.L4_382(%rip), %xmm3	/* Move 126 */
	cmp		$0, %eax		/* Test for denormals */
	jne		LBL(.LB_DENORMs)

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
	vandps	.L4_380(%rip), %xmm1, %xmm1	/* ma = IAND(ia,'007fffff'x) */
	vpsrld	$23, %xmm2, %xmm2		/* nx = ISHFT(ia,-23) */
	vandps	%xmm0, %xmm7, %xmm7		/* Mask for nan, inf, neg and 0.0 */
	vmovaps	%xmm1, %xmm6		/* move ma for ig */
	vpsubd	.L4_381(%rip), %xmm1, %xmm1	/* ms = ma - '3504f3'x */
	vpsubd	%xmm3, %xmm2, %xmm2		/* nx = ISHFT(ia,-23) - 126 */
	vorps	.L4_383(%rip), %xmm6, %xmm6	/* ig = IOR(ma,'3f000000'x) */
	vmovaps	%xmm1, %xmm0		/* move ms for tbl ms */
	vandps	.L4_384(%rip), %xmm1, %xmm1	/* mx = IAND(ms,'00800000'x) */
	vandps	.L4_385(%rip), %xmm0, %xmm0	/* ms = IAND(ms,'007f0000'x) */
	vorps	%xmm1, %xmm6, %xmm6		/* ig = IOR(ig, mx) */
	vpsrad	$23, %xmm1, %xmm1		/* ISHFT(mx,-23) */
	vpsrad	$12, %xmm0, %xmm0		/* ISHFT(ms,-12) for 128 bit reads */
	vmovmskps %xmm7, %eax		/* Move xmm7 mask to eax */
	vpsubd	%xmm1, %xmm2, %xmm2		/* nx = nx - ISHFT(mx,-23) */
	vmovaps	%xmm0, RZ_OFF(40)(%rsp)	/* Move to memory */
	vcvtdq2ps  %xmm2, %xmm0		/* xn = real(nx) */

	movl	RZ_OFF(40)(%rsp), %ecx		/* Move to gp register */
	vmovaps	(%r8,%rcx,1), %xmm1		/* Read from 1st table location */
	movl	RZ_OFF(36)(%rsp), %edx		/* Move to gp register */
	vmovaps	(%r8,%rdx,1), %xmm2		/* Read from 2nd table location */
	movl	RZ_OFF(32)(%rsp), %ecx		/* Move to gp register */
	vmovaps	(%r8,%rcx,1), %xmm3		/* Read from 3rd table location */
	movl	RZ_OFF(28)(%rsp), %edx		/* Move to gp register */
	vmovaps	(%r8,%rdx,1), %xmm4		/* Read from 4th table location */

	/* So, we do 4 reads of a,b,c into registers xmm1, xmm2, xmm3, xmm4
	 * Assume we need to keep rg in xmm6, xn in xmm0
	 * The following shuffle gets them into SIMD mpy form:
	 */

	vsubps	.L4_386(%rip), %xmm6, %xmm6 	/* x0 = rg - 1.0 */

	vmovaps	%xmm1, %xmm5		/* Store 1/3, c0, b0, a0 */
	vmovaps	%xmm3, %xmm7		/* Store 1/3, c2, b2, a2 */

	vunpcklps %xmm2, %xmm1, %xmm1		/* b1, b0, a1, a0 */
	vunpcklps %xmm4, %xmm3, %xmm3		/* b3, b2, a3, a2 */
	vunpckhps %xmm2, %xmm5, %xmm5		/* 1/3, 1/3, c1, c0 */
	vunpckhps %xmm4, %xmm7, %xmm7		/* 1/3, 1/3, c3, c2 */

	vmovaps	%xmm6, %xmm4		/* move x0 */

	vmovaps		%xmm1, %xmm2		/* Store b1, b0, a1, a0 */
	vmovlhps	%xmm3, %xmm1, %xmm1		/* a3, a2, a1, a0 */
	vmovlhps	%xmm7, %xmm5, %xmm5		/* c3, c2, c1, c0 */
	vmovhlps	%xmm2, %xmm3, %xmm3		/* b3, b2, b1, b0 */

/* ==zz== #ifdef TARGET_FMA
#	VFMADDPS	%xmm3,%xmm6,%xmm1,%xmm1
	VFMA_213PS	(%xmm3,%xmm6,%xmm1)
#else */
	vmulps		%xmm6, %xmm1, %xmm1		/* COEFFS(mt) * x0 */
	vaddps		%xmm3, %xmm1, %xmm1		/* COEFFS(mt) * g + COEFFS(mt+1) */
/* ==zz== #endif */
	vmulps		%xmm6, %xmm6, %xmm6		/* xsq = x0 * x0 */
	vmovhlps	%xmm7, %xmm7, %xmm7		/* 1/3, 1/3, 1/3, 1/3 */

	vmovaps	%xmm4, %xmm2		/* move x0 */

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
/*	vaddps	%xmm3, %xmm1, %xmm1 */		/* COEFFS(mt) * g + COEFFS(mt+1) */
	vmulps	%xmm6, %xmm4, %xmm4		/* xcu = xsq * x0 */
	vmulps	.L4_383(%rip), %xmm6, %xmm6	/* x1 = 0.5 * xsq */
#ifdef TARGET_FMA
#	VFMADDPS	%xmm5,%xmm2,%xmm1,%xmm1
	VFMA_213PS	(%xmm5,%xmm2,%xmm1)
#else
	vmulps	%xmm2, %xmm1, %xmm1		/* * x0 */
	vaddps	%xmm5, %xmm1, %xmm1		/* + COEFFS(mt+2) = rp */
#endif
	vmulps	%xmm7, %xmm4, %xmm4		/* x2 = thrd * xcu */
	vmovaps	%xmm6, %xmm3		/* move x1 */
#ifdef TARGET_FMA
#	VFNMADDPS	%xmm1,%xmm6,%xmm6,%xmm1
	VFNMA_231PS	(%xmm6,%xmm6,%xmm1)
#else
	vmulps	%xmm6, %xmm6, %xmm6		/* x3 = x1 * x1 */
/*	vaddps	%xmm5, %xmm1, %xmm1 */		/* + COEFFS(mt+2) = rp */
	vsubps	%xmm6, %xmm1, %xmm1		/* rp - x3 */
#endif
	vmovaps	.L4_388(%rip), %xmm7	/* Move c1 */
        vmovaps  .L4_389(%rip), %xmm6	/* Move c2 */
	vaddps	%xmm1, %xmm4, %xmm4		/* rp - x3 + x2 */
	vsubps	%xmm3, %xmm4, %xmm4		/* rp - x3 + x2 - x1 */
	vaddps	%xmm2, %xmm4, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
#ifdef TARGET_FMA
#	VFMADDPS	%xmm4,%xmm0,%xmm7,%xmm4
	VFMA_231PS	(%xmm0,%xmm7,%xmm4)
#	VFMADDPS	%xmm4,%xmm6,%xmm0,%xmm0
	VFMA_213PS	(%xmm4,%xmm6,%xmm0)
#else
	vmulps   %xmm0, %xmm7, %xmm7		/* xn * c1 */
	vaddps   %xmm7, %xmm4, %xmm4		/* (xn * c1 + rz) */
        vmulps   %xmm6, %xmm0, %xmm0		/* xn * c2 */
        vaddps   %xmm4, %xmm0, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */
#endif

	/* Compare exception mask now and jump if no exceptions */
	cmp	$15, %eax
	jne 	LBL(.LB_EXCEPTs)

LBL(.LB_900):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(72)(%rsp), %ymm6
	vmovdqu	RZ_OFF(104)(%rsp), %ymm7
#endif

	RZ_POP
	rep
	ret

LBL(.LB_EXCEPTs):
        /* Handle all exceptions by masking in xmm */
        vmovaps  RZ_OFF(24)(%rsp), %xmm1	/* original input */
        vmovaps  RZ_OFF(24)(%rsp), %xmm2	/* original input */
        vmovaps  RZ_OFF(24)(%rsp), %xmm3	/* original input */
        vxorps   %xmm7, %xmm7, %xmm7            /* xmm7 = 0.0 */
        vxorps   %xmm6, %xmm6, %xmm6            /* xmm6 = 0.0 */
	vmovaps	.L4_394(%rip), %xmm5	/* convert nan bit */
        vxorps   %xmm4, %xmm4, %xmm4            /* xmm4 = 0.0 */

        vcmpunordps %xmm1, %xmm7, %xmm7		/* Test if unordered */
        vcmpltps %xmm6, %xmm2, %xmm2		/* Test if a < 0.0 */
        vcmpordps %xmm1, %xmm6, %xmm6		/* Test if ordered */

        vandps	%xmm7, %xmm5, %xmm5            /* And nan bit where unordered */
        vorps	%xmm7, %xmm4, %xmm4            /* Or all masks together */
        vandps	%xmm1, %xmm7, %xmm7            /* And input where unordered */
	vorps	%xmm5, %xmm7, %xmm7		/* Convert unordered nans */

        vxorps   %xmm5, %xmm5, %xmm5            /* xmm5 = 0.0 */
        vandps   %xmm2, %xmm6, %xmm6            /* Must be ordered and < 0.0 */
        vorps    %xmm6, %xmm4, %xmm4            /* Or all masks together */
        vandps   .L4_390(%rip), %xmm6, %xmm6    /* And -nan if < 0.0 and ordered */

        vcmpeqps .L4_387(%rip), %xmm3, %xmm3	/* Test if equal to infinity */
        vcmpeqps %xmm5, %xmm1, %xmm1		/* Test if eq 0.0 */
        vorps    %xmm6, %xmm7, %xmm7            /* or in < 0.0 */

        vorps    %xmm3, %xmm4, %xmm4            /* Or all masks together */
        vandps   .L4_387(%rip), %xmm3, %xmm3    /* inf and inf mask */
        vmovaps  %xmm0, %xmm2
        vorps    %xmm3, %xmm7, %xmm7            /* or in infinity */

        vorps    %xmm1, %xmm4, %xmm4            /* Or all masks together */
        vandps   .L4_391(%rip), %xmm1, %xmm1    /* And -inf if == 0.0 */
        vmovaps  %xmm4, %xmm0
        vorps    %xmm1, %xmm7, %xmm7            /* or in -infinity */

        vandnps  %xmm2, %xmm0, %xmm0            /* Where mask not set, use result */
        vorps    %xmm7, %xmm0, %xmm0            /* or in exceptional values */
	jmp	LBL(.LB_900)

LBL(.LB_DENORMs):
	/* Have the denorm mask in xmm4, so use it to scale a and the subtractor */
	vmovaps	%xmm4, %xmm5		/* Move mask */
	vmovaps	%xmm4, %xmm6		/* Move mask */
	vandps	.L4_392(%rip), %xmm4, %xmm4	/* Have 2**23 where denorms are, 0 else */
	vandnps	%xmm1, %xmm5, %xmm5		/* Have a where denormals aren't */
	vmulps	%xmm4, %xmm1, %xmm1		/* denormals * 2**23 */
	vandps	.L4_393(%rip), %xmm6, %xmm6	/* have 23 where denorms are, 0 else */
	vorps	%xmm5, %xmm1, %xmm1		/* Or in the original a */
	vpaddd	%xmm6, %xmm3, %xmm3		/* Add 23 to 126 for offseting exponent */
	vmovaps	%xmm1, %xmm2		/* Move to the next location */
	jmp	LBL(.LB_100)

	ELF_FUNC(ASM_CONCAT(__fvs_log_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fvs_log_,TARGET_VEX_OR_FMA))


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
 *  float __fss_log(float f)
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
	.globl	ENT(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(96)(%rsp)
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

	vmovss	%xmm0, RZ_OFF(4)(%rsp)
        vmovss	.L4_384(%rip), %xmm2	/* Move smallest normalized number */
	movl	RZ_OFF(4)(%rsp), %ecx
	andl	$8388607, %ecx		/* ma = IAND(ia,'007fffff'x) */
	leaq 	-3474675(%rcx), %rdx	/* ms = ma - '3504f3'x */
	orl	$1056964608, %ecx	/* ig = IOR(ma,'3f000000'x) */
	vcmpnless %xmm0, %xmm2, %xmm2		/* '00800000'x <= a, xmm2 1 where not */
        vcmpeqss	.L4_387(%rip), %xmm0, %xmm0	/* Test for == +inf */
	movl	%edx, %eax		/* move ms */
	andl	$8388608, %edx		/* mx = IAND(ms,'00800000'x) */
	orl	%edx, %ecx		/* ig = IOR(ig,mx) */
	movl	%ecx, RZ_OFF(8)(%rsp)	/* move back over to fp sse */
	shrl	$23, %edx		/* ISHFT(mx,-23) */
        vunpcklps %xmm2, %xmm0, %xmm0		/* Mask for nan, inf, neg and 0.0 */

	leaq	.L_STATICS1(%rip), %r8
	movl	RZ_OFF(4)(%rsp), %ecx	/* ia */
	andl	$8323072, %eax		/* ms = IAND(ms,'007f0000'x) */
	vmovss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	vmovmskps %xmm0, %r9d		/* move exception mask to gp reg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$126, %ecx		/* nx = ISHFT(ia,-23) - 126 */
	vmovss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
        shrl    $14, %eax		/* mt1 */
	and	$3, %r9d		/* mask with 3 */
	vmovss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
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

	vmovd %ecx, %xmm0
	vcvtdq2ps %xmm0, %xmm0
	vsubss	.L4_386(%rip), %xmm1, %xmm1	/* x0 = rg - 1.0 */
	vsubss	.L4_386(%rip), %xmm6, %xmm6	/* x0 = rg - 1.0 */
	vsubss	.L4_386(%rip), %xmm4, %xmm4	/* x0 = rg - 1.0 */
	vsubss	.L4_386(%rip), %xmm2, %xmm2	/* x0 = rg - 1.0 */
	vmulss	(%r8,%rax,4), %xmm1, %xmm1	/* COEFFS(mt) * x0 */
	vmulss   %xmm6, %xmm6, %xmm6		/* xsq = x0 * x0 */
	vaddss	4(%r8,%rax,4), %xmm1, %xmm1	/* COEFFS(mt) * x0 + COEFFS(mt+1) */
	vmulss   %xmm6, %xmm4, %xmm4		/* xcu = xsq * x0 */
	vmulss   .L4_383(%rip), %xmm6, %xmm6	/* x1 = 0.5 * xsq */
	vmovaps	%xmm6, %xmm3		/* move x1 */
#ifdef TARGET_FMA
#	VFMADDSS	8(%r8,%rax,4),%xmm1,%xmm2,%xmm1
	VFMA_213SS	(8(%r8,%rax,4),%xmm2,%xmm1)
#	VFNMADDSS	%xmm1,%xmm6,%xmm6,%xmm1
	VFNMA_231SS	(%xmm6,%xmm6,%xmm1)
#else
	vmulss   %xmm2, %xmm1, %xmm1		/* * x0 */
	vaddss	8(%r8,%rax,4), %xmm1, %xmm1	/* + COEFFS(mt+2) = rp */
	vmulss	%xmm6, %xmm6, %xmm6		/* x3 = x1 * x1 */
	vsubss	%xmm6, %xmm1, %xmm1		/* rp - x3 */
#endif
	vmovss	.L4_388(%rip), %xmm5	/* Move c1 */
        vmovss   .L4_389(%rip), %xmm6	/* Move c2 */

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDSS	%xmm1,12(%r8,%rax,4),%xmm4,%xmm4
	VFMA_213SS	(VFMADDSS,%xmm1,12(%r8,%rax,4),%xmm4)
#else */
	vmulss	12(%r8,%rax,4), %xmm4, %xmm4	/* x2 = thrd * xcu */
	vaddss	%xmm1, %xmm4, %xmm4		/* rp - x3 + x2 */
/* #endif */
	vsubss	%xmm3, %xmm4, %xmm4		/* rp - x3 + x2 - x1 */
	vaddss	%xmm2, %xmm4, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
#ifdef TARGET_FMA
#	VFMADDSS	%xmm4,%xmm0,%xmm5,%xmm4
	VFMA_231SS	(%xmm0,%xmm5,%xmm4)
#	VFMADDSS	%xmm4,%xmm0,%xmm6,%xmm0
	VFMA_213SS	(%xmm4,%xmm6,%xmm0)
#else
	vmulss   %xmm0, %xmm5, %xmm5		/* xn * c1 */
	vaddss   %xmm5, %xmm4, %xmm4		/* (xn * c1 + rz) */
        vmulss   %xmm6, %xmm0, %xmm0		/* xn * c2 */
        vaddss   %xmm4, %xmm0, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */
#endif

LBL(.LB1_900):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(96)(%rsp), %ymm6
#endif
	RZ_POP
	rep
	ret

	ALN_WORD
LBL(.LB1_800):
	/* ir = 'ff800000'x */
	xorq	%rax,%rax
	vmovss	RZ_OFF(4)(%rsp), %xmm0
	vmovd 	%rax, %xmm1
	vcomiss	%xmm1, %xmm0
	jp	LBL(.LB1_cvt_nan)
#ifdef FMATH_EXCEPTIONS
	vmovss  .L4_386(%rip), %xmm1
	vdivss	%xmm0, %xmm1, %xmm0	/* Generate div-by-zero op when x=0 */
#endif
	vmovss	.L4_391(%rip),%xmm0	/* Move -inf */
	je	LBL(.LB1_900)
#ifdef FMATH_EXCEPTIONS
	vsqrtss	%xmm0, %xmm0, %xmm0	/* Generate invalid op when x < 0 */
#endif
	vmovss	.L4_390(%rip),%xmm0	/* Move -nan */
	jb	LBL(.LB1_900)
	vmovss	.L4_387(%rip), %xmm0	/* Move +inf */
	vmovss	RZ_OFF(4)(%rsp), %xmm1
	vcomiss	%xmm1, %xmm0
	je	LBL(.LB1_900)

	/* Otherwise, we had a denormal as an input */
	vmulss	.L4_392(%rip), %xmm1, %xmm1	/* a * scale factor */
	vmovss	%xmm1, RZ_OFF(4)(%rsp)
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
	vmovss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$149, %ecx		/* nx = ISHFT(ia,-23) - (126 + 23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
        shrl    $14, %eax		/* mt1 */
	jmp	LBL(.LB1_100)

LBL(.LB1_cvt_nan):
	vmovss	.L4_394(%rip), %xmm1	/* nan bit */
	vorps	%xmm1, %xmm0, %xmm0
	jmp	LBL(.LB1_900)

	ELF_FUNC(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_log_,TARGET_VEX_OR_FMA))



/* ======================================================================== */

/* Log10, at the urging of John Levesque */

    	.text
    	ALN_FUNC
	.globl ENT(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(72)(%rsp)
#endif

	vmovdqa	%xmm0, RZ_OFF(40)(%rsp)	/* save the input values */
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm4
	vpxor	%xmm1, %xmm1, %xmm1
	vcmppd	$6, .L__real_maxfp(%rip), %xmm2, %xmm2
	vcmppd 	$1, .L__real_mindp(%rip), %xmm4, %xmm4
	vmovdqa	%xmm0, %xmm3
	vpsrlq	$52, %xmm3, %xmm3
	vorpd	%xmm2, %xmm4, %xmm4
	vpsubq	.L__mask_1023(%rip),%xmm3, %xmm3
	vmovmskpd %xmm4, %r8d
	vpackssdw %xmm1, %xmm3, %xmm3
	vcvtdq2pd %xmm3, %xmm6		/* xexp */
	vmovdqa	%xmm0, %xmm2
	xorq	%rax, %rax
	vsubpd	.L__real_one(%rip), %xmm2, %xmm2
	test	$3, %r8d
	jnz	LBL(.L__Scalar_fvdlog10)

	vmovdqa	%xmm0,%xmm3
	vandpd	.L__real_notsign(%rip),%xmm2,%xmm2
	vpand	.L__real_mant(%rip),%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm4
	vmovapd	.L__real_half(%rip),%xmm5	/* .5 */

	vcmppd	$1,.L__real_threshold(%rip),%xmm2,%xmm2
	vmovmskpd %xmm2,%r10d
	cmp	$3,%r10d
	jz	LBL(.Lall_nearone_log10)

	vpsrlq	$45,%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm2
	vpsrlq	$1,%xmm3,%xmm3
	vpaddq	.L__mask_040(%rip),%xmm3,%xmm3
	vpand	.L__mask_001(%rip),%xmm2,%xmm2
	vpaddq	%xmm2,%xmm3,%xmm3

	vpackssdw %xmm1,%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm1
	xorq	 %rcx,%rcx
	vmovq	 %xmm3,RZ_OFF(24)(%rsp)

	vpor	.L__real_half(%rip),%xmm4,%xmm4
	vmovdqa	%xmm4,%xmm2
	vmulpd	.L__real_3f80000000000000(%rip),%xmm1,%xmm1	/* f1 = index/128 */

	leaq	.L__np_ln_lead_table(%rip),%rdx
	mov	RZ_OFF(24)(%rsp),%eax

	vsubpd	%xmm1,%xmm2,%xmm2				/* f2 = f - f1 */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,%xmm2,%xmm5,%xmm1
	VFMA_231PD	(%xmm2,%xmm5,%xmm1)
#else
	vmulpd	%xmm2,%xmm5,%xmm5
	vaddpd	%xmm5,%xmm1,%xmm1
#endif

	vdivpd	%xmm1,%xmm2,%xmm2				/* u */

	vmovsd	 -512(%rdx,%rax,8),%xmm0		/* z1 */
	mov	RZ_OFF(20)(%rsp),%ecx
	vmovhpd	 -512(%rdx,%rcx,8),%xmm0,%xmm0		/* z1 */
	vmovapd	%xmm2,%xmm1				/* u */
	vmulpd	%xmm2,%xmm2,%xmm2				/* u^2 */
	vmovapd	%xmm2,%xmm5
	vmovapd	.L__real_cb3(%rip),%xmm3
	vmulpd	%xmm1,%xmm5,%xmm5				/* u^3 */
#ifdef TARGET_FMA
#	VFMADDPD	.L__real_cb2(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_cb2(%rip),%xmm2,%xmm3)
#else
	vmulpd	%xmm2,%xmm3,%xmm3				/* Cu2 */
	vaddpd	.L__real_cb2(%rip),%xmm3,%xmm3 		/* B+Cu2 */
#endif

	vmulpd	%xmm5,%xmm2,%xmm2				/* u^5 */
	vmovapd	.L__real_log2_lead(%rip),%xmm4

#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,.L__real_cb1(%rip),%xmm5,%xmm1
	VFMA_231PD	(.L__real_cb1(%rip),%xmm5,%xmm1)
#	VFMADDPD	%xmm1,%xmm3,%xmm2,%xmm1
	VFMA_231PD	(%xmm3,%xmm2,%xmm1)
#else
	vmulpd	.L__real_cb1(%rip),%xmm5,%xmm5 		/* Au3 */
	vaddpd	%xmm5,%xmm1,%xmm1				/* u+Au3 */
	vmulpd	%xmm3,%xmm2,%xmm2				/* u5(B+Cu2) */
	vaddpd	%xmm2,%xmm1,%xmm1				/* poly */
#endif

	vmovapd	%xmm0,%xmm3

#ifdef TARGET_FMA
#	VFMADDPD	%xmm0,%xmm6,%xmm4,%xmm0
	VFMA_231PD	(%xmm6,%xmm4,%xmm0)
#	VFMADDPD	%xmm3,%xmm6,%xmm4,%xmm3
	VFMA_231PD	(%xmm6,%xmm4,%xmm3)
#else
	vmulpd	%xmm6,%xmm4,%xmm4				/* xexp * log2_lead */
	vaddpd	%xmm4,%xmm0,%xmm0				/* r1 */
	vaddpd	%xmm4,%xmm3,%xmm3				/* r1 */
#endif
	leaq	.L__np_ln_tail_table(%rip),%rdx
	vmovsd	 -512(%rdx,%rax,8),%xmm4		/* z2+=q */
	vmovhpd	 -512(%rdx,%rcx,8),%xmm4,%xmm4		/* z2+=q */

	vaddpd	%xmm4,%xmm1,%xmm1

	vmulpd	.L__log10_multiplier2(%rip),%xmm3,%xmm3
	vmulpd	.L__log10_multiplier1(%rip),%xmm0,%xmm0
#ifdef TARGET_FMA
#	VFMADDPD	%xmm1,.L__real_log2_tail(%rip),%xmm6,%xmm1
	VFMA_231PD	(.L__real_log2_tail(%rip),%xmm6,%xmm1)
#	VFMADDPD	%xmm3,.L__log10_multiplier(%rip),%xmm1,%xmm1
	VFMA_132PD	(.L__log10_multiplier(%rip),%xmm3,%xmm1)
#else
	vmulpd	.L__real_log2_tail(%rip),%xmm6,%xmm6
	vaddpd	%xmm6,%xmm1,%xmm1				/* r2 */
	vmulpd	.L__log10_multiplier(%rip),%xmm1,%xmm1
	vaddpd	%xmm3,%xmm1,%xmm1
#endif
	vaddpd	%xmm1,%xmm0,%xmm0

	test	$3,%r10d
	jnz	LBL(.Lnear_one_log10)

LBL(.Lfinishn1_log10):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(72)(%rsp), %ymm6
#endif
	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.Lall_nearone_log10):
	vmovapd	.L__real_two(%rip),%xmm2
	vsubpd	.L__real_one(%rip),%xmm0,%xmm0	/* r */
	vaddpd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivpd	%xmm2,%xmm1,%xmm1			/* u */
	vmovapd	.L__real_ca4(%rip),%xmm4  	/* D */
	vmovapd	.L__real_ca3(%rip),%xmm5 	/* C */
	vmovapd	%xmm0,%xmm6
	vmulpd	%xmm1,%xmm6,%xmm6			/* correction */
	vaddpd	%xmm1,%xmm1,%xmm1			/* u */
	vmovapd	%xmm1,%xmm2
	vmulpd	%xmm2,%xmm2,%xmm2			/* v =u^2 */
	vmulpd	%xmm1,%xmm5,%xmm5			/* Cu */
	vmovapd	%xmm1,%xmm3
	vmulpd	%xmm2,%xmm3,%xmm3			/* u^3 */
	vmulpd	.L__real_ca2(%rip),%xmm2,%xmm2	/* Bu^2 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm3,%xmm4,%xmm5
	VFMA_231PD	(%xmm3,%xmm4,%xmm5)
#else
	vmulpd	%xmm3,%xmm4,%xmm4			/* Du^3 */
	vaddpd	%xmm4,%xmm5,%xmm5			/* Cu+Du3 */
#endif

	vaddpd	.L__real_ca1(%rip),%xmm2,%xmm2	/* +A */
	vmovapd	%xmm3,%xmm1
	vmulpd	%xmm1,%xmm1,%xmm1			/* u^6 */
/*	vaddpd	%xmm4,%xmm5,%xmm5 */			/* Cu+Du3 */

	vmovapd	%xmm0,%xmm4
	vmulpd	%xmm3,%xmm2,%xmm2			/* u3(A+Bu2) */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,%xmm5,%xmm1,%xmm2
	VFMA_231PD	(%xmm5,%xmm1,%xmm2)
#else
	vmulpd	%xmm5,%xmm1,%xmm1			/* u6(Cu+Du3) */
	vaddpd	%xmm1,%xmm2,%xmm2
#endif
	vsubpd	%xmm6,%xmm2,%xmm2			/*  -correction */

/*	vmulpd	.L__log10_multiplier1(%rip),%xmm0,%xmm0 */
	vmulpd	.L__log10_multiplier2(%rip),%xmm4,%xmm4
#ifdef TARGET_FMA
#	VFMADDPD	%xmm4,.L__log10_multiplier(%rip),%xmm2,%xmm2
	VFMA_132PD	(.L__log10_multiplier(%rip),%xmm4,%xmm2)
#	VFMADDPD	%xmm2,.L__log10_multiplier1(%rip),%xmm0,%xmm0
	VFMA_132PD	(.L__log10_multiplier1(%rip),%xmm2,%xmm0)
#else
	vmulpd	.L__log10_multiplier(%rip),%xmm2,%xmm2
	vaddpd	%xmm4,%xmm2,%xmm2
	vmulpd	.L__log10_multiplier1(%rip),%xmm0,%xmm0
	vaddpd	%xmm2,%xmm0,%xmm0
#endif
	jmp	LBL(.Lfinishn1_log10)

	ALN_QUAD
LBL(.Lnear_one_log10):
	test	$1,%r10d
	jz	LBL(.Llnn12_log10)

	vmovlpd	RZ_OFF(40)(%rsp),%xmm0,%xmm0
	call	LBL(.Lln1_log10)

LBL(.Llnn12_log10):
	test	$2,%r10d			/* second number? */
	jz	LBL(.Llnn1e_log10)
	vmovlpd	%xmm0,RZ_OFF(40)(%rsp)
	vmovsd	RZ_OFF(32)(%rsp),%xmm0
	call	LBL(.Lln1_log10)
	vmovlpd	%xmm0,RZ_OFF(32)(%rsp)
	vmovapd	RZ_OFF(40)(%rsp),%xmm0

LBL(.Llnn1e_log10):
	jmp	LBL(.Lfinishn1_log10)

LBL(.Lln1_log10):
	vmovsd	.L__real_two(%rip),%xmm2
	vsubsd	.L__real_one(%rip),%xmm0,%xmm0	/* r */
	vaddsd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivsd	%xmm2,%xmm1,%xmm1			/* u */
	vmovsd	.L__real_ca4(%rip),%xmm4	/* D */
	vmovsd	.L__real_ca3(%rip),%xmm5	/* C */
	vmovapd	%xmm0,%xmm6
	vmulsd	%xmm1,%xmm6,%xmm6			/* correction */
	vaddsd	%xmm1,%xmm1,%xmm1			/* u */
	vmovapd	%xmm1,%xmm2
	vmulsd	%xmm2,%xmm2,%xmm2			/* v =u^2 */
	vmulsd	%xmm1,%xmm5,%xmm5			/* Cu */
	vmovapd	%xmm1,%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3			/* u^3 */
	vmulsd	.L__real_ca2(%rip),%xmm2,%xmm2	/*Bu^2 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm3,%xmm4,%xmm5
	VFMA_231SD	(%xmm3,%xmm4,%xmm5)
#else
	vmulsd	%xmm3,%xmm4,%xmm4			/*Du^3 */
	vaddsd	%xmm4,%xmm5,%xmm5			/* Cu+Du3 */
#endif
	vaddsd	.L__real_ca1(%rip),%xmm2,%xmm2	/* +A */
	vmovapd	%xmm3,%xmm1
	vmulsd	%xmm1,%xmm1,%xmm1			/* u^6 */
/*	vaddsd	%xmm4,%xmm5,%xmm5 */			/* Cu+Du3 */

	vmovapd	%xmm0,%xmm4
	vmulsd	%xmm5,%xmm1,%xmm1			/* u6(Cu+Du3) */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,%xmm3,%xmm2,%xmm2
	VFMA_213SD	(%xmm1,%xmm3,%xmm2)
#else
	vmulsd	%xmm3,%xmm2,%xmm2			/* u3(A+Bu2) */
	vaddsd	%xmm1,%xmm2,%xmm2
#endif
	vsubsd	%xmm6,%xmm2,%xmm2			/* -correction */
	vmulsd	.L__log10_multiplier2(%rip), %xmm4,%xmm4

#ifdef TARGET_FMA
#        VFMADDPD        %xmm4,.L__log10_multiplier(%rip),%xmm2,%xmm2
	VFMA_132PD	(.L__log10_multiplier(%rip),%xmm4,%xmm2)
/*        VFMADDPD        %xmm2,.L__log10_multiplier1(%rip),%xmm0,%xmm0 */	/* This clears out the upper half of xmm0 */
#else
        vmulsd  .L__log10_multiplier(%rip),%xmm2,%xmm2
        vaddsd  %xmm4,%xmm2,%xmm2
#endif
        vmulsd  .L__log10_multiplier1(%rip),%xmm0,%xmm0
        vaddsd  %xmm2,%xmm0,%xmm0

	ret

#define _X0 0
#define _X1 8

#define _R0 32
#define _R1 40

LBL(.L__Scalar_fvdlog10):
        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp
        vmovapd  %xmm0, _X0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_log10_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _R0(%rsp)

        vmovsd   _X1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fsd_log10_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _R1(%rsp)

        vmovapd  _R0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.Lfinishn1_log10)

        ELF_FUNC(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA))



/*============================================================ */

	.text
	ALN_FUNC
        .globl  ENT(ASM_CONCAT(__fsd_log10_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_log10_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(96)(%rsp)
#endif
	/* Get input x into the range [0.5,1) */
	/* compute the index into the log tables */

	vcomisd	.L__real_mindp(%rip), %xmm0
	vmovdqa	%xmm0,%xmm3
	vmovapd	%xmm0,%xmm1
	jb	LBL(.L__z_or_n_dlog10)

	vpsrlq	$52,%xmm3,%xmm3
	vsubsd	.L__real_one(%rip),%xmm1,%xmm1
	vpsubq	.L__mask_1023(%rip),%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm6	/* xexp */

LBL(.L__100_dlog10):
	vmovdqa	%xmm0,%xmm3
	vpand	.L__real_mant(%rip),%xmm3,%xmm3
	xorq	%r8,%r8
	vmovdqa	%xmm3,%xmm4
	vmovsd	.L__real_half(%rip),%xmm5	/* .5 */
	/* Now  x = 2**xexp  * f,  1/2 <= f < 1. */
	vpsrlq	$45,%xmm3,%xmm3
	vmovdqa	%xmm3,%xmm2
	vpsrlq	$1,%xmm3,%xmm3
	vpaddq	.L__mask_040(%rip),%xmm3,%xmm3
	vpand	.L__mask_001(%rip),%xmm2,%xmm2
	vpaddq	%xmm2,%xmm3,%xmm3

	vandpd	.L__real_notsign(%rip),%xmm1,%xmm1
	vcomisd	.L__real_threshold(%rip),%xmm1
	vcvtdq2pd %xmm3,%xmm1
	jb	LBL(.L__near_one_dlog10)
	vmovd	%xmm3,%r8d

	/* reduce and get u */
	vpor	.L__real_half(%rip),%xmm4,%xmm4
	vmovdqa	%xmm4,%xmm2

	leaq	.L__np_ln_lead_table(%rip),%r9

	vmulsd	.L__real_3f80000000000000(%rip),%xmm1,%xmm1	/* f1 = index/128 */
	vsubsd	%xmm1,%xmm2,%xmm2				/* f2 = f - f1 */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,%xmm2,%xmm5,%xmm1
	VFMA_231SD	(%xmm2,%xmm5,%xmm1)
#else
	vmulsd	%xmm2,%xmm5,%xmm5
	vaddsd	%xmm5,%xmm1,%xmm1
#endif

	vdivsd	%xmm1,%xmm2,%xmm2				/* u */

	/* Check for +inf */
	vcomisd	.L__real_inf(%rip),%xmm0
	je	LBL(.L__finish_dlog10)

	vmovsd	-512(%r9,%r8,8),%xmm0 			/* z1 */
	/* solve for ln(1+u) */
	vmovapd	%xmm2,%xmm1				/* u */
	vmulsd	%xmm2,%xmm2,%xmm2				/* u^2 */
	vmovapd	%xmm2,%xmm5
	vmovapd	.L__real_cb3(%rip),%xmm3
	vmulsd	%xmm1,%xmm5,%xmm5				/* u^3 */
#ifdef TARGET_FMA
#	VFMADDSD	.L__real_cb2(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_cb2(%rip),%xmm2,%xmm3)
#else
	vmulsd	%xmm2,%xmm3,%xmm3				/* Cu2 */
	vaddsd	.L__real_cb2(%rip),%xmm3,%xmm3 		/* B+Cu2 */
#endif
	vmovapd	%xmm2,%xmm4
	vmulsd	%xmm5,%xmm4,%xmm4				/* u^5 */
	vmovsd	.L__real_log2_lead(%rip),%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm1,.L__real_cb1(%rip),%xmm5,%xmm1
	VFMA_231SD	(.L__real_cb1(%rip),%xmm5,%xmm1)
#	VFMADDSD	%xmm1,%xmm3,%xmm4,%xmm1
	VFMA_231SD	(%xmm3,%xmm4,%xmm1)
#else
	vmulsd	.L__real_cb1(%rip),%xmm5,%xmm5 		/* Au3 */
	vaddsd	%xmm5,%xmm1,%xmm1				/* u+Au3 */
	vmulsd	%xmm3,%xmm4,%xmm4				/* u5(B+Cu2) */
	vaddsd	%xmm4,%xmm1,%xmm1				/* poly */
#endif
	vmovapd	%xmm0,%xmm3

	/* recombine */
	leaq	.L__np_ln_tail_table(%rip),%rdx
	vaddsd	-512(%rdx,%r8,8),%xmm1,%xmm1 			/* z2	+=q */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,%xmm2,%xmm6,%xmm0
	VFMA_231SD	(%xmm2,%xmm6,%xmm0)
#	VFMADDSD	%xmm3,%xmm2,%xmm6,%xmm2
	VFMA_213SD	(%xmm3,%xmm6,%xmm2)
#	VFMADDSD	%xmm1,.L__real_log2_tail(%rip),%xmm6,%xmm1
	VFMA_231SD	(.L__real_log2_tail(%rip),%xmm6,%xmm1)
#else
	vmulsd	%xmm6,%xmm2,%xmm2				/* npi2 * log2_lead */
	vaddsd	%xmm2,%xmm0,%xmm0				/* r1 */
	vaddsd	%xmm3,%xmm2,%xmm2				/* r1 */
	vmulsd	.L__real_log2_tail(%rip),%xmm6,%xmm6
	vaddsd	%xmm6,%xmm1,%xmm1				/* r2 */
#endif
	vmulsd	.L__log10_multiplier1(%rip), %xmm0,%xmm0
	vmulsd	.L__log10_multiplier2(%rip), %xmm2,%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,.L__log10_multiplier(%rip),%xmm1,%xmm1
	VFMA_132SD	(.L__log10_multiplier(%rip),%xmm2,%xmm1)
#else
	vmulsd	.L__log10_multiplier(%rip), %xmm1,%xmm1
	vaddsd	%xmm2,%xmm1,%xmm1
#endif

LBL(.L__cvt_to_dlog10):
	vaddsd	%xmm1,%xmm0,%xmm0

LBL(.L__finish_dlog10):
#if defined(_WIN64)
	vmovdqu	RZ_OFF(96)(%rsp), %ymm6
#endif

	RZ_POP
	rep
	ret

	ALN_QUAD
LBL(.L__near_one_dlog10):
	/* saves 10 cycles */
	/* r = x - 1.0; */
	vmovsd	.L__real_two(%rip),%xmm2
	vsubsd	.L__real_one(%rip),%xmm0,%xmm0

	/* u = r / (2.0 + r); */
	vaddsd	%xmm0,%xmm2,%xmm2
	vmovapd	%xmm0,%xmm1
	vdivsd	%xmm2,%xmm1,%xmm1
	vmovsd	.L__real_ca4(%rip),%xmm4
	vmovsd	.L__real_ca3(%rip),%xmm5
	/* correction = r * u; */
	vmovapd	%xmm0,%xmm6
	vmulsd	%xmm1,%xmm6,%xmm6

	/* u = u + u; */
	vaddsd	%xmm1,%xmm1,%xmm1
	vmovapd	%xmm1,%xmm2
	vmulsd	%xmm2,%xmm2,%xmm2
	/* r2 = (u * v * (ca_1 + v * (ca_2 + v * (ca_3 + v * ca_4))) - correction); */
	vmulsd	%xmm1,%xmm5,%xmm5
	vmovapd	%xmm1,%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3
	vmulsd	.L__real_ca2(%rip),%xmm2,%xmm2

	vaddsd	.L__real_ca1(%rip),%xmm2,%xmm2
	vmovapd	%xmm3,%xmm1
	vmulsd	%xmm1,%xmm1,%xmm1
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm4,%xmm3,%xmm5
	VFMA_231SD	(%xmm4,%xmm3,%xmm5)
#else
	vmulsd	%xmm3,%xmm4,%xmm4
	vaddsd	%xmm4,%xmm5,%xmm5
#endif

	vmovapd	%xmm0,%xmm4
	vmulsd	%xmm3,%xmm2,%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,%xmm1,%xmm5,%xmm1
	VFMA_213SD	(%xmm2,%xmm5,%xmm1)
#else
	vmulsd	%xmm5,%xmm1,%xmm1
	vaddsd	%xmm2,%xmm1,%xmm1
#endif
	vsubsd	%xmm6,%xmm1,%xmm1
	vmulsd	.L__log10_multiplier1(%rip), %xmm0,%xmm0
	vmulsd	.L__log10_multiplier2(%rip), %xmm4,%xmm4
#ifdef TARGET_FMA
#	VFMADDSD	%xmm4,.L__log10_multiplier(%rip),%xmm1,%xmm1
	VFMA_132SD	(.L__log10_multiplier(%rip),%xmm4,%xmm1)
#else
	vmulsd	.L__log10_multiplier(%rip), %xmm1,%xmm1
	vaddsd	%xmm4,%xmm1,%xmm1
#endif
	jmp	LBL(.L__cvt_to_dlog10)

	/* Start here for all the conditional cases */
	/* we have a zero, a negative number, denorm, or nan. */
LBL(.L__z_or_n_dlog10):
	jp	LBL(.L__lnan_dlog10)
	vxorpd	%xmm1, %xmm1, %xmm1
	vcomisd	%xmm1, %xmm0
	je	LBL(.L__zero_dlog10)
	jbe	LBL(.L__negative_x_dlog10)

	/* A Denormal input, scale appropriately */
	vmulsd	.L__real_scale(%rip), %xmm0, %xmm0
	vmovdqa	%xmm0, %xmm3
	vmovapd	%xmm0, %xmm1

	vpsrlq	$52,%xmm3,%xmm3
	vsubsd	.L__real_one(%rip),%xmm1,%xmm1
	vpsubq	.L__mask_1075(%rip),%xmm3,%xmm3
	vcvtdq2pd %xmm3,%xmm6
	jmp	LBL(.L__100_dlog10)

	/* x == +/-0.0 */
LBL(.L__zero_dlog10):
#ifdef FMATH_EXCEPTIONS
        vmovsd  .L__real_one(%rip), %xmm1
        vdivsd  %xmm0, %xmm1, %xmm0 /* Generate divide-by-zero op */
#endif
	vmovsd	.L__real_ninf(%rip),%xmm0  /* C99 specs -inf for +-0 */
	jmp	LBL(.L__finish_dlog10)

	/* x < 0.0 */
LBL(.L__negative_x_dlog10):
#ifdef FMATH_EXCEPTIONS
	vsqrtsd	%xmm0, %xmm0, %xmm0
#endif
	vmovsd	.L__real_nan(%rip),%xmm0
	jmp	LBL(.L__finish_dlog10)

	/* NaN */
LBL(.L__lnan_dlog10):
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	.L__real_qnanbit(%rip), %xmm1	/* convert to quiet */
	vorpd	%xmm1, %xmm0, %xmm0
	jmp	LBL(.L__finish_dlog10)

        ELF_FUNC(ASM_CONCAT(__fsd_log10_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_log10_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(72)(%rsp)
	vmovdqu	%ymm7, RZ_OFF(104)(%rsp)
#endif

/* Fast vector natural logarithm code goes here... */
        /* First check for valid input:
         * if (a .gt. 0.0) then */
	vmovaps  .L4_384(%rip), %xmm4	/* Move min arg to xmm4 */
	vxorps	%xmm7, %xmm7, %xmm7		/* Still need 0.0 */
	vmovaps	%xmm0, %xmm2		/* Move for nx */
	vmovaps	%xmm0, %xmm1		/* Move to xmm1 for later ma */

	/* Check exceptions and valid range */
	vcmpleps	%xmm0, %xmm4, %xmm4		/* '00800000'x <= a, xmm4 1 where true */
	vcmpltps	%xmm0, %xmm7, %xmm7		/* Test for 0.0 < a, xmm7 1 where true */
	vcmpneqps	.L4_387(%rip), %xmm0, %xmm0	/* Test for == +inf */
	vxorps		%xmm7, %xmm4, %xmm4		/* xor to find just denormal inputs */
	vmovmskps	%xmm4, %eax		/* Move denormal mask to gp ref */
	vmovaps		%xmm2, RZ_OFF(24)(%rsp)	/* Move for exception processing */
	vmovaps		.L4_382(%rip), %xmm3	/* Move 126 */
	cmp		$0, %eax		/* Test for denormals */
	jne		LBL(.LB_DENORMs_log10)

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
	vandps	.L4_380(%rip), %xmm1, %xmm1	/* ma = IAND(ia,'007fffff'x) */
	vpsrld	$23, %xmm2, %xmm2		/* nx = ISHFT(ia,-23) */
	vandps	%xmm0, %xmm7, %xmm7		/* Mask for nan, inf, neg and 0.0 */
	vmovaps	%xmm1, %xmm6		/* move ma for ig */
	vpsubd	.L4_381(%rip), %xmm1, %xmm1	/* ms = ma - '3504f3'x */
	vpsubd	%xmm3, %xmm2, %xmm2		/* nx = ISHFT(ia,-23) - 126 */
	vorps	.L4_383(%rip), %xmm6, %xmm6	/* ig = IOR(ma,'3f000000'x) */
	vmovaps	%xmm1, %xmm0		/* move ms for tbl ms */
	vandps	.L4_384(%rip), %xmm1, %xmm1	/* mx = IAND(ms,'00800000'x) */
	vandps	.L4_385(%rip), %xmm0, %xmm0	/* ms = IAND(ms,'007f0000'x) */
	vorps	%xmm1, %xmm6, %xmm6		/* ig = IOR(ig, mx) */
	vpsrad	$23, %xmm1, %xmm1		/* ISHFT(mx,-23) */
	vpsrad	$12, %xmm0, %xmm0		/* ISHFT(ms,-12) for 128 bit reads */
	vmovmskps %xmm7, %eax		/* Move xmm7 mask to eax */
	vpsubd	%xmm1, %xmm2, %xmm2		/* nx = nx - ISHFT(mx,-23) */
	vmovaps	%xmm0, RZ_OFF(40)(%rsp)	/* Move to memory */
	vcvtdq2ps  %xmm2, %xmm0		/* xn = real(nx) */

	movl	RZ_OFF(40)(%rsp), %ecx		/* Move to gp register */
	vmovaps	(%r8,%rcx,1), %xmm1		/* Read from 1st table location */
	movl	RZ_OFF(36)(%rsp), %edx		/* Move to gp register */
	vmovaps	(%r8,%rdx,1), %xmm2		/* Read from 2nd table location */
	movl	RZ_OFF(32)(%rsp), %ecx		/* Move to gp register */
	vmovaps	(%r8,%rcx,1), %xmm3		/* Read from 3rd table location */
	movl	RZ_OFF(28)(%rsp), %edx		/* Move to gp register */
	vmovaps	(%r8,%rdx,1), %xmm4		/* Read from 4th table location */

	/* So, we do 4 reads of a,b,c into registers xmm1, xmm2, xmm3, xmm4
	 * Assume we need to keep rg in xmm6, xn in xmm0
	 * The following shuffle gets them into SIMD mpy form:
	 */

	vsubps	.L4_386(%rip), %xmm6, %xmm6 	/* x0 = rg - 1.0 */

	vmovaps	%xmm1, %xmm5		/* Store 1/3, c0, b0, a0 */
	vmovaps	%xmm3, %xmm7		/* Store 1/3, c2, b2, a2 */

	vunpcklps %xmm2, %xmm1, %xmm1		/* b1, b0, a1, a0 */
	vunpcklps %xmm4, %xmm3, %xmm3		/* b3, b2, a3, a2 */
	vunpckhps %xmm2, %xmm5, %xmm5		/* 1/3, 1/3, c1, c0 */
	vunpckhps %xmm4, %xmm7, %xmm7		/* 1/3, 1/3, c3, c2 */

	vmovaps		%xmm6, %xmm4		/* move x0 */

	vmovaps		%xmm1, %xmm2		/* Store b1, b0, a1, a0 */
	vmovlhps	%xmm3, %xmm1, %xmm1		/* a3, a2, a1, a0 */
	vmovlhps	%xmm7, %xmm5, %xmm5		/* c3, c2, c1, c0 */
	vmovhlps	%xmm2, %xmm3, %xmm3		/* b3, b2, b1, b0 */

/* Causing inconsistent results between vector and scalar versions (FS#21062) */
/* #ifdef TARGET_FMA
#	VFMADDPS	%xmm3,%xmm6, %xmm1, %xmm1
	VFMA_213PS	(%xmm3,%xmm6,%xmm1)
#else */
	vmulps		%xmm6, %xmm1, %xmm1		/* COEFFS(mt) * x0 */
	vaddps		%xmm3, %xmm1, %xmm1		/* COEFFS(mt) * g + COEFFS(mt+1) */
/* #endif */
	vmulps		%xmm6, %xmm6, %xmm6		/* xsq = x0 * x0 */
	vmovhlps	%xmm7, %xmm7, %xmm7		/* 1/3, 1/3, 1/3, 1/3 */

	vmovaps		%xmm4, %xmm2		/* move x0 */

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

	vmulps	%xmm6, %xmm4, %xmm4		/* xcu = xsq * x0 */
	vmulps	.L4_383(%rip), %xmm6, %xmm6	/* x1 = 0.5 * xsq */
	vmovaps	%xmm6, %xmm3		/* move x1 */

#ifdef TARGET_FMA
#	VFMADDPS	%xmm5,%xmm2, %xmm1, %xmm1
	VFMA_213PS	(%xmm5,%xmm2,%xmm1)
#	VFNMADDPS	%xmm1, %xmm6, %xmm6, %xmm1
	VFNMA_231PS	(%xmm6,%xmm6,%xmm1)
#else
	vmulps	%xmm2, %xmm1, %xmm1		/* * x0 */
	vaddps	%xmm5, %xmm1, %xmm1		/* + COEFFS(mt+2) = rp */
	vmulps	%xmm6, %xmm6, %xmm6		/* x3 = x1 * x1 */
	vsubps	%xmm6, %xmm1, %xmm1		/* rp - x3 */
#endif
	vmulps	%xmm7, %xmm4, %xmm4		/* x2 = thrd * xcu */
	vmovaps	.L4_396(%rip), %xmm7	/* Move c1 */
        vmovaps  .L4_397(%rip), %xmm6	/* Move c2 */
	vaddps	%xmm1, %xmm4, %xmm4		/* rp - x3 + x2 */
	vsubps	%xmm3, %xmm4, %xmm4		/* rp - x3 + x2 - x1 */
	vmulps	.L4_395(%rip),%xmm2, %xmm2     /* mpy for log10 */

#ifdef TARGET_FMA
#	VFMADDPS	%xmm2,.L4_395(%rip),%xmm4,%xmm4
	VFMA_132PS	(.L4_395(%rip),%xmm2,%xmm4)
#else
	vmulps	.L4_395(%rip),%xmm4, %xmm4     /* mpy for log10 */
	vaddps	%xmm2, %xmm4, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
#endif

#ifdef TARGET_FMA
#	VFMADDPS	%xmm4,%xmm0,%xmm7,%xmm4	/* We can do this because xmm7 is set to be 0 later */
	VFMA_231PS	(%xmm0,%xmm7,%xmm4)
#	VFMADDPS	%xmm4,%xmm6,%xmm0,%xmm0
	VFMA_213PS	(%xmm4,%xmm6,%xmm0)
#else
	vmulps   %xmm0, %xmm7, %xmm7		/* xn * c1 */
	vaddps   %xmm7, %xmm4, %xmm4		/* (xn * c1 + rz) */
        vmulps   %xmm6, %xmm0, %xmm0		/* xn * c2 */
        vaddps   %xmm4, %xmm0, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */
#endif

	/* Compare exception mask now and jump if no exceptions */
	cmp	$15, %eax
	jne 	LBL(.LB_EXCEPTs_log10)

LBL(.LB_900_log10):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(72)(%rsp), %ymm6
	vmovdqu	RZ_OFF(104)(%rsp), %ymm7
#endif

	RZ_POP
	rep
	ret

LBL(.LB_EXCEPTs_log10):
        /* Handle all exceptions by masking in xmm */
        vmovaps  RZ_OFF(24)(%rsp), %xmm1	/* original input */
        vmovaps  RZ_OFF(24)(%rsp), %xmm2	/* original input */
        vmovaps  RZ_OFF(24)(%rsp), %xmm3	/* original input */
        vxorps   %xmm7, %xmm7, %xmm7            /* xmm7 = 0.0 */
        vxorps   %xmm6, %xmm6, %xmm6            /* xmm6 = 0.0 */
	vmovaps	.L4_394(%rip), %xmm5	/* convert nan bit */
        vxorps   %xmm4, %xmm4, %xmm4            /* xmm4 = 0.0 */

        vcmpunordps	%xmm1, %xmm7, %xmm7         /* Test if unordered */
        vcmpltps	%xmm6, %xmm2, %xmm2            /* Test if a < 0.0 */
        vcmpordps	%xmm1, %xmm6, %xmm6           /* Test if ordered */

        vandps   %xmm7, %xmm5, %xmm5            /* And nan bit where unordered */
        vorps    %xmm7, %xmm4, %xmm4            /* Or all masks together */
        vandps   %xmm1, %xmm7, %xmm7            /* And input where unordered */
	vorps	%xmm5, %xmm7, %xmm7		/* Convert unordered nans */

        vxorps   %xmm5, %xmm5, %xmm5            /* xmm5 = 0.0 */
        vandps   %xmm2, %xmm6, %xmm6            /* Must be ordered and < 0.0 */
        vorps    %xmm6, %xmm4, %xmm4            /* Or all masks together */
        vandps   .L4_390(%rip), %xmm6, %xmm6    /* And -nan if < 0.0 and ordered */

        vcmpeqps	.L4_387(%rip), %xmm3, %xmm3    /* Test if equal to infinity */
        vcmpeqps	%xmm5, %xmm1, %xmm1            /* Test if eq 0.0 */
        vorps		%xmm6, %xmm7, %xmm7            /* or in < 0.0 */

        vorps    %xmm3, %xmm4, %xmm4            /* Or all masks together */
        vandps   .L4_387(%rip), %xmm3, %xmm3    /* inf and inf mask */
        vmovaps  %xmm0, %xmm2
        vorps    %xmm3, %xmm7, %xmm7            /* or in infinity */

        vorps    %xmm1, %xmm4, %xmm4            /* Or all masks together */
        vandps   .L4_391(%rip), %xmm1, %xmm1    /* And -inf if == 0.0 */
        vmovaps  %xmm4, %xmm0
        vorps    %xmm1, %xmm7, %xmm7            /* or in -infinity */

        vandnps  %xmm2, %xmm0, %xmm0            /* Where mask not set, use result */
        vorps    %xmm7, %xmm0, %xmm0            /* or in exceptional values */
	jmp	LBL(.LB_900_log10)

LBL(.LB_DENORMs_log10):
	/* Have the denorm mask in xmm4, so use it to scale a and the subtractor */
	vmovaps	%xmm4, %xmm5		/* Move mask */
	vmovaps	%xmm4, %xmm6		/* Move mask */
	vandps	.L4_392(%rip), %xmm4, %xmm4	/* Have 2**23 where denorms are, 0 else */
	vandnps	%xmm1, %xmm5, %xmm5		/* Have a where denormals aren't */
	vmulps	%xmm4, %xmm1, %xmm1		/* denormals * 2**23 */
	vandps	.L4_393(%rip), %xmm6, %xmm6	/* have 23 where denorms are, 0 else */
	vorps	%xmm5, %xmm1, %xmm1		/* Or in the original a */
	vpaddd	%xmm6, %xmm3, %xmm3		/* Add 23 to 126 for offseting exponent */
	vmovaps	%xmm1, %xmm2		/* Move to the next location */
	jmp	LBL(.LB_100_log10)

	ELF_FUNC(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA))


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
 *  float __fss_log(float f)
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
	.globl	ENT(ASM_CONCAT(__fss_log10_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_log10_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(96)(%rsp)
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

	vmovss	%xmm0, RZ_OFF(4)(%rsp)
        vmovss	.L4_384(%rip), %xmm2	/* Move smallest normalized number */
	movl	RZ_OFF(4)(%rsp), %ecx
	andl	$8388607, %ecx		/* ma = IAND(ia,'007fffff'x) */
	leaq 	-3474675(%rcx), %rdx	/* ms = ma - '3504f3'x */
	orl	$1056964608, %ecx	/* ig = IOR(ma,'3f000000'x) */
	vcmpnless	%xmm0, %xmm2, %xmm2		/* '00800000'x <= a, xmm2 1 where not */
        vcmpeqss	.L4_387(%rip), %xmm0,%xmm0	/* Test for == +inf */
	movl	%edx, %eax		/* move ms */
	andl	$8388608, %edx		/* mx = IAND(ms,'00800000'x) */
	orl	%edx, %ecx		/* ig = IOR(ig,mx) */
	movl	%ecx, RZ_OFF(8)(%rsp)	/* move back over to fp sse */
	shrl	$23, %edx		/* ISHFT(mx,-23) */
        vunpcklps %xmm2, %xmm0, %xmm0		/* Mask for nan, inf, neg and 0.0 */

	leaq	.L_STATICS1(%rip), %r8
	movl	RZ_OFF(4)(%rsp), %ecx	/* ia */
	andl	$8323072, %eax		/* ms = IAND(ms,'007f0000'x) */
	vmovss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	vmovmskps %xmm0, %r9d		/* move exception mask to gp reg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$126, %ecx		/* nx = ISHFT(ia,-23) - 126 */
	vmovss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
        shrl    $14, %eax		/* mt1 */
	and	$3, %r9d		/* mask with 3 */
	vmovss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
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

	vmovd %ecx, %xmm0
	vcvtdq2ps %xmm0, %xmm0
	vsubss	.L4_386(%rip), %xmm1, %xmm1	/* x0 = rg - 1.0 */
	vsubss	.L4_386(%rip), %xmm6, %xmm6	/* x0 = rg - 1.0 */
	vsubss	.L4_386(%rip), %xmm4, %xmm4	/* x0 = rg - 1.0 */
	vsubss	.L4_386(%rip), %xmm2, %xmm2	/* x0 = rg - 1.0 */
	vmulss	(%r8,%rax,4), %xmm1, %xmm1	/* COEFFS(mt) * x0 */
	vmulss   %xmm6, %xmm6, %xmm6		/* xsq = x0 * x0 */
	vaddss	4(%r8,%rax,4), %xmm1, %xmm1	/* COEFFS(mt) * x0 + COEFFS(mt+1) */
	vmulss   %xmm6, %xmm4, %xmm4		/* xcu = xsq * x0 */
	vmulss   .L4_383(%rip), %xmm6, %xmm6	/* x1 = 0.5 * xsq */
	vmulss	12(%r8,%rax,4), %xmm4, %xmm4	/* x2 = thrd * xcu */
	vmovaps	%xmm6, %xmm3		/* move x1 */
#ifdef TARGET_FMA
#	VFMADDSS	8(%r8,%rax,4),%xmm1,%xmm2,%xmm1
	VFMA_213SS	(8(%r8,%rax,4),%xmm2,%xmm1)
#	VFNMADDSS	%xmm1,%xmm6,%xmm6,%xmm1
	VFNMA_231SS	(%xmm6,%xmm6,%xmm1)
#else
	vmulss   %xmm2, %xmm1, %xmm1		/* * x0 */
	vaddss	8(%r8,%rax,4), %xmm1, %xmm1	/* + COEFFS(mt+2) = rp */
	vmulss	%xmm6, %xmm6, %xmm6		/* x3 = x1 * x1 */
	vsubss	%xmm6, %xmm1, %xmm1		/* rp - x3 */
#endif
	vmovss	.L4_396(%rip), %xmm5		/* Move c1 */
        vmovss	.L4_397(%rip), %xmm6		/* Move c2 */
	vaddss	%xmm1, %xmm4, %xmm4		/* rp - x3 + x2 */
	vsubss	%xmm3, %xmm4, %xmm4		/* rp - x3 + x2 - x1 */
	vmulss	.L4_395(%rip), %xmm2, %xmm2

#ifdef TARGET_FMA
#	VFMADDSS	%xmm2,.L4_395(%rip),%xmm4,%xmm4
	VFMA_132SS	(.L4_395(%rip),%xmm2,%xmm4)
#	VFMADDSS	%xmm4,%xmm0,%xmm5,%xmm4
	VFMA_231SS	(%xmm0,%xmm5,%xmm4)
#	VFMADDSS	%xmm4,%xmm0,%xmm6,%xmm0
	VFMA_213SS	(%xmm4,%xmm6,%xmm0)
#else
	vmulss	.L4_395(%rip), %xmm4, %xmm4
	vaddss	%xmm2, %xmm4, %xmm4		/* rp - x3 + x2 - x1 + x0 = rz */
	vmulss   %xmm0, %xmm5, %xmm5		/* xn * c1 */
	vaddss   %xmm5, %xmm4, %xmm4		/* (xn * c1 + rz) */
        vmulss   %xmm6, %xmm0, %xmm0		/* xn * c2 */
        vaddss   %xmm4, %xmm0, %xmm0		/* rr = (xn * c1 + rz) + xn * c2 */
#endif

LBL(.LB1_900_log10):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(96)(%rsp), %ymm6
#endif
	RZ_POP
	rep
	ret

	ALN_WORD
LBL(.LB1_800_log10):
	/* ir = 'ff800000'x */
	xorq	%rax,%rax
	vmovss	RZ_OFF(4)(%rsp), %xmm0
	vmovd 	%rax, %xmm1
	vcomiss	%xmm1, %xmm0
	jp	LBL(.LB1_cvt_nan)
#ifdef FMATH_EXCEPTIONS
	vmovss  .L4_386(%rip), %xmm1
        vdivss  %xmm0, %xmm1, %xmm0     /* Generate div-by-zero op when x=0 */
#endif
	vmovss	.L4_391(%rip),%xmm0	/* Move -inf */
	je	LBL(.LB1_900)
#ifdef FMATH_EXCEPTIONS
	vsqrtss	%xmm0, %xmm0, %xmm0	/* Generate invalid op for x<0 */
#endif
	vmovss	.L4_390(%rip),%xmm0	/* Move -nan */
	jb	LBL(.LB1_900)
	vmovss	.L4_387(%rip), %xmm0	/* Move +inf */
	vmovss	RZ_OFF(4)(%rsp), %xmm1
	vcomiss	%xmm1, %xmm0
	je	LBL(.LB1_900_log10)

	/* Otherwise, we had a denormal as an input */
	vmulss	.L4_392(%rip), %xmm1, %xmm1	/* a * scale factor */
	vmovss	%xmm1, RZ_OFF(4)(%rsp)
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
	vmovss	RZ_OFF(8)(%rsp), %xmm1	/* rg */
	shrl	$23, %ecx		/* ISHFT(ia,-23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm6	/* rg */
	subl	$149, %ecx		/* nx = ISHFT(ia,-23) - (126 + 23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm4	/* rg */
	subl	%edx, %ecx		/* nx = nx - ISHFT(mx,-23) */
	vmovss	RZ_OFF(8)(%rsp), %xmm2	/* rg */
        shrl    $14, %eax		/* mt1 */
	jmp	LBL(.LB1_100_log10)

LBL(.LB1_cvt_nan_log10):
	vmovss	.L4_394(%rip), %xmm1	/* nan bit */
	vorps	%xmm1, %xmm0, %xmm0
	jmp	LBL(.LB1_900_log10)

	ELF_FUNC(ASM_CONCAT(__fss_log10_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_log10_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm2
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3

	/* Set n = nearest integer to r */
	vcomiss	.L_sp_sinh_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_inf)
	vcomiss	.L_sp_sinh_min_singleval(%rip), %xmm0
	jb	LBL(.L_sp_cosh_ninf)

	vcvtpd2dq %xmm3,%xmm4	/* convert to integer */
	vcvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm2,.L__real_log2_by_32(%rip),%xmm1,%xmm2
	VFNMA_231SD	(.L__real_log2_by_32(%rip),%xmm1,%xmm2)
#else
	vmulsd	.L__real_log2_by_32(%rip),%xmm1,%xmm1
	vsubsd	%xmm1,%xmm2,%xmm2	/* r1 in xmm2, */
#endif
	vmovd	%xmm4,%ecx
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
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm1
	vmovsd	.L__real_3fe0000000000000(%rip),%xmm5
	sub	%r8d,%r9d
	sar	$5,%r9d

	vmovapd	%xmm2,%xmm0
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm5,%xmm1,%xmm2,%xmm5
	VFNMA_231SD	(%xmm1,%xmm2,%xmm5)
#	VFMADDSD	.L__real_3fe0000000000000(%rip),%xmm1,%xmm2,%xmm1
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm1)
	vmulsd		%xmm2,%xmm2,%xmm2
#	VFMSUBSD	%xmm0,%xmm2,%xmm5,%xmm5
	VFMS_213SD	(%xmm0,%xmm2,%xmm5)
#	VFMADDSD	%xmm0,%xmm1,%xmm2,%xmm2
	VFMA_213SD	(%xmm0,%xmm1,%xmm2)
#else
	vmulsd	%xmm2,%xmm1,%xmm1
	vsubsd	%xmm1,%xmm5,%xmm5                             /* exp(-x) */
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm1,%xmm1   /* exp(x) */

	vmulsd	%xmm2,%xmm2,%xmm2

	vmulsd	%xmm2,%xmm5,%xmm5        /* exp(-x) */
	vsubsd	%xmm0,%xmm5,%xmm5        /* exp(-x) */
	vmulsd	%xmm1,%xmm2,%xmm2        /* exp(x) */
	vaddsd	%xmm0,%xmm2,%xmm2        /* exp(x) */
#endif

	vmovsd	(%rdx,%r8,8),%xmm3   /* exp(-x) */
	vmovsd	(%rdx,%rax,8),%xmm4   /* exp(x) */

	/* *z2 = f2 + ((f1 + f2) * q); */
        add	$1022, %ecx	/* add bias */
        add	$1022, %r9d	/* add bias */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm3,%xmm3,%xmm5,%xmm5
	VFMA_213SD	(%xmm3,%xmm3,%xmm5)
#	VFMADDSD	%xmm4,%xmm4,%xmm2,%xmm2
	VFMA_213SD	(%xmm4,%xmm4,%xmm2)
#else
	vmulsd	%xmm3,%xmm5,%xmm5
	vaddsd	%xmm3,%xmm5,%xmm5  /* z = z1 + z2   done with 1,2,3,4,5 */
	vmulsd	%xmm4,%xmm2,%xmm2
	vaddsd	%xmm4,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

	shlq	$52,%rcx        /* build 2^n */
        shlq	$52,%r9         /* build 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(16)(%rsp),%xmm5,%xmm5	/* result *= 2^n */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,RZ_OFF(24)(%rsp),%xmm2,%xmm2
	VFMA_132SD	(RZ_OFF(24)(%rsp),%xmm5,%xmm2)
#else
	vmulsd	RZ_OFF(24)(%rsp),%xmm2,%xmm2	/* result *= 2^n */
	vaddsd	%xmm5,%xmm2,%xmm2		/* result = exp(x) + exp(-x) */
#endif
	vunpcklpd %xmm2, %xmm2,%xmm2
	vcvtpd2ps %xmm2, %xmm0

	RZ_POP
	rep
	ret

	ELF_FUNC(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        vmovd    %xmm0, %eax
        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vunpcklps %xmm0, %xmm0, %xmm0
	vcvtps2pd %xmm0, %xmm2

	shrl	$23, %eax
	andl	$0xff, %eax
	cmpl	$122, %eax
	jb	LBL(.L__fss_sinh_shortcuts)

	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm2,%xmm3,%xmm3

	/* Set n = nearest integer to r */
	vcomiss	.L_sp_sinh_max_singleval(%rip), %xmm0
	ja	LBL(.L_sp_inf)
	vcomiss	.L_sp_sinh_min_singleval(%rip), %xmm0
	jb	LBL(.L_sp_sinh_ninf)

	vcvtpd2dq %xmm3,%xmm4	/* convert to integer */
	vcvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm2,.L__real_log2_by_32(%rip),%xmm1,%xmm2
	VFNMA_231SD	(.L__real_log2_by_32(%rip),%xmm1,%xmm2)
#else
	vmulsd	.L__real_log2_by_32(%rip),%xmm1,%xmm1
	vsubsd	%xmm1,%xmm2,%xmm2	/* r1 in xmm2, */
#endif
	vmovd	%xmm4,%ecx
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
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm1
	vmovsd	.L__real_3fe0000000000000(%rip),%xmm5
	sub	%r8d,%r9d
	sar	$5,%r9d

	vmovapd	%xmm2,%xmm0
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm5,%xmm1,%xmm2,%xmm5
	VFNMA_231SD	(%xmm1,%xmm2,%xmm5)
#	VFMADDSD	.L__real_3fe0000000000000(%rip),%xmm2,%xmm1,%xmm1
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm1)
	vmulsd	%xmm2,%xmm2,%xmm2
#	VFMSUBSD	%xmm0,%xmm2,%xmm5,%xmm5
	VFMS_213SD	(%xmm0,%xmm2,%xmm5)
#	VFMADDSD	%xmm0,%xmm1,%xmm2,%xmm2
	VFMA_213SD	(%xmm0,%xmm1,%xmm2)
#else
	vmulsd	%xmm2,%xmm1,%xmm1
	vsubsd	%xmm1,%xmm5,%xmm5                             /* exp(-x) */
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm1,%xmm1   /* exp(x) */

	vmulsd	%xmm2,%xmm2,%xmm2

	vmulsd	%xmm2,%xmm5,%xmm5        /* exp(-x) */
	vsubsd	%xmm0,%xmm5,%xmm5        /* exp(-x) */
	vmulsd	%xmm1,%xmm2,%xmm2        /* exp(x) */
	vaddsd	%xmm0,%xmm2,%xmm2        /* exp(x) */
#endif
	vmovsd	(%rdx,%r8,8),%xmm3   /* exp(-x) */
	vmovsd	(%rdx,%rax,8),%xmm4   /* exp(x) */

	/* *z2 = f2 + ((f1 + f2) * q); */
        add	$1022, %ecx	/* add bias */
        add	$1022, %r9d	/* add bias */

#ifdef TARGET_FMA
#	VFMADDSD	%xmm3,%xmm3,%xmm5,%xmm5
	VFMA_213SD	(%xmm3,%xmm3,%xmm5)
#	VFMADDSD	%xmm4,%xmm4,%xmm2,%xmm2
	VFMA_213SD	(%xmm4,%xmm4,%xmm2)
#else
	vmulsd	%xmm3,%xmm5,%xmm5
	vmulsd	%xmm4,%xmm2,%xmm2
	vaddsd	%xmm3,%xmm5,%xmm5  /* z = z1 + z2   done with 1,2,3,4,5 */
	vaddsd	%xmm4,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif
        shlq	$52,%rcx        /* build 2^n */
        shlq	$52,%r9         /* build 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%r9,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(16)(%rsp),%xmm5,%xmm5	/* result *= 2^n */
#ifdef TARGET_FMA
#	VFMSUBSD	%xmm5,RZ_OFF(24)(%rsp),%xmm2,%xmm2
	VFMS_132SD	(RZ_OFF(24)(%rsp),%xmm5,%xmm2)
#else
	vmulsd	RZ_OFF(24)(%rsp),%xmm2,%xmm2	/* result *= 2^n */
	vsubsd	%xmm5,%xmm2,%xmm2             /* result = exp(x) - exp(-x) */
#endif

LBL(.L__fss_sinh_done):
	vunpcklpd %xmm2, %xmm2,%xmm2
	vcvtpd2ps %xmm2, %xmm0

	RZ_POP
	rep
	ret

LBL(.L__fss_sinh_shortcuts):
	vmovapd	%xmm2, %xmm1
	vmovapd	%xmm2, %xmm0
	vmulsd	%xmm2, %xmm2,%xmm2
	vmulsd 	%xmm1, %xmm1,%xmm1
	vmulsd	.L__dsinh_shortval_y4(%rip), %xmm2,%xmm2
	vaddsd	.L__dsinh_shortval_y3(%rip), %xmm2,%xmm2
#ifdef TARGET_FMA
#	VFMADDSD	.L__dsinh_shortval_y2(%rip),%xmm1,%xmm2,%xmm2
	VFMA_213SD	(.L__dsinh_shortval_y2(%rip),%xmm1,%xmm2)
#	VFMADDSD        .L__dsinh_shortval_y1(%rip),%xmm1,%xmm2,%xmm2
	VFMA_213SD	(.L__dsinh_shortval_y1(%rip),%xmm1,%xmm2)
	vmulsd		%xmm0, %xmm1,%xmm1
#	VFMADDSD        %xmm0,%xmm1,%xmm2,%xmm2
	VFMA_213SD	(%xmm0,%xmm1,%xmm2)
#else
	vmulsd	%xmm1, %xmm2,%xmm2
	vaddsd	.L__dsinh_shortval_y2(%rip), %xmm2,%xmm2
	vmulsd	%xmm1, %xmm2,%xmm2
	vaddsd	.L__dsinh_shortval_y1(%rip), %xmm2,%xmm2
	vmulsd	%xmm0, %xmm1,%xmm1
	vmulsd	%xmm1, %xmm2,%xmm2
	vaddsd	%xmm0, %xmm2,%xmm2
#endif
	jmp	LBL(.L__fss_sinh_done)

	ELF_FUNC(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fsd_cosh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_cosh_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(96)(%rsp)
#endif

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm0,%xmm3,%xmm3

	/* Set n = nearest integer to r */
	vcomisd	.L_sinh_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	vcomisd	.L_sinh_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_cosh_ninf)
	vcvtpd2dq %xmm3,%xmm4	/* convert to integer */
	vcvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovsd	.L__real_log2_by_32_lead(%rip),%xmm2

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm2,%xmm2
	VFNMA_213SD	(%xmm0,%xmm1,%xmm2)
#else
	vmulsd	%xmm1,%xmm2,%xmm2
	vsubsd	%xmm2,%xmm0,%xmm2	/* r1 in xmm0, */
#endif
	vmovd	%xmm4,%ecx
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
#ifdef TARGET_FMA
#	VFMADDSD        %xmm2,.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2
	VFMA_231SD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2)
#else
	vmulsd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1
	vaddsd	%xmm1,%xmm2,%xmm2    /* r = r1 + r2 */
#endif

	/* j = n & 0x0000001f; */
	movq	%r8,%rax
	andl	%ecx,%eax
	subl	%ecx,%r9d
	andl	%r9d,%r8d


	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%eax,%ecx
	sarl	$5,%ecx
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
	vmovapd	%xmm2,%xmm1
	vmovsd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm0
	vmovsd	.L__real_3F811115B7AA905E(%rip),%xmm5
	vmovsd	.L__real_3fe0000000000000(%rip),%xmm6

	vmulsd	%xmm2,%xmm1,%xmm1
	vmovapd	%xmm1,%xmm4

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm5,%xmm2,%xmm3,%xmm5
	VFNMA_231SD	(%xmm2,%xmm3,%xmm5)
#	VFMADDSD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFNMADDSD	%xmm6,%xmm0,%xmm2,%xmm6
	VFNMA_231SD	(%xmm0,%xmm2,%xmm6)
#	VFMADDSD	.L__real_3fe0000000000000(%rip),%xmm2,%xmm0,%xmm0
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
#else
	vmulsd	%xmm2,%xmm3,%xmm3
	vsubsd	%xmm3,%xmm5,%xmm5
	vaddsd	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3
	vmulsd	%xmm2,%xmm0,%xmm0
	vsubsd	%xmm0,%xmm6,%xmm6
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm0,%xmm0
#endif

	vmulsd	%xmm1,%xmm4,%xmm4

#ifdef TARGET_FMA
#	VFMSUBSD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5,%xmm5
	VFMS_213SD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5)
#	VFMADDSD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMSUBSD	%xmm2,%xmm1,%xmm6,%xmm6
	VFMS_213SD	(%xmm2,%xmm1,%xmm6)
#        VFMADDSD        %xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#        VFNMADDSD       %xmm6,%xmm4,%xmm5,%xmm6
	VFNMA_231SD	(%xmm4,%xmm5,%xmm6)
#	VFMADDSD	%xmm0,%xmm4,%xmm3,%xmm0
	VFMA_231SD	(%xmm4,%xmm3,%xmm0)
#else
	vmulsd	%xmm2,%xmm5,%xmm5
	vsubsd	.L__real_3FA5555555545D4E(%rip),%xmm5,%xmm5
	vmulsd	%xmm2,%xmm3,%xmm3
	vaddsd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3
	vmulsd	%xmm1,%xmm6,%xmm6
	vsubsd	%xmm2,%xmm6,%xmm6
	vmulsd	%xmm1,%xmm0,%xmm0
	vaddsd	%xmm2,%xmm0,%xmm0
	vmulsd	%xmm4,%xmm5,%xmm5
	vsubsd	%xmm5,%xmm6,%xmm6
	vmulsd	%xmm4,%xmm3,%xmm3
	vaddsd	%xmm3,%xmm0,%xmm0
#endif

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%rdx,%r8,8),%xmm4   /* exp(-x) */
	vmovsd	(%rdx,%rax,8),%xmm5   /* exp(x) */
	/* deal with infinite results */
        movslq	%ecx,%rcx
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213SD	(%xmm5,%xmm5,%xmm0)
#else
	vmulsd	%xmm5,%xmm0,%xmm0
	vaddsd	%xmm5,%xmm0,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

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
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

	/* deal with infinite results */
        movslq	%r9d,%rcx
#ifdef TARGET_FMA
#	VFMADDSD	%xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213SD	(%xmm4,%xmm4,%xmm6)
#else
	vmulsd	%xmm4,%xmm6,%xmm6
	vaddsd	%xmm4,%xmm6,%xmm6  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

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
	vmulsd	RZ_OFF(24)(%rsp),%xmm6,%xmm6	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
#ifdef TARGET_FMA
#	VFMADDSD	%xmm0,RZ_OFF(24)(%rsp),%xmm6,%xmm0
	VFMA_231SD	(RZ_OFF(24)(%rsp),%xmm6,%xmm0)
#else
	vmulsd	RZ_OFF(24)(%rsp),%xmm6,%xmm6	/* result *= 2^n */
	vaddsd	%xmm6, %xmm0,%xmm0
#endif

#if defined(_WIN64)
	vmovdqu	RZ_OFF(96)(%rsp), %ymm6
#endif

	RZ_POP
	rep
	ret

	ELF_FUNC(ASM_CONCAT(__fsd_cosh_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fsd_cosh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fsd_sinh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_sinh_,TARGET_VEX_OR_FMA)):

	RZ_PUSH
#if defined(_WIN64)
	vmovdqu	%ymm6, RZ_OFF(96)(%rsp)
#endif

	vmovd 	%xmm0, %rax
        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulsd	%xmm0,%xmm3,%xmm3
	andq	.L__real_mask_unsign(%rip), %rax
	shrq	$47, %rax

	/* Set n = nearest integer to r */
	vcomisd	.L_sinh_max_doubleval(%rip), %xmm0
	ja	LBL(.L_inf)
	vcomisd	.L_sinh_min_doubleval(%rip), %xmm0
	jbe	LBL(.L_sinh_ninf)
	cmpq	$0x7fd4, %rax
	jbe	LBL(.L__fsd_sinh_shortcuts)

	vcvtpd2dq %xmm3,%xmm4	/* convert to integer */
	vcvtdq2pd %xmm4,%xmm1	/* and back to float. */
	xorl	%r9d,%r9d
	movq	$0x1f,%r8

	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovsd	.L__real_log2_by_32_lead(%rip),%xmm2
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,%xmm1,%xmm2,%xmm0
	VFNMA_231SD	(%xmm1,%xmm2,%xmm0)
#else
	vmulsd	%xmm1,%xmm2,%xmm2
	vsubsd	%xmm2,%xmm0,%xmm0	/* r1 in xmm0, */
#endif
	vmovd	%xmm4,%ecx
	leaq	.L__two_to_jby32_table(%rip),%rdx

	/* r2 = - n * logbaseof2_by_32_trail; */
	/* j = n & 0x0000001f; */
	movq	%r8,%rax
	andl	%ecx,%eax
	subl	%ecx,%r9d
	andl	%r9d,%r8d

	vmovapd	%xmm0,%xmm2

#ifdef TARGET_FMA
#	VFMADDSD	%xmm2,.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2
	VFMA_231SD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2)
#else
	vmulsd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1
	vaddsd	%xmm1,%xmm2,%xmm2    /* r = r1 + r2 */
#endif
	/* f1 = .L__two_to_jby32_lead_table[j];  */
	/* f2 = .L__two_to_jby32_trail_table[j]; */
	/* *m = (n - j) / 32; */
	subl	%eax,%ecx
	sarl	$5,%ecx
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

	vmovapd	%xmm2,%xmm1
	vmovsd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovsd	.L__real_3FC5555555548F7C(%rip),%xmm0
	vmovsd	.L__real_3F811115B7AA905E(%rip),%xmm5
	vmovsd	.L__real_3fe0000000000000(%rip),%xmm6

	vmulsd	%xmm2,%xmm1,%xmm1
	vmovapd	%xmm1,%xmm4

#ifdef TARGET_FMA
#	VFNMADDSD	%xmm5,%xmm2,%xmm3,%xmm5
	VFNMA_231SD	(%xmm2,%xmm3,%xmm5)
#	VFMADDSD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFNMADDSD	%xmm6,%xmm2,%xmm0,%xmm6
	VFNMA_231SD	(%xmm2,%xmm0,%xmm6)
#	VFMADDSD	.L__real_3fe0000000000000(%rip),%xmm0,%xmm2,%xmm0
	VFMA_213SD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
	vmulsd		%xmm1,%xmm4,%xmm4
#	VFMSUBSD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5,%xmm5
	VFMS_213SD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5)
#	VFMADDSD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213SD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMSUBSD	%xmm2,%xmm1,%xmm6,%xmm6
	VFMS_213SD	(%xmm2,%xmm1,%xmm6)
#	VFMADDSD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#	VFNMADDSD	%xmm6,%xmm4,%xmm5,%xmm6
	VFNMA_231SD	(%xmm4,%xmm5,%xmm6)
#	VFMADDSD	%xmm0,%xmm3,%xmm4,%xmm0
	VFMA_231SD	(%xmm3,%xmm4,%xmm0)
#else
	vmulsd	%xmm2,%xmm3,%xmm3
	vsubsd	%xmm3,%xmm5,%xmm5
	vaddsd	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3
	vmulsd	%xmm2,%xmm0,%xmm0
	vsubsd	%xmm0,%xmm6,%xmm6
	vaddsd	.L__real_3fe0000000000000(%rip),%xmm0,%xmm0

	vmulsd	%xmm1,%xmm4,%xmm4

	vmulsd	%xmm2,%xmm5,%xmm5
	vsubsd	.L__real_3FA5555555545D4E(%rip),%xmm5,%xmm5
	vmulsd	%xmm2,%xmm3,%xmm3
	vaddsd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3
	vmulsd	%xmm1,%xmm6,%xmm6
	vsubsd	%xmm2,%xmm6,%xmm6
	vmulsd	%xmm1,%xmm0,%xmm0
	vaddsd	%xmm2,%xmm0,%xmm0
	vmulsd	%xmm4,%xmm5,%xmm5
	vsubsd	%xmm5,%xmm6,%xmm6
	vmulsd	%xmm4,%xmm3,%xmm3
	vaddsd	%xmm3,%xmm0,%xmm0
#endif

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%rdx,%r8,8),%xmm4   /* exp(-x) */
	vmovsd	(%rdx,%rax,8),%xmm5   /* exp(x) */
	/* deal with infinite results */
        movslq	%ecx,%rcx
#ifdef TARGET_FMA
#	VFMADDSD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213SD	(%xmm5,%xmm5,%xmm0)
#else
	vmulsd	%xmm5,%xmm0,%xmm0
	vaddsd	%xmm5,%xmm0,%xmm0  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

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
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	vmulsd	RZ_OFF(24)(%rsp),%xmm0,%xmm0	/* result *= 2^n */

	/* deal with infinite results */
        movslq	%r9d,%rcx
#ifdef TARGET_FMA
#	VFMADDSD	%xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213SD	(%xmm4,%xmm4,%xmm6)
#else
	vmulsd	%xmm4,%xmm6,%xmm6
	vaddsd	%xmm4,%xmm6,%xmm6  /* z = z1 + z2   done with 1,2,3,4,5 */
#endif

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
	vmulsd	RZ_OFF(24)(%rsp),%xmm6,%xmm6	/* result *= 2^n */

	/* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
	/* Step 3. Reconstitute. */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
#ifdef TARGET_FMA
#	VFNMADDSD	%xmm0,RZ_OFF(24)(%rsp),%xmm6,%xmm0
	VFNMA_231SD	(RZ_OFF(24)(%rsp),%xmm6,%xmm0)
#else
	vmulsd	RZ_OFF(24)(%rsp),%xmm6,%xmm6	/* result *= 2^n */
	vsubsd	%xmm6, %xmm0,%xmm0
#endif

LBL(.L__fsd_sinh_done):

#if defined(_WIN64)
	vmovdqu	RZ_OFF(96)(%rsp), %ymm6
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
  	     dsihn = (((y4*x2 + y3)*x2 + y2)*x2 + y1)*x3 + x
	*/
	vmovapd	%xmm0, %xmm1
	vmovapd	%xmm0, %xmm2
	vmulsd	%xmm0, %xmm0, %xmm0
	vmulsd 	%xmm1, %xmm1, %xmm1
	vmulsd	.L__dsinh_shortval_y7(%rip), %xmm0, %xmm0
	vaddsd	.L__dsinh_shortval_y6(%rip), %xmm0, %xmm0
#ifdef TARGET_FMA
#	VFMADDSD	.L__dsinh_shortval_y5(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dsinh_shortval_y5(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dsinh_shortval_y4(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dsinh_shortval_y4(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dsinh_shortval_y3(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dsinh_shortval_y3(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dsinh_shortval_y2(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dsinh_shortval_y2(%rip),%xmm1,%xmm0)
#	VFMADDSD	.L__dsinh_shortval_y1(%rip),%xmm0,%xmm1,%xmm0
	VFMA_213SD	(.L__dsinh_shortval_y1(%rip),%xmm1,%xmm0)
	vmulsd		%xmm2, %xmm1, %xmm1
#	VFMADDSD	%xmm2,%xmm0,%xmm1,%xmm0
	VFMA_213SD	(%xmm2,%xmm1,%xmm0)
#else
	vmulsd	%xmm1, %xmm0, %xmm0
	vaddsd	.L__dsinh_shortval_y5(%rip), %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vaddsd	.L__dsinh_shortval_y4(%rip), %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vaddsd	.L__dsinh_shortval_y3(%rip), %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vaddsd	.L__dsinh_shortval_y2(%rip), %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vaddsd	.L__dsinh_shortval_y1(%rip), %xmm0, %xmm0
	vmulsd	%xmm2, %xmm1, %xmm1
	vmulsd	%xmm1, %xmm0, %xmm0
	vaddsd	%xmm2, %xmm0, %xmm0
#endif
	jmp	LBL(.L__fsd_sinh_done)

	ELF_FUNC(ASM_CONCAT(__fsd_sinh_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fsd_sinh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        movq    %rsi, 64(%rsp)
        movq    %rdi, 56(%rsp)
        vmovdqu %ymm6, 192(%rsp)
        vmovdqu %ymm7, 224(%rsp)  /* COSH needs xmm7 */
#endif

        /* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
        /* Step 1. Reduce the argument. */
        /* r = x * thirtytwo_by_logbaseof2; */
        vmovhlps  %xmm0, %xmm1, %xmm1
        vmovaps  %xmm0, %xmm5
        vcvtps2pd %xmm0, %xmm2          /* xmm2 = dble(a(2)), dble(a(1)) */
        vcvtps2pd %xmm1, %xmm1          /* xmm1 = dble(a(4)), dble(a(3)) */
        vandps   .L__ps_mask_unsign(%rip), %xmm5, %xmm5
        vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm3
        vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm4
        vcmpps  $6, .L_sp_sinh_max_singleval(%rip), %xmm5, %xmm5
        vmulpd  %xmm2, %xmm3, %xmm3
        vmulpd  %xmm1, %xmm4, %xmm4
        vmovmskps %xmm5, %r8d

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm3,%xmm5   /* convert to integer */
        vcvtpd2dq %xmm4,%xmm6   /* convert to integer */
        test     $15, %r8d
        vcvtdq2pd %xmm5,%xmm3   /* and back to float. */
        vcvtdq2pd %xmm6,%xmm4   /* and back to float. */
        jnz     LBL(.L__Scalar_fvscosh)

        /* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#        VFNMADDPD       %xmm2,.L__real_log2_by_32(%rip),%xmm3,%xmm2
	VFNMA_231PD	(.L__real_log2_by_32(%rip),%xmm3,%xmm2)
#        VFNMADDPD       %xmm1,.L__real_log2_by_32(%rip),%xmm4,%xmm1
	VFNMA_231PD	(.L__real_log2_by_32(%rip),%xmm4,%xmm1)
#else
        vmulpd  .L__real_log2_by_32(%rip),%xmm3,%xmm3
        vsubpd  %xmm3,%xmm2,%xmm2       /* r1 in xmm2, */
        vmulpd  .L__real_log2_by_32(%rip),%xmm4,%xmm4
        vsubpd  %xmm4,%xmm1,%xmm1       /* r1 in xmm1, */
#endif
        vmovq   %xmm5,168(%rsp)
        vmovq   %xmm6,160(%rsp)
        leaq    .L__two_to_jby32_table(%rip),%rax

        /* j = n & 0x0000001f; */
        mov     172(%rsp),%r8d
        mov     168(%rsp),%r9d
        mov     164(%rsp),%r10d
        mov     160(%rsp),%r11d
        movq    $0x1f, %rcx
        and     %r8d, %ecx
        movq    $0x1f, %rdx
        and     %r9d, %edx
        vmovapd %xmm2,%xmm0
        vmovapd %xmm1,%xmm3
        vmovapd %xmm2,%xmm4
        vmovapd %xmm1,%xmm5

        vxorps  %xmm6, %xmm6, %xmm6             /* COSH zero out this register */
        vpsubd  160(%rsp), %xmm6, %xmm6
        vmovdqa %xmm6, 160(%rsp) /* Now contains -n */

        movq    $0x1f, %rsi
        and     %r10d, %esi
        movq    $0x1f, %rdi
        and     %r11d, %edi

        vmovapd .L__real_3fe0000000000000(%rip),%xmm6  /* COSH needs */
        vmovapd .L__real_3fe0000000000000(%rip),%xmm7

        sub     %ecx,%r8d
        sar     $5,%r8d
        sub     %edx,%r9d
        sar     $5,%r9d

        /* Step 2. Compute the polynomial. */
        /* q = r1 + (r2 +
           r*r*( 5.00000000000000008883e-01 +
           r*( 1.66666666665260878863e-01 +
           r*( 4.16666666662260795726e-02 +
           r*( 8.33336798434219616221e-03 +
           r*( 1.38889490863777199667e-03 ))))));
           q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
        vmulpd  .L__real_3FC5555555548F7C(%rip),%xmm0,%xmm0
        vmulpd  .L__real_3FC5555555548F7C(%rip),%xmm1,%xmm1

        sub     %esi,%r10d
        sar     $5,%r10d
        sub     %edi,%r11d
        sar     $5,%r11d

        vmulpd  %xmm2,%xmm2,%xmm2
        vmulpd  %xmm3,%xmm3,%xmm3
        vsubpd  %xmm0,%xmm6,%xmm6   /* COSH exp(-x) */
        vsubpd  %xmm1,%xmm7,%xmm7   /* COSH exp(-x) */
        vaddpd  .L__real_3fe0000000000000(%rip),%xmm0,%xmm0
        vaddpd  .L__real_3fe0000000000000(%rip),%xmm1,%xmm1
#ifdef TARGET_FMA
#        VFMSUBPD        %xmm4,%xmm2,%xmm6,%xmm6
	VFMS_213PD	(%xmm4,%xmm2,%xmm6)
#        VFMSUBPD        %xmm5,%xmm3,%xmm7,%xmm7
	VFMS_213PD	(%xmm5,%xmm3,%xmm7)
#        VFMADDPD        %xmm4,%xmm0,%xmm2,%xmm2
	VFMA_213PD	(%xmm4,%xmm0,%xmm2)
#        VFMADDPD        %xmm5,%xmm1,%xmm3,%xmm3
	VFMA_213PD	(%xmm5,%xmm1,%xmm3)
#else
        vmulpd  %xmm2,%xmm6,%xmm6   /* COSH exp(-x) */
        vsubpd  %xmm4,%xmm6,%xmm6   /* COSH exp(-x) */
        vmulpd  %xmm3,%xmm7,%xmm7   /* COSH exp(-x) */
        vsubpd  %xmm5,%xmm7,%xmm7   /* COSH exp(-x) */
        vmulpd  %xmm0,%xmm2,%xmm2
        vaddpd  %xmm4,%xmm2,%xmm2
        vmulpd  %xmm1,%xmm3,%xmm3
        vaddpd  %xmm5,%xmm3,%xmm3
#endif

        vmovsd  (%rax,%rdx,8),%xmm0
        vmovhpd (%rax,%rcx,8),%xmm0,%xmm0
        vmovsd  (%rax,%rdi,8),%xmm1
        vmovhpd (%rax,%rsi,8),%xmm1,%xmm1
        movq    $0x1f, %rcx
        andl    172(%rsp), %ecx
        movq    $0x1f, %rdx
        andl    168(%rsp), %edx
        movq    $0x1f, %rsi
        andl    164(%rsp), %esi
        movq    $0x1f, %rdi
        andl    160(%rsp), %edi

/*      vsubpd  %xmm4,%xmm6,%xmm6 */            /* COSH exp(-x) */
/*      vsubpd  %xmm5,%xmm7,%xmm7 */            /* COSH exp(-x) */
/*      vaddpd  %xmm4,%xmm2,%xmm2 */
/*      vaddpd  %xmm5,%xmm3,%xmm3 */

        vmovsd  (%rax,%rdx,8),%xmm4
        vmovhpd (%rax,%rcx,8),%xmm4,%xmm4
        vmovsd  (%rax,%rdi,8),%xmm5
        vmovhpd (%rax,%rsi,8),%xmm5,%xmm5

        /* *z2 = f2 + ((f1 + f2) * q); */
        add     $1022, %r8d     /* add bias */
        add     $1022, %r9d     /* add bias */
        add     $1022, %r10d    /* add bias */
        add     $1022, %r11d    /* add bias */

        /* deal with infinite and denormal results */

#ifdef TARGET_FMA
#        VFMADDPD        %xmm0,%xmm0,%xmm2,%xmm2
	VFMA_213PD	(%xmm0,%xmm0,%xmm2)
#        VFMADDPD        %xmm1,%xmm1,%xmm3,%xmm3
	VFMA_213PD	(%xmm1,%xmm1,%xmm3)
#        VFMADDPD        %xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213PD	(%xmm4,%xmm4,%xmm6)
#        VFMADDPD        %xmm5,%xmm5,%xmm7,%xmm7
	VFMA_213PD	(%xmm5,%xmm5,%xmm7)
#else
        vmulpd  %xmm0,%xmm2,%xmm2
        vaddpd  %xmm0,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
        vmulpd  %xmm1,%xmm3,%xmm3
        vaddpd  %xmm1,%xmm3,%xmm3  /* z = z1 + z2   done with 1,2,3,4,5 */
        vmulpd  %xmm4,%xmm6,%xmm6   /* COSH exp(-x) */
        vaddpd  %xmm4,%xmm6,%xmm6   /* COSH exp(-x) */
        vmulpd  %xmm5,%xmm7,%xmm7   /* COSH exp(-x) */
        vaddpd  %xmm5,%xmm7,%xmm7   /* COSH exp(-x) */
#endif

        shlq    $52,%r8
        shlq    $52,%r9
        shlq    $52,%r10
        shlq    $52,%r11

        /* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
        /* Step 3. Reconstitute. */
        movq    %r9,104(%rsp)    /* get 2^n to memory */
        movq    %r8,112(%rsp)    /* get 2^n to memory */

        movq    %r11,88(%rsp)   /* get 2^n to memory */
        movq    %r10,96(%rsp)   /* get 2^n to memory */

        mov     172(%rsp),%r8d
        mov     168(%rsp),%r9d
        mov     164(%rsp),%r10d
        mov     160(%rsp),%r11d

        vmulpd  104(%rsp),%xmm2,%xmm2    /* result *= 2^n */
        vmulpd  88(%rsp),%xmm3,%xmm3    /* result *= 2^n */

        sub     %ecx,%r8d
        sar     $5,%r8d
        sub     %edx,%r9d
        sar     $5,%r9d
        sub     %esi,%r10d
        sar     $5,%r10d
        sub     %edi,%r11d
        sar     $5,%r11d

        add     $1022, %r8d     /* add bias */
        add     $1022, %r9d     /* add bias */
        add     $1022, %r10d    /* add bias */
        add     $1022, %r11d    /* add bias */

        shlq    $52,%r8
        shlq    $52,%r9
        shlq    $52,%r10
        shlq    $52,%r11

        movq    %r9,104(%rsp)    /* get 2^n to memory */
        movq    %r8,112(%rsp)    /* get 2^n to memory */

        movq    %r11,88(%rsp)   /* get 2^n to memory */
        movq    %r10,96(%rsp)   /* get 2^n to memory */

#ifdef TARGET_FMA
#        VFMADDPD        %xmm2,104(%rsp),%xmm6,%xmm2
	VFMA_231PD	(104(%rsp),%xmm6,%xmm2)
#        VFMADDPD        %xmm3,88(%rsp),%xmm7,%xmm3
	VFMA_231PD	(88(%rsp),%xmm7,%xmm3)
#else
        vmulpd  104(%rsp),%xmm6,%xmm6    /* result *= 2^n */
        vmulpd  88(%rsp),%xmm7,%xmm7    /* result *= 2^n */

        vaddpd  %xmm6,%xmm2,%xmm2       /* COSH result = exp(x) + exp(-x) */
        vaddpd  %xmm7,%xmm3,%xmm3       /* COSH result = exp(x) + exp(-x) */
#endif

        vcvtpd2ps %xmm2,%xmm0
        vcvtpd2ps %xmm3,%xmm1
        vshufps $68,%xmm1,%xmm0,%xmm0

LBL(.L_fvcosh_final_check):

#if defined(_WIN64)
        movq    64(%rsp), %rsi
        movq    56(%rsp), %rdi
        vmovdqu 192(%rsp), %ymm6
        vmovdqu 224(%rsp), %ymm7
#endif

        movq    %rbp, %rsp
        popq    %rbp
        rep
        ret

LBL(.L__Scalar_fvscosh):
        /* Need to restore callee-saved regs can do here for this path
         * because entry was only thru fvs_cosh_fma4/fvs_cosh_vex
         */
#if defined(_WIN64)
        movq    64(%rsp), %rsi
        movq    56(%rsp), %rdi
        vmovdqu 192(%rsp), %ymm6
        vmovdqu 224(%rsp), %ymm7
#endif
        vmovaps  %xmm0, _SX0(%rsp)

        CALL(ENT(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovss   _SX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovss   _SX2(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovss   _SX3(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_cosh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	rep
	ret

        ELF_FUNC(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        movq    %rsi, 64(%rsp)
        movq    %rdi, 56(%rsp)
        vmovdqu %ymm6, 192(%rsp)
        vmovdqu %ymm7, 224(%rsp)  /* COSH needs xmm7 */
#endif

        /* Assume a(4) a(3) a(2) a(1) coming in */

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
        /* Step 1. Reduce the argument. */
        /* r = x * thirtytwo_by_logbaseof2; */
        vmovhlps  %xmm0, %xmm1, %xmm1
        vmovaps  %xmm0, %xmm5
        vmovaps .L__ps_vssinh_too_small(%rip), %xmm3
        vandps   .L__ps_mask_unsign(%rip), %xmm5, %xmm5
        vcmpps  $5, %xmm5, %xmm3, %xmm3
        vcmpps  $6, .L_sp_sinh_max_singleval(%rip), %xmm5, %xmm5
        vmovmskps %xmm3, %r9d
        test     $15, %r9d
        jnz     LBL(.L__Scalar_fvssinh)
        vmovmskps %xmm5, %r8d
        test     $15, %r8d
        jnz     LBL(.L__Scalar_fvssinh)

        vcvtps2pd %xmm0, %xmm2          /* xmm2 = dble(a(2)), dble(a(1)) */
        vcvtps2pd %xmm1, %xmm1          /* xmm1 = dble(a(4)), dble(a(3)) */
        vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm3
        vmovapd .L__real_thirtytwo_by_log2(%rip),%xmm4
        vmulpd  %xmm2, %xmm3, %xmm3
        vmulpd  %xmm1, %xmm4, %xmm4

        /* Set n = nearest integer to r */
        vcvtpd2dq %xmm3,%xmm5   /* convert to integer */
        vcvtpd2dq %xmm4,%xmm6   /* convert to integer */
        vcvtdq2pd %xmm5,%xmm3   /* and back to float. */
        vcvtdq2pd %xmm6,%xmm4   /* and back to float. */

        /* r1 = x - n * logbaseof2_by_32_lead; */
#ifdef TARGET_FMA
#        VFNMADDPD       %xmm2,.L__real_log2_by_32(%rip),%xmm3,%xmm2
	VFNMA_231PD	(.L__real_log2_by_32(%rip),%xmm3,%xmm2)
#        VFNMADDPD       %xmm1,.L__real_log2_by_32(%rip),%xmm4,%xmm1
	VFNMA_231PD	(.L__real_log2_by_32(%rip),%xmm4,%xmm1)
#else
        vmulpd  .L__real_log2_by_32(%rip),%xmm3,%xmm3
        vsubpd  %xmm3,%xmm2,%xmm2       /* r1 in xmm2, */
        vmulpd  .L__real_log2_by_32(%rip),%xmm4,%xmm4
        vsubpd  %xmm4,%xmm1,%xmm1       /* r1 in xmm1, */
#endif
        vmovq   %xmm5,168(%rsp)
        vmovq   %xmm6,160(%rsp)
        leaq    .L__two_to_jby32_table(%rip),%rax

        /* j = n & 0x0000001f; */
        mov     172(%rsp),%r8d
        mov     168(%rsp),%r9d
        mov     164(%rsp),%r10d
        mov     160(%rsp),%r11d
        movq    $0x1f, %rcx
        and     %r8d, %ecx
        movq    $0x1f, %rdx
        and     %r9d, %edx
        vmovapd %xmm2,%xmm0
        vmovapd %xmm1,%xmm3
        vmovapd %xmm2,%xmm4
        vmovapd %xmm1,%xmm5

        vxorps  %xmm6, %xmm6,%xmm6              /* SINH zero out this register */
        vpsubd  160(%rsp), %xmm6,%xmm6
        vmovdqa %xmm6, 160(%rsp) /* Now contains -n */

        movq    $0x1f, %rsi
        and     %r10d, %esi
        movq    $0x1f, %rdi
        and     %r11d, %edi

        vmovapd .L__real_3fe0000000000000(%rip),%xmm6  /* SINH needs */
        vmovapd .L__real_3fe0000000000000(%rip),%xmm7

        sub     %ecx,%r8d
        sar     $5,%r8d
        sub     %edx,%r9d
        sar     $5,%r9d

        /* Step 2. Compute the polynomial. */
        /* q = r1 + (r2 +
           r*r*( 5.00000000000000008883e-01 +
           r*( 1.66666666665260878863e-01 +
           r*( 4.16666666662260795726e-02 +
           r*( 8.33336798434219616221e-03 +
           r*( 1.38889490863777199667e-03 ))))));
           q = r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 */
        vmulpd  .L__real_3FC5555555548F7C(%rip),%xmm0,%xmm0
        vmulpd  .L__real_3FC5555555548F7C(%rip),%xmm1,%xmm1

        sub     %esi,%r10d
        sar     $5,%r10d
        sub     %edi,%r11d
        sar     $5,%r11d

        vmulpd  %xmm2,%xmm2,%xmm2
        vmulpd  %xmm3,%xmm3,%xmm3
        vsubpd  %xmm0,%xmm6,%xmm6   /* SINH exp(-x) */
        vsubpd  %xmm1,%xmm7,%xmm7   /* SINH exp(-x) */
        vaddpd  .L__real_3fe0000000000000(%rip),%xmm0,%xmm0
        vaddpd  .L__real_3fe0000000000000(%rip),%xmm1,%xmm1
#ifdef TARGET_FMA
#        VFMSUBPD        %xmm4,%xmm2,%xmm6,%xmm6
	VFMS_213PD	(%xmm4,%xmm2,%xmm6)
#        VFMSUBPD        %xmm5,%xmm3,%xmm7,%xmm7
	VFMS_213PD	(%xmm5,%xmm3,%xmm7)
#        VFMADDPD        %xmm4,%xmm0,%xmm2,%xmm2
	VFMA_213PD	(%xmm4,%xmm0,%xmm2)
#        VFMADDPD        %xmm5,%xmm1,%xmm3,%xmm3
	VFMA_213PD	(%xmm5,%xmm1,%xmm3)
#else
        vmulpd  %xmm2,%xmm6,%xmm6   /* SINH exp(-x) */
        vsubpd  %xmm4,%xmm6,%xmm6   /* SINH exp(-x) */
        vmulpd  %xmm3,%xmm7,%xmm7   /* SINH exp(-x) */
        vsubpd  %xmm5,%xmm7,%xmm7   /* SINH exp(-x) */
        vmulpd  %xmm0,%xmm2,%xmm2
        vaddpd  %xmm4,%xmm2,%xmm2
        vmulpd  %xmm1,%xmm3,%xmm3
        vaddpd  %xmm5,%xmm3,%xmm3
#endif

        vmovsd  (%rax,%rdx,8),%xmm0
        vmovhpd (%rax,%rcx,8),%xmm0,%xmm0
        vmovsd  (%rax,%rdi,8),%xmm1
        vmovhpd (%rax,%rsi,8),%xmm1,%xmm1
        movq    $0x1f, %rcx
        andl    172(%rsp), %ecx
        movq    $0x1f, %rdx
        andl    168(%rsp), %edx
        movq    $0x1f, %rsi
        andl    164(%rsp), %esi
        movq    $0x1f, %rdi
        andl    160(%rsp), %edi

        vmovsd  (%rax,%rdx,8),%xmm4
        vmovhpd (%rax,%rcx,8),%xmm4,%xmm4
        vmovsd  (%rax,%rdi,8),%xmm5
        vmovhpd (%rax,%rsi,8),%xmm5,%xmm5

        /* *z2 = f2 + ((f1 + f2) * q); */
        add     $1022, %r8d     /* add bias */
        add     $1022, %r9d     /* add bias */
        add     $1022, %r10d    /* add bias */
        add     $1022, %r11d    /* add bias */

        /* deal with infinite and denormal results */
#ifdef TARGET_FMA
#        VFMADDPD        %xmm0,%xmm0,%xmm2,%xmm2
	VFMA_213PD	(%xmm0,%xmm0,%xmm2)
#        VFMADDPD        %xmm1,%xmm1,%xmm3,%xmm3
	VFMA_213PD	(%xmm1,%xmm1,%xmm3)
#        VFMADDPD        %xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213PD	(%xmm4,%xmm4,%xmm6)
#        VFMADDPD        %xmm5,%xmm5,%xmm7,%xmm7
	VFMA_213PD	(%xmm5,%xmm5,%xmm7)
#else
        vmulpd  %xmm0,%xmm2,%xmm2
        vaddpd  %xmm0,%xmm2,%xmm2  /* z = z1 + z2   done with 1,2,3,4,5 */
        vmulpd  %xmm1,%xmm3,%xmm3
        vaddpd  %xmm1,%xmm3,%xmm3  /* z = z1 + z2   done with 1,2,3,4,5 */
        vmulpd  %xmm4,%xmm6,%xmm6   /* SINH exp(-x) */
        vaddpd  %xmm4,%xmm6,%xmm6   /* SINH exp(-x) */
        vmulpd  %xmm5,%xmm7,%xmm7   /* SINH exp(-x) */
        vaddpd  %xmm5,%xmm7,%xmm7   /* SINH exp(-x) */
#endif

        shlq    $52,%r8
        shlq    $52,%r9
        shlq    $52,%r10
        shlq    $52,%r11

        /* end of splitexp */
        /* Scale (z1 + z2) by 2.0**m */
        /* Step 3. Reconstitute. */
        movq    %r9,104(%rsp)    /* get 2^n to memory */
        movq    %r8,112(%rsp)    /* get 2^n to memory */

        movq    %r11,88(%rsp)   /* get 2^n to memory */
        movq    %r10,96(%rsp)   /* get 2^n to memory */

        mov     172(%rsp),%r8d
        mov     168(%rsp),%r9d
        mov     164(%rsp),%r10d
        mov     160(%rsp),%r11d

        vmulpd  104(%rsp),%xmm2,%xmm2    /* result *= 2^n */
        vmulpd  88(%rsp),%xmm3,%xmm3    /* result *= 2^n */

        sub     %ecx,%r8d
        sar     $5,%r8d
        sub     %edx,%r9d
        sar     $5,%r9d
        sub     %esi,%r10d
        sar     $5,%r10d
        sub     %edi,%r11d
        sar     $5,%r11d

        add     $1022, %r8d     /* add bias */
        add     $1022, %r9d     /* add bias */
        add     $1022, %r10d    /* add bias */
        add     $1022, %r11d    /* add bias */

        shlq    $52,%r8
        shlq    $52,%r9
        shlq    $52,%r10
        shlq    $52,%r11

        movq    %r9,104(%rsp)    /* get 2^n to memory */
        movq    %r8,112(%rsp)    /* get 2^n to memory */

        movq    %r11,88(%rsp)   /* get 2^n to memory */
        movq    %r10,96(%rsp)   /* get 2^n to memory */

#ifdef TARGET_FMA
#        VFNMADDPD       %xmm2,104(%rsp),%xmm6,%xmm2
	VFNMA_231PD	(104(%rsp),%xmm6,%xmm2)
#        VFNMADDPD       %xmm3,88(%rsp),%xmm7,%xmm3
	VFNMA_231PD	(88(%rsp),%xmm7,%xmm3)
#else
        vmulpd  104(%rsp),%xmm6,%xmm6    /* result *= 2^n */
        vmulpd  88(%rsp),%xmm7,%xmm7    /* result *= 2^n */

        vsubpd  %xmm6,%xmm2,%xmm2       /* SINH result = exp(x) - exp(-x) */
        vsubpd  %xmm7,%xmm3,%xmm3       /* SINH result = exp(x) - exp(-x) */
#endif

        vcvtpd2ps %xmm2,%xmm0
        vcvtpd2ps %xmm3,%xmm1
        vshufps $68,%xmm1,%xmm0,%xmm0

LBL(.L_fvsinh_final_check):

#if defined(_WIN64)
        movq    64(%rsp), %rsi
        movq    56(%rsp), %rdi
        vmovdqu 192(%rsp), %ymm6
        vmovdqu 224(%rsp), %ymm7
#endif

        movq    %rbp, %rsp
        popq    %rbp
        rep
        ret

LBL(.L__Scalar_fvssinh):
        /* Need to restore callee-saved regs can do here for this path
         * because entry was only thru fvs_sinh_fma4/fvs_sinh_vex
         */
#if defined(_WIN64)
        movq    64(%rsp), %rsi
        movq    56(%rsp), %rdi
        vmovdqu 192(%rsp), %ymm6
        vmovdqu 224(%rsp), %ymm7
#endif
        vmovaps  %xmm0, _SX0(%rsp)

        CALL(ENT(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR0(%rsp)

        vmovss   _SX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR1(%rsp)

        vmovss   _SX2(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR2(%rsp)

        vmovss   _SX3(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fss_sinh_,TARGET_VEX_OR_FMA)))

        vmovss   %xmm0, _SR3(%rsp)

        vmovaps  _SR0(%rsp), %xmm0

        movq    %rbp, %rsp
        popq    %rbp
	rep
	ret

        ELF_FUNC(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd	%xmm0, %xmm2
	vmovapd	.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulpd	%xmm0,%xmm3,%xmm3

	/* save x for later. */
	vandpd	.L__real_mask_unsign(%rip), %xmm2,%xmm2

        /* Set n = nearest integer to r */
	vcvtpd2dq %xmm3,%xmm4
	vcmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2,%xmm2
	vcvtdq2pd %xmm4,%xmm1
	vmovmskpd %xmm2, %r8d

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovapd	.L__real_log2_by_32_lead(%rip),%xmm2
	vmulpd	%xmm1,%xmm2,%xmm2
	vmovq	 %xmm4,RZ_OFF(24)(%rsp)
	testl	$3, %r8d
	jnz	LBL(.L__Scalar_fvdcosh)

#if defined(_WIN64)
        vmovdqu  %ymm6, 72(%rsp)
#endif

	/* r2 =   - n * logbaseof2_by_32_trail; */
	vsubpd	%xmm2,%xmm0,%xmm0 	/* r1 in xmm0, */

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	movl	RZ_OFF(24)(%rsp),%ecx
	andl	%ecx,%r9d

	movl	RZ_OFF(20)(%rsp),%edx
	andl	%edx,%r8d
	vmovapd	%xmm0,%xmm2

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
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2
	VFMA_231PD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2)
#else
	vmulpd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1 	/* r2 in xmm1 */
	vaddpd	%xmm1,%xmm2,%xmm2    /* r = r1 + r2 */
#endif

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
	vmovapd	%xmm2,%xmm1

	vmovapd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovapd	.L__real_3FC5555555548F7C(%rip),%xmm0
	vmovapd	.L__real_3F811115B7AA905E(%rip),%xmm5
	vmovapd	.L__real_3fe0000000000000(%rip),%xmm6

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	leaq	.L__two_to_jby32_table(%rip),%r11

	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

#ifdef TARGET_FMA
#	VFNMADDPD	%xmm5,%xmm2,%xmm3,%xmm5
	VFNMA_231PD	(%xmm2,%xmm3,%xmm5)
#	VFMADDPD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFNMADDPD	%xmm6,%xmm0,%xmm2,%xmm6
	VFNMA_231PD	(%xmm0,%xmm2,%xmm6)
#	VFMADDPD	.L__real_3fe0000000000000(%rip),%xmm2,%xmm0,%xmm0
	VFMA_213PD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
#else
	vmulpd	%xmm2,%xmm3,%xmm3	/* r*c5 */
	vsubpd	%xmm3,%xmm5,%xmm5	/* c4 - r*c5 */
	vaddpd	.L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3  /* c4 + r*c5 */
	vmulpd	%xmm2,%xmm0,%xmm0	/* r*c2 */
	vsubpd	%xmm0,%xmm6,%xmm6	                        /* c1 - r*c2 */
	vaddpd	.L__real_3fe0000000000000(%rip),%xmm0,%xmm0  /* c1 + r*c2 */
#endif

	vmulpd	%xmm2,%xmm1,%xmm1	/* r*r */
	vmovapd	%xmm1,%xmm4
	vmulpd	%xmm1,%xmm4,%xmm4	/* r^4 */

#ifdef TARGET_FMA
#	VFMSUBPD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5,%xmm5
	VFMS_213PD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5)
#	VFMADDPD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#        VFMSUBPD        %xmm2,%xmm1,%xmm6,%xmm6
	VFMS_213PD	(%xmm2,%xmm1,%xmm6)
#        VFMADDPD        %xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm1,%xmm0)
#else
	vmulpd	%xmm2,%xmm5,%xmm5	/* r*c4 - r^2*c5 */
	vsubpd	.L__real_3FA5555555545D4E(%rip),%xmm5,%xmm5 /* -c3 + r*c4 - r^2*c5 */
	vmulpd	%xmm2,%xmm3,%xmm3	/* r*c4 + r^2*c5 */
	vaddpd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3 /* c3 + r*c4 + r^2*c5 */
	vmulpd	%xmm1,%xmm6,%xmm6	/* r^2*c1 - r^3*c2 */
	vsubpd	%xmm2,%xmm6,%xmm6	/* -r + r^2*c1 - r^3*c2 */
	vmulpd	%xmm1,%xmm0,%xmm0	/* r^2*c1 + r^3*c2 */
	vaddpd	%xmm2,%xmm0,%xmm0	/* r + r^2*c1 + r^3*c2 */
#endif

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1022,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

#ifdef TARGET_FMA
#	VFNMADDPD	%xmm6,%xmm4,%xmm5,%xmm6
	VFNMA_231PD	(%xmm4,%xmm5,%xmm6)
#	VFMADDPD	%xmm0,%xmm4,%xmm3,%xmm0
	VFMA_231PD	(%xmm4,%xmm3,%xmm0)
#else
	vmulpd	%xmm4,%xmm5,%xmm5	/* -r^4*c3 + r^5*c4 - r^6*c5 */
	vsubpd	%xmm5,%xmm6,%xmm6	/* q = final sum */
	vmulpd	%xmm4,%xmm3,%xmm3	/* r^4*c3 + r^5*c4 + r^6*c5 */
	vaddpd	%xmm3,%xmm0,%xmm0	/* q = final sum */
#endif

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	vmovhpd	(%r11,%r8,8),%xmm5,%xmm5 	/* f1 + f2 */
	movl	RZ_OFF(16)(%rsp),%r9d
	movl	RZ_OFF(12)(%rsp),%r8d

	vmovsd	(%r11,%r9,8),%xmm4 	/* f1 + f2 */
	vmovhpd	(%r11,%r8,8),%xmm4,%xmm4 	/* f1 + f2 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm5,%xmm0)
#	VFMADDPD	%xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213PD	(%xmm4,%xmm4,%xmm6)
#else
	vmulpd	%xmm5,%xmm0,%xmm0
	vaddpd	%xmm5,%xmm0,%xmm0		/* z = z1 + z2 */

	vmulpd	%xmm4,%xmm6,%xmm6
	vaddpd	%xmm4,%xmm6,%xmm6		/* z = z1 + z2 */
#endif

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
	vmulpd	RZ_OFF(40)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(56)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(48)(%rsp) 	/* get 2^n to memory */
	vmulpd	RZ_OFF(56)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

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
	vmulpd	RZ_OFF(40)(%rsp),%xmm6,%xmm6  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
#ifdef TARGET_FMA
#	VFMADDPD	%xmm0,RZ_OFF(24)(%rsp),%xmm6,%xmm0
	VFMA_231PD	(RZ_OFF(24)(%rsp),%xmm6,%xmm0)
#else
	vmulpd	RZ_OFF(24)(%rsp),%xmm6,%xmm6  /* result*= 2^n */
	vaddpd	%xmm6,%xmm0,%xmm0		/* done with cosh */
#endif

#if defined(_WIN64)
        vmovdqu  72(%rsp),%ymm6
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
        vmovapd  %xmm0, _DX0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_cosh_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR0(%rsp)

        vmovsd   _DX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fsd_cosh_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR1(%rsp)

        vmovapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

	ELF_FUNC(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA))


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
	.globl ENT(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA)):

	RZ_PUSH

        /* Find m, z1 and z2 such that exp(x) = 2**m * (z1 + z2) */
	/* Step 1. Reduce the argument. */
	/* r = x * thirtytwo_by_logbaseof2; */
	vmovapd		%xmm0, %xmm2
	vmovddup	.L__dsinh_too_small(%rip),%xmm5
	vmovapd		.L__real_thirtytwo_by_log2(%rip),%xmm3
	vmulpd		%xmm0,%xmm3,%xmm3

	/* save x for later. */
	vandpd	.L__real_mask_unsign(%rip), %xmm2,%xmm2

        /* Set n = nearest integer to r */
	vcmppd	$5, %xmm2, %xmm5,%xmm5
	vcmppd	$6, .L__real_ln_max_doubleval(%rip), %xmm2,%xmm2
	vmovmskpd %xmm5, %r9d
	vmovmskpd %xmm2, %r8d

	testl	$3, %r9d
	jnz	LBL(.L__Scalar_fvdsinh)

	testl	$3, %r8d
	jnz	LBL(.L__Scalar_fvdsinh)

#if defined(_WIN64)
        vmovdqu  %ymm6, 72(%rsp)
#endif

	vcvtpd2dq %xmm3,%xmm4
	vcvtdq2pd %xmm4,%xmm1

 	/* r1 = x - n * logbaseof2_by_32_lead; */
	vmovapd	.L__real_log2_by_32_lead(%rip),%xmm2
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm0,%xmm1,%xmm2,%xmm0
	VFNMA_231PD	(%xmm1,%xmm2,%xmm0)
#else
	vmulpd	%xmm1,%xmm2,%xmm2
	vsubpd	%xmm2,%xmm0,%xmm0	/* r1 in xmm0, */
#endif
	vmovq	 %xmm4,RZ_OFF(24)(%rsp)
	/* r2 =   - n * logbaseof2_by_32_trail; */
/*	vsubpd	%xmm2,%xmm0,%xmm0 */	/* r1 in xmm0, */
/*	vmulpd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1 */ 	/* r2 in xmm1 */

	/* j = n & 0x0000001f; */
	movq	$0x01f,%r9
	movq	%r9,%r8
	movl	RZ_OFF(24)(%rsp),%ecx
	andl	%ecx,%r9d

	movl	RZ_OFF(20)(%rsp),%edx
	andl	%edx,%r8d
	vmovapd	%xmm0,%xmm2

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
#ifdef TARGET_FMA
#	VFMADDPD	%xmm2,.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2
	VFMA_231PD	(.L__real_log2_by_32_tail(%rip),%xmm1,%xmm2)
#else
	vmulpd	.L__real_log2_by_32_tail(%rip),%xmm1,%xmm1 	/* r2 in xmm1 */
	vaddpd	%xmm1,%xmm2,%xmm2    /* r = r1 + r2 */
#endif
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
	vmovapd	%xmm2,%xmm1

	vmovapd	.L__real_3f56c1728d739765(%rip),%xmm3
	vmovapd	.L__real_3FC5555555548F7C(%rip),%xmm0
	vmovapd	.L__real_3F811115B7AA905E(%rip),%xmm5
	vmovapd	.L__real_3fe0000000000000(%rip),%xmm6

	movslq	%ecx,%rcx
	movslq	%edx,%rdx
	movq	$1, %rax
	leaq	.L__two_to_jby32_table(%rip),%r11

	/* rax = 1, rcx = exp, r10 = mul */
	/* rax = 1, rdx = exp, r11 = mul */

#ifdef TARGET_FMA
#	VFNMADDPD	%xmm5,%xmm2,%xmm3,%xmm5
	VFNMA_231PD	(%xmm2,%xmm3,%xmm5)
#	VFMADDPD	.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3F811115B7AA905E(%rip),%xmm2,%xmm3)
#	VFNMADDPD	%xmm6,%xmm2,%xmm0,%xmm6
	VFNMA_231PD	(%xmm2,%xmm0,%xmm6)
#	VFMADDPD	.L__real_3fe0000000000000(%rip),%xmm2,%xmm0,%xmm0
	VFMA_213PD	(.L__real_3fe0000000000000(%rip),%xmm2,%xmm0)
#else
	vmulpd	%xmm2,%xmm3,%xmm3	/* r*c5 */
	vsubpd	%xmm3,%xmm5,%xmm5	/* c4 - r*c5 */
	vaddpd	 .L__real_3F811115B7AA905E(%rip),%xmm3,%xmm3  /* c4 + r*c5 */
	vmulpd	%xmm2,%xmm0,%xmm0	/* r*c2 */
	vsubpd	%xmm0,%xmm6,%xmm6	                        /* c1 - r*c2 */
	vaddpd	 .L__real_3fe0000000000000(%rip),%xmm0,%xmm0  /* c1 + r*c2 */
#endif
	vmulpd	%xmm2,%xmm1,%xmm1	/* r*r */
	vmovapd	%xmm1,%xmm4
	vmulpd	%xmm1,%xmm4,%xmm4	/* r^4 */

#ifdef TARGET_FMA
#	VFMSUBPD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5,%xmm5
	VFMS_213PD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm5)
#	VFMADDPD	.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3,%xmm3
	VFMA_213PD	(.L__real_3FA5555555545D4E(%rip),%xmm2,%xmm3)
#	VFMSUBPD	%xmm2,%xmm1,%xmm6,%xmm6
	VFMS_213PD	(%xmm2,%xmm1,%xmm6)
#	VFMADDPD	%xmm2,%xmm1,%xmm0,%xmm0
	VFMA_213PD	(%xmm2,%xmm1,%xmm0)
#	VFNMADDPD	%xmm6,%xmm4,%xmm5,%xmm6
	VFNMA_231PD	(%xmm4,%xmm5,%xmm6)
#	VFMADDPD	%xmm0,%xmm4,%xmm3,%xmm0
	VFMA_231PD	(%xmm4,%xmm3,%xmm0)
#else
	vmulpd	%xmm2,%xmm5,%xmm5	/* r*c4 - r^2*c5 */
	vsubpd	.L__real_3FA5555555545D4E(%rip),%xmm5,%xmm5 /* -c3 + r*c4 - r^2*c5 */
	vmulpd	%xmm2,%xmm3,%xmm3	/* r*c4 + r^2*c5 */
	vaddpd	.L__real_3FA5555555545D4E(%rip),%xmm3,%xmm3 /* c3 + r*c4 + r^2*c5 */
	vmulpd	%xmm1,%xmm6,%xmm6	/* r^2*c1 - r^3*c2 */
	vsubpd	%xmm2,%xmm6,%xmm6	/* -r + r^2*c1 - r^3*c2 */
	vmulpd	%xmm1,%xmm0,%xmm0	/* r^2*c1 + r^3*c2 */
	vaddpd	%xmm2,%xmm0,%xmm0	/* r + r^2*c1 + r^3*c2 */
	vmulpd	%xmm4,%xmm5,%xmm5	/* -r^4*c3 + r^5*c4 - r^6*c5 */
	vsubpd	%xmm5,%xmm6,%xmm6	/* q = final sum */
	vmulpd	%xmm4,%xmm3,%xmm3	/* r^4*c3 + r^5*c4 + r^6*c5 */
	vaddpd	%xmm3,%xmm0,%xmm0	/* q = final sum */
#endif

	/* deal with denormal and close to infinity */
	movq	%rax, %r10	/* 1 */
	addq	$1022,%rcx	/* add bias */
	cmovleq	%rcx, %r10
	cmovleq	%rax, %rcx
	addq	$1022,%r10	/* add bias */
	shlq	$52,%r10	/* build 2^n */

	/* *z2 = f2 + ((f1 + f2) * q); */
	vmovsd	(%r11,%r9,8),%xmm5 	/* f1 + f2 */
	vmovhpd	(%r11,%r8,8),%xmm5,%xmm5 	/* f1 + f2 */
	movl	RZ_OFF(16)(%rsp),%r9d
	movl	RZ_OFF(12)(%rsp),%r8d

	vmovsd	(%r11,%r9,8),%xmm4 	/* f1 + f2 */
	vmovhpd	(%r11,%r8,8),%xmm4,%xmm4 	/* f1 + f2 */

#ifdef TARGET_FMA
#	VFMADDPD	%xmm5,%xmm5,%xmm0,%xmm0
	VFMA_213PD	(%xmm5,%xmm5,%xmm0)
#	VFMADDPD	%xmm4,%xmm4,%xmm6,%xmm6
	VFMA_213PD	(%xmm4,%xmm4,%xmm6)
#else
	vmulpd	%xmm5,%xmm0,%xmm0
	vaddpd	%xmm5,%xmm0,%xmm0		/* z = z1 + z2 */

	vmulpd	%xmm4,%xmm6,%xmm6
	vaddpd	%xmm4,%xmm6,%xmm6		/* z = z1 + z2 */
#endif

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
	vmulpd	RZ_OFF(40)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(56)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(48)(%rsp) 	/* get 2^n to memory */
	vmulpd	RZ_OFF(56)(%rsp),%xmm0,%xmm0  /* result*= 2^n */

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
	vmulpd	RZ_OFF(40)(%rsp),%xmm6,%xmm6  /* result*= 2^n */

	shlq	$52,%rcx		/* build 2^n */
	shlq	$52,%rdx		/* build 2^n */
	movq	%rcx,RZ_OFF(24)(%rsp) 	/* get 2^n to memory */
	movq	%rdx,RZ_OFF(16)(%rsp) 	/* get 2^n to memory */
#ifdef TARGET_FMA
#	VFNMADDPD	%xmm0,RZ_OFF(24)(%rsp),%xmm6,%xmm0
	VFNMA_231PD	(RZ_OFF(24)(%rsp),%xmm6,%xmm0)
#else
	vmulpd	RZ_OFF(24)(%rsp),%xmm6,%xmm6  /* result*= 2^n */
	vsubpd	%xmm6,%xmm0,%xmm0		/* done with sinh */
#endif

#if defined(_WIN64)
        vmovdqu  72(%rsp),%ymm6
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
        vmovapd  %xmm0, _DX0(%rsp)

        CALL(ENT(ASM_CONCAT(__fsd_sinh_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR0(%rsp)

        vmovsd   _DX1(%rsp), %xmm0
        CALL(ENT(ASM_CONCAT(__fsd_sinh_,TARGET_VEX_OR_FMA)))

        vmovsd   %xmm0, _DR1(%rsp)

        vmovapd  _DR0(%rsp), %xmm0
        movq    %rbp, %rsp
        popq    %rbp
	jmp	LBL(.L__final_check)

	ELF_FUNC(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA))
	ELF_SIZE(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision exp
 *
 *  Prototype:
 *
 *      single __fvs_exp_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        movq    %rsi, 192(%rsp)
        movq    %rdi, 224(%rsp)
#endif
        vmovups %ymm0, 48(%rsp)

	CALL(ENT(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

	CALL(ENT(ASM_CONCAT(__fvs_exp_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        movq    %rsi, 192(%rsp)
        movq    %rdi, 224(%rsp)
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_exp_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision exp
 *
 *  Prototype:
 *
 *      double __fvd_exp_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_exp_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_exp_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision sin
 *
 *  Prototype:
 *
 *      single __fvs_sin_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sin_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_sin_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision sin
 *
 *  Prototype:
 *
 *      double __fvd_sin_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sin_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_sin_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision cos
 *
 *  Prototype:
 *
 *      single __fvs_cos_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_cos_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_cos_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision cos
 *
 *  Prototype:
 *
 *      double __fvd_cos_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_cos_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_cos_,TARGET_VEX_OR_FMA,_256))




/* ------------------------------------------------------------------------- */
/*
 *  vector single precision log
 *
 *  Prototype:
 *
 *      single __fvs_log_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $512, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
        vmovdqu %ymm8, 192(%rsp)
        vmovdqu %ymm9, 224(%rsp)
        vmovdqu %ymm10, 256(%rsp)
        vmovdqu %ymm11, 288(%rsp)
        vmovdqu %ymm12, 320(%rsp)
        vmovdqu %ymm13, 352(%rsp)
#endif

        vmovups  .L4_384(%rip), %ymm4   /* Move min arg to xmm4 */
        vxorps  %ymm7, %ymm7, %ymm7             /* Still need 0.0 */
        vmovaps %ymm0, %ymm2            /* Move for nx */
        vmovaps %ymm0, %ymm1            /* Move to xmm1 for later ma */

        /* Check exceptions and valid range */
        vcmpleps        %ymm0, %ymm4, %ymm4             /* '00800000'x <= a, xmm4 1 where true */
        vcmpltps        %ymm0, %ymm7, %ymm7             /* Test for 0.0 < a, xmm7 1 where true */
        vcmpneqps       .L4_387(%rip), %ymm0, %ymm0     /* Test for == +inf */
        vxorps          %ymm7, %ymm4, %ymm4             /* xor to find just denormal inputs */
        vmovmskps       %ymm4, %eax             /* Move denormal mask to gp ref */
        vmovups         %ymm2, 24(%rsp) /* Move for exception processing */
        vmovups         .L4_382(%rip), %ymm3    /* Move 126 */
        cmp             $0, %eax                /* Test for denormals */
        jne             LBL(.LB_DENORMs_256)

LBL(.LB_100_256):
        leaq    .L_STATICS1(%rip),%r8
        vandps  .L4_380(%rip), %ymm1, %ymm1     /* ma = IAND(ia,'007fffff'x) */
        vextractf128    $1, %ymm2, %xmm8
        vpsrld  $23, %xmm2, %xmm2               /* nx = ISHFT(ia,-23) */
        vpsrld  $23, %xmm8, %xmm8

        vandps  %ymm0, %ymm7, %ymm7             /* Mask for nan, inf, neg and 0.0 */
        vmovaps %ymm1, %ymm6            /* move ma for ig */

        vextractf128    $1, %ymm1, %xmm9
        vpsubd  .L4_381(%rip), %xmm1, %xmm1     /* ms = ma - '3504f3'x */
        vpsubd  .L4_381(%rip), %xmm9, %xmm9     /* ms = ma - '3504f3'x */
        vinsertf128     $1, %xmm9, %ymm1, %ymm1

        vextractf128    $1, %ymm3, %xmm10
        vpsubd  %xmm3, %xmm2, %xmm2             /* nx = ISHFT(ia,-23) - 126 */
        vpsubd  %xmm10, %xmm8, %xmm8             /* nx = ISHFT(ia,-23) - 126 */

        vorps   .L4_383(%rip), %ymm6, %ymm6     /* ig = IOR(ma,'3f000000'x) */
        vmovaps %ymm1, %ymm0            /* move ms for tbl ms */
        vandps  .L4_384(%rip), %ymm1, %ymm1     /* mx = IAND(ms,'00800000'x) */
        vandps  .L4_385(%rip), %ymm0, %ymm0     /* ms = IAND(ms,'007f0000'x) */
        vorps   %ymm1, %ymm6, %ymm6             /* ig = IOR(ig, mx) */

        vextractf128    $1, %ymm1, %xmm9
        vpsrad  $23, %xmm1, %xmm1               /* ISHFT(mx,-23) */
        vpsrad  $23, %xmm9, %xmm9               /* ISHFT(mx,-23) */

        vextractf128    $1, %ymm0, %xmm11
        vpsrad  $12, %xmm0, %xmm0               /* ISHFT(ms,-12) for 128 bit reads */
        vpsrad  $12, %xmm11, %xmm11
        vinsertf128     $1, %xmm11, %ymm0, %ymm0

        vmovmskps %ymm7, %eax           /* Move xmm7 mask to eax */

        vpsubd  %xmm1, %xmm2, %xmm2             /* nx = nx - ISHFT(mx,-23) */
        vpsubd  %xmm9, %xmm8, %xmm8             /* nx = nx - ISHFT(mx,-23) */
        vinsertf128     $1, %xmm8, %ymm2, %ymm2
/*      vinsertf128     $1, %xmm9, %ymm1, %ymm1 */

        vmovups %ymm0, 60(%rsp) /* Move to memory */
        vcvtdq2ps  %ymm2, %ymm0         /* xn = real(nx) */

        movl    60(%rsp), %ecx          /* Move to gp register */
        vmovups (%r8,%rcx,1), %xmm1             /* Read from 1st table location */
        movl    64(%rsp), %edx          /* Move to gp register */
        vmovups (%r8,%rdx,1), %xmm2             /* Read from 2nd table location */
        movl    68(%rsp), %ecx          /* Move to gp register */
        vmovups (%r8,%rcx,1), %xmm3             /* Read from 3rd table location */
        movl    72(%rsp), %edx          /* Move to gp register */
        vmovups (%r8,%rdx,1), %xmm4             /* Read from 4th table location */
        movl    76(%rsp), %ecx          /* Move to gp register */
        vmovups (%r8,%rcx,1), %xmm8             /* Read from 5th table location */
        movl    80(%rsp), %edx          /* Move to gp register */
        vmovups (%r8,%rdx,1), %xmm9             /* Read from 6th table location */
        movl    84(%rsp), %ecx          /* Move to gp register */
        vmovups (%r8,%rcx,1), %xmm10             /* Read from 7th table location */
        movl    88(%rsp), %edx          /* Move to gp register */
        vmovups (%r8,%rdx,1), %xmm11             /* Read from 8th table location */

/* first 4*/

        vsubps  .L4_386(%rip), %ymm6, %ymm6     /* x0 = rg - 1.0 */

        vmovaps %xmm1, %xmm5            /* Store 1/3, c0, b0, a0 */
        vmovaps %xmm3, %xmm7            /* Store 1/3, c2, b2, a2 */

        vunpcklps %xmm2, %xmm1, %xmm1           /* b1, b0, a1, a0 */
        vunpcklps %xmm4, %xmm3, %xmm3           /* b3, b2, a3, a2 */
        vunpckhps %xmm2, %xmm5, %xmm5           /* 1/3, 1/3, c1, c0 */
        vunpckhps %xmm4, %xmm7, %xmm7           /* 1/3, 1/3, c3, c2 */

        vmovaps %ymm6, %ymm4            /* move x0 */

        vmovaps         %xmm1, %xmm2            /* Store b1, b0, a1, a0 */

        vmovlhps        %xmm3, %xmm1, %xmm1             /* a3, a2, a1, a0 */
        vmovlhps        %xmm7, %xmm5, %xmm5             /* c3, c2, c1, c0 */
        vmovhlps        %xmm2, %xmm3, %xmm3             /* b3, b2, b1, b0 */
        vmovhlps        %xmm7, %xmm7, %xmm7             /* 1/3, 1/3, 1/3, 1/3 */

/* last 4 */

        vmovaps %xmm8, %xmm12            /* Store 1/3, c4, b4, a4 */
        vmovaps %xmm10, %xmm13            /* Store 1/3, c5, b5, a5 */

        vunpcklps %xmm9, %xmm8, %xmm8           /* b5, b4, a5, a4 */
        vunpcklps %xmm11, %xmm10, %xmm10           /* b7, b6, a7, a6 */
        vunpckhps %xmm9, %xmm12, %xmm12           /* 1/3, 1/3, c5, c4 */
        vunpckhps %xmm11, %xmm13, %xmm13           /* 1/3, 1/3, c7, c6 */

        vmovaps         %xmm8, %xmm9            /* Store b5, b4, a5, a4 */

        vmovlhps        %xmm10, %xmm8, %xmm8             /* a7, a6, a5, a4 */
        vmovlhps        %xmm13, %xmm12, %xmm12             /* c7, c6, c5, c4 */
        vmovhlps        %xmm9, %xmm10, %xmm10             /* b7, b6, b5, b4 */

/* combine 8 */

        vinsertf128     $1, %xmm8, %ymm1, %ymm1         /* a7, a6, a5, a4, a3, a2, a1, a0 */
        vinsertf128     $1, %xmm10, %ymm3, %ymm3        /* b7, b6, b5, b4, b3, b2, b1, b0 */
        vinsertf128     $1, %xmm12, %ymm5, %ymm5        /* c7, c6, c5, c4, c3, c2, c1, c0 */
        vinsertf128     $1, %xmm7, %ymm7, %ymm7         /* 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3 */

#ifdef TARGET_FMA
#        VFMADDPS        %ymm3,%ymm6,%ymm1,%ymm1
	VFMA_213PS	(%ymm3,%ymm6,%ymm1)
#else
        vmulps          %ymm6, %ymm1, %ymm1             /* COEFFS(mt) * x0 */
        vaddps          %ymm3, %ymm1, %ymm1             /* COEFFS(mt) * g + COEFFS(mt+1) */
#endif
        vmulps          %ymm6, %ymm6, %ymm6             /* xsq = x0 * x0 */

        vmovaps %ymm4, %ymm2            /* move x0 */

        vmulps  %ymm6, %ymm4, %ymm4             /* xcu = xsq * x0 */
        vmulps  .L4_383(%rip), %ymm6, %ymm6     /* x1 = 0.5 * xsq */
#ifdef TARGET_FMA
#        VFMADDPS        %ymm5,%ymm2,%ymm1,%ymm1
	VFMA_213PS	(%ymm5,%ymm2,%ymm1)
#else
        vmulps  %ymm2, %ymm1, %ymm1             /* * x0 */
        vaddps  %ymm5, %ymm1, %ymm1             /* + COEFFS(mt+2) = rp */
#endif
        vmulps  %ymm7, %ymm4, %ymm4             /* x2 = thrd * xcu */
        vmovaps %ymm6, %ymm3            /* move x1 */

#ifdef TARGET_FMA
#        VFNMADDPS       %ymm1,%ymm6,%ymm6,%ymm1
	VFNMA_231PS	(%ymm6,%ymm6,%ymm1)
#else
        vmulps  %ymm6, %ymm6, %ymm6             /* x3 = x1 * x1 */
/*      vaddps  %ymm5, %ymm1, %ymm1 */          /* + COEFFS(mt+2) = rp */
        vsubps  %ymm6, %ymm1, %ymm1             /* rp - x3 */
#endif
        vmovups .L4_388(%rip), %ymm7    /* Move c1 */
        vmovups .L4_389(%rip), %ymm6   /* Move c2 */
        vaddps  %ymm1, %ymm4, %ymm4             /* rp - x3 + x2 */
        vsubps  %ymm3, %ymm4, %ymm4             /* rp - x3 + x2 - x1 */
        vaddps  %ymm2, %ymm4, %ymm4             /* rp - x3 + x2 - x1 + x0 = rz */

#ifdef TARGET_FMA
#        VFMADDPS        %ymm4,%ymm0,%ymm7,%ymm4
	VFMA_231PS	(%ymm0,%ymm7,%ymm4)
#        VFMADDPS        %ymm4,%ymm6,%ymm0,%ymm0
	VFMA_213PS	(%ymm4,%ymm6,%ymm0)
#else
        vmulps   %ymm0, %ymm7, %ymm7            /* xn * c1 */
        vaddps   %ymm7, %ymm4, %ymm4            /* (xn * c1 + rz) */
        vmulps   %ymm6, %ymm0, %ymm0            /* xn * c2 */
        vaddps   %ymm4, %ymm0, %ymm0            /* rr = (xn * c1 + rz) + xn * c2 */
#endif

        cmp     $255, %eax
        jne     LBL(.LB_EXCEPTs_256)

LBL(.LB_900_256):


/*******************************************/

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
        vmovdqu 192(%rsp), %ymm8
        vmovdqu 224(%rsp), %ymm9
        vmovdqu 256(%rsp), %ymm10
        vmovdqu 288(%rsp), %ymm11
        vmovdqu 320(%rsp), %ymm12
        vmovdqu 352(%rsp), %ymm13
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

/*******************************************/
LBL(.LB_EXCEPTs_256):
        /* Handle all exceptions by masking in xmm */
        vmovups  24(%rsp), %ymm1        /* original input */
        vmovups  24(%rsp), %ymm2        /* original input */
        vmovups  24(%rsp), %ymm3        /* original input */
        vxorps   %ymm7, %ymm7, %ymm7            /* xmm7 = 0.0 */
        vxorps   %ymm6, %ymm6, %ymm6            /* xmm6 = 0.0 */
        vmovups .L4_394(%rip), %ymm5    /* convert nan bit */
        vxorps   %ymm4, %ymm4, %ymm4            /* xmm4 = 0.0 */

        vcmpunordps %ymm1, %ymm7, %ymm7         /* Test if unordered */
        vcmpltps %ymm6, %ymm2, %ymm2            /* Test if a < 0.0 */
        vcmpordps %ymm1, %ymm6, %ymm6           /* Test if ordered */

        vandps  %ymm7, %ymm5, %ymm5            /* And nan bit where unordered */
        vorps   %ymm7, %ymm4, %ymm4            /* Or all masks together */
        vandps  %ymm1, %ymm7, %ymm7            /* And input where unordered */
        vorps   %ymm5, %ymm7, %ymm7             /* Convert unordered nans */

        vxorps   %ymm5, %ymm5, %ymm5            /* xmm5 = 0.0 */
        vandps   %ymm2, %ymm6, %ymm6            /* Must be ordered and < 0.0 */
        vorps    %ymm6, %ymm4, %ymm4            /* Or all masks together */
        vandps   .L4_390(%rip), %ymm6, %ymm6    /* And -nan if < 0.0 and ordered */

        vcmpeqps .L4_387(%rip), %ymm3, %ymm3    /* Test if equal to infinity */
        vcmpeqps %ymm5, %ymm1, %ymm1            /* Test if eq 0.0 */
        vorps    %ymm6, %ymm7, %ymm7            /* or in < 0.0 */

        vorps    %ymm3, %ymm4, %ymm4            /* Or all masks together */
        vandps   .L4_387(%rip), %ymm3, %ymm3    /* inf and inf mask */
        vmovaps  %ymm0, %ymm2
        vorps    %ymm3, %ymm7, %ymm7            /* or in infinity */

        vorps    %ymm1, %ymm4, %ymm4            /* Or all masks together */
        vandps   .L4_391(%rip), %ymm1, %ymm1    /* And -inf if == 0.0 */
        vmovaps  %ymm4, %ymm0
        vorps    %ymm1, %ymm7, %ymm7            /* or in -infinity */

        vandnps  %ymm2, %ymm0, %ymm0            /* Where mask not set, use result */
        vorps    %ymm7, %ymm0, %ymm0            /* or in exceptional values */
        jmp     LBL(.LB_900_256)

LBL(.LB_DENORMs_256):
        /* Have the denorm mask in xmm4, so use it to scale a and the subtractor */
        vmovaps %ymm4, %ymm5            /* Move mask */
        vmovaps %ymm4, %ymm6            /* Move mask */
        vandps  .L4_392(%rip), %ymm4, %ymm4     /* Have 2**23 where denorms are, 0 else */
        vandnps %ymm1, %ymm5, %ymm5             /* Have a where denormals aren't */
        vmulps  %ymm4, %ymm1, %ymm1             /* denormals * 2**23 */
        vandps  .L4_393(%rip), %ymm6, %ymm6     /* have 23 where denorms are, 0 else */
        vorps   %ymm5, %ymm1, %ymm1             /* Or in the original a */

        vextractf128    $1, %ymm6, %xmm11
        vextractf128    $1, %ymm3, %xmm12
        vpaddd  %xmm6, %xmm3, %xmm3             /* Add 23 to 126 for offseting exponent */
        vpaddd  %xmm11, %xmm12, %xmm12             /* Add 23 to 126 for offseting exponent */
        vinsertf128     $1, %xmm12, %ymm3, %ymm3

        vmovaps %ymm1, %ymm2            /* Move to the next location */
        jmp     LBL(.LB_100_256)


/*******************************************/





        ELF_FUNC(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_log_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision log
 *
 *  Prototype:
 *
 *      double __fvd_log_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_log_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_log_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision log10
 *
 *  Prototype:
 *
 *      single __fvs_log10_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_log10_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_log10_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision log10
 *
 *  Prototype:
 *
 *      double __fvd_log10_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_log10_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_log10_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision sinh
 *
 *  Prototype:
 *
 *      single __fvs_sinh_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
	movq	%rsi, 192(%rsp)
	movq	%rdi, 224(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sinh_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
	movq	192(%rsp), %rsi
	movq	224(%rsp), %rdi
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_sinh_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision sinh
 *
 *  Prototype:
 *
 *      double __fvd_sinh_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sinh_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_sinh_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision cosh
 *
 *  Prototype:
 *
 *      single __fvs_cosh_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
        movq    %rsi, 192(%rsp)
        movq    %rdi, 224(%rsp)
#endif

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_cosh_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
        movq    192(%rsp), %rsi
        movq    224(%rsp), %rdi
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_cosh_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision cosh
 *
 *  Prototype:
 *
 *      double __fvd_cosh_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 48(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA)))


        vmovups 48(%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm1, 80(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_cosh_,TARGET_VEX_OR_FMA)))

        vmovups 80(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_cosh_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision sincos
 *
 *  Prototype:
 *
 *      single __fvs_sincos_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
#endif

        vmovups %ymm0, 32(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)))


        vmovups 32(%rsp), %ymm2
        vmovaps %xmm0, %xmm3
	vmovaps	%xmm1, %xmm4
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm3, 64(%rsp)
	vmovups	%ymm4, 96(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_sincos_,TARGET_VEX_OR_FMA)))

        vmovups 64(%rsp), %ymm3
        vinsertf128     $1, %xmm0, %ymm3, %ymm0
        vmovups 96(%rsp), %ymm4
        vinsertf128     $1, %xmm1, %ymm4, %ymm1

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_sincos_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision sincos
 *
 *  Prototype:
 *
 *      single __fvd_sincos_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
        vmovdqu %ymm7, 160(%rsp)
        vmovdqu	%ymm8, 192(%rsp)
#endif

        vmovups %ymm0, 32(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)))


        vmovups 32(%rsp), %ymm2
        vmovaps %xmm0, %xmm3
        vmovaps %xmm1, %xmm4
        vextractf128    $1, %ymm2, %xmm2
        vmovaps %xmm2, %xmm0
        vmovups %ymm3, 64(%rsp)
        vmovups %ymm4, 96(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_sincos_,TARGET_VEX_OR_FMA)))

        vmovups 64(%rsp), %ymm3
        vinsertf128     $1, %xmm0, %ymm3, %ymm0
        vmovups 96(%rsp), %ymm4
        vinsertf128     $1, %xmm1, %ymm4, %ymm1

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
        vmovdqu 160(%rsp), %ymm7
        vmovdqu	192(%rsp), %ymm8
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_sincos_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector sinle precision pow
 *
 *  Prototype:
 *
 *      single __fvs_pow_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $128, %rsp

        vmovups %ymm0, 32(%rsp)
	vmovups	%ymm1, 96(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA)))


        vmovups		32(%rsp), %ymm2
	vmovups		96(%rsp), %ymm4
        vmovaps 	%xmm0, %xmm3
        vextractf128    $1, %ymm2, %xmm2
	vextractf128	$1, %ymm4, %xmm4
        vmovaps		%xmm2, %xmm0
	vmovaps		%xmm4, %xmm1
        vmovups		%ymm3, 64(%rsp)

        CALL(ENT(ASM_CONCAT(__fvs_pow_,TARGET_VEX_OR_FMA)))

        vmovups 64(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_pow_,TARGET_VEX_OR_FMA,_256))



/* ------------------------------------------------------------------------- */
/*
 *  vector double precision pow
 *
 *  Prototype:
 *
 *      single __fvd_pow_vex/fma4_256(float *x);
 *
 */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256)):


        pushq   %rbp
        movq    %rsp, %rbp
        subq    $256, %rsp

#if defined(_WIN64)
        vmovdqu %ymm6, 128(%rsp)
#endif

        vmovups %ymm0, 32(%rsp)
        vmovups %ymm1, 96(%rsp)
        CALL(ENT(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA)))


        vmovups         32(%rsp), %ymm2
        vmovups         96(%rsp), %ymm4
        vmovaps         %xmm0, %xmm3
        vextractf128    $1, %ymm2, %xmm2
        vextractf128    $1, %ymm4, %xmm4
        vmovaps         %xmm2, %xmm0
        vmovaps         %xmm4, %xmm1
        vmovups         %ymm3, 64(%rsp)

        CALL(ENT(ASM_CONCAT(__fvd_pow_,TARGET_VEX_OR_FMA)))

        vmovups 64(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

#if defined(_WIN64)
        vmovdqu 128(%rsp), %ymm6
#endif

        movq    %rbp, %rsp
        popq    %rbp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_pow_,TARGET_VEX_OR_FMA,_256))


/* -------------------------------------------------------------------------
 *  vector single precision tangent - 128 bit
 *
 *  Prototype:
 *
 *      float * __fvs_tan_[fma4/vex](float *x);
 *
 * ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA)):


	subq    $40, %rsp

        vmovupd  %xmm0, (%rsp)                 /* Save xmm0 */
#if	! defined(TARGET_WIN_X8664)
	vzeroupper
#endif

        CALL(ENT(__mth_i_tan))                 /* tan(x(1)) */
        vmovss   %xmm0, 16(%rsp)               /* Save first result */

        vmovss 4(%rsp),%xmm0                   /* Fetch x(2) */
        CALL(ENT(__mth_i_tan))                 /* tan(x(2)) */
        vmovss   %xmm0, 20(%rsp)               /* Save second result */

        vmovss 8(%rsp),%xmm0                   /* Fetch x(3) */
        CALL(ENT(__mth_i_tan))                 /* tan(x(3)) */
        vmovss   %xmm0, 24(%rsp)               /* Save third result */

        vmovss 12(%rsp),%xmm0                  /* Fetch x(4) */
        CALL(ENT(__mth_i_tan))                 /* tan(x(4)) */
        vmovss   %xmm0, 28(%rsp)               /* Save fourth result */

        vmovupd  16(%rsp), %xmm0                 /* Put all results in xmm0 */

	addq    $40, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA))


/* -------------------------------------------------------------------------
 *  vector single precision tangent - 256 bit
 *
 *  Prototype:
 *
 *      float * __fvs_tan_[fma4/vex]_256(float *x);
 *
 * ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256)):


        subq    $72, %rsp

        vmovups %ymm0, (%rsp)
#if	! defined(TARGET_WIN_X8664)
	vzeroupper
#endif

	CALL(ENT(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA)))


        vmovups (%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm0
        vmovups %ymm1, 32(%rsp)

	CALL(ENT(ASM_CONCAT(__fvs_tan_,TARGET_VEX_OR_FMA)))

        vmovups 32(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        addq    $72, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvs_tan_,TARGET_VEX_OR_FMA,_256))


/* -------------------------------------------------------------------------
 *  scalar single precision tangent
 *
 *  Prototype:
 *
 *      float __fss_tan_[fma4/vex](float *x);
 *
 * ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fss_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fss_tan_,TARGET_VEX_OR_FMA)):

        subq $8, %rsp

        CALL(ENT(__mth_i_tan))

        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__fss_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fss_tan_,TARGET_VEX_OR_FMA))


/* -------------------------------------------------------------------------
 *  vector double precision tangent - 128 bit
 *
 *  Prototype:
 *
 *      double * __fvd_tan_[fma4/vex](double *x);
 *
 * ------------------------------------------------------------------------- */


        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA)):


	subq    $40, %rsp

        vmovupd  %xmm0, (%rsp)                 /* Save xmm0 */
#if	! defined(TARGET_WIN_X8664)
	vzeroupper
#endif

        CALL(ENT(__mth_i_dtan))                /* tan(x(1)) */
        vmovsd   %xmm0, 16(%rsp)               /* Save first result */

        vmovsd 8(%rsp),%xmm0                   /* Fetch x(2) */
        CALL(ENT(__mth_i_dtan))                /* tan(x(2)) */
        vmovsd   %xmm0, 24(%rsp)               /* Save second result */

        vmovupd  16(%rsp), %xmm0               /* Put all results in xmm0 */

	addq    $40, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA))


/* -------------------------------------------------------------------------
 *  vector double precision tangent - 256 bit
 *
 *  Prototype:
 *
 *      double * __fvd_tan_[fma4/vex]_256(double *x);
 *
 * ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256))
ENT(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256)):


        subq    $72, %rsp

        vmovups %ymm0, (%rsp)
#if	! defined(TARGET_WIN_X8664)
	vzeroupper
#endif

	CALL(ENT(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA)))


        vmovups (%rsp), %ymm2
        vmovaps %xmm0, %xmm1
        vextractf128    $1, %ymm2, %xmm0
        vmovups %ymm1, 32(%rsp)

	CALL(ENT(ASM_CONCAT(__fvd_tan_,TARGET_VEX_OR_FMA)))

        vmovups 32(%rsp), %ymm1
        vinsertf128     $1, %xmm0, %ymm1, %ymm0

        addq    $72, %rsp
        ret

        ELF_FUNC(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256))
        ELF_SIZE(ASM_CONCAT3(__fvd_tan_,TARGET_VEX_OR_FMA,_256))


/* -------------------------------------------------------------------------
 *  scalar double precision tangent
 *
 *  Prototype:
 *
 *      double __fsd_tan_[fma4/vex](double *x);
 *
 * ------------------------------------------------------------------------- */

        .text
        ALN_FUNC
        .globl ENT(ASM_CONCAT(__fsd_tan_,TARGET_VEX_OR_FMA))
ENT(ASM_CONCAT(__fsd_tan_,TARGET_VEX_OR_FMA)):

        subq $8, %rsp

        CALL(ENT(__mth_i_dtan))

        addq $8, %rsp
        ret

        ELF_FUNC(ASM_CONCAT(__fsd_tan_,TARGET_VEX_OR_FMA))
        ELF_SIZE(ASM_CONCAT(__fsd_tan_,TARGET_VEX_OR_FMA))

