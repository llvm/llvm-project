/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


 
#include "directives.h"
#include "x86id.h" 
 
/*
 *   float __mth_i_anint(float f)
 */
	.text
	ALN_FUNC
	.globl	ENT(__mth_i_anint)
ENT(__mth_i_anint):
#if  defined(TARGET_WIN_X8664)
     movl    ENT(X86IDFN(hw_features))(%rip), %eax
#else
     movq    ENT(X86IDFN(hw_features))@GOTPCREL(%rip), %rax
     movl    (%rax), %eax
#endif

1:
	testl	$HW_AVX512F, %eax
	jnz	.Ldo_avx512
	testl	$HW_AVX, %eax
	jnz	.Ldo_avx
	testl	$HW_SSE, %eax
	jnz	.Ldo_sse	// Can't assume do_sse on first pass
	subq	$8, %rsp	// Save %xmm0
	movss	%xmm0, (%rsp)	// Do not use vmovss
	movl	%eax, I1W	// Input to X86IDFN(init_hw_feature)()
	CALL	(ENT(X86IDFN(init_hw_features)))
//	(%eax) = hw_features
	movss	(%rsp), %xmm0	// Restore %xmm0 - do not use vmovss
	addq	$8, %rsp
	jmp	1b		// Restart feature tests

/*
 *	Intel intrinsic equivalent:
 *	#include "immintrin.h"
 *	#define FL_SIGN_BIT 0x80000000
 *	
 *	__m512 __roundf_kernel(__m512 x)
 *	{
 *	    // copysignf(0.5, x);
 *	    //__m512 f05 = _mm512_or_ps(_mm512_and_ps(x, _mm512_set1_ps(I2F(FL_SIGN_BIT))), _mm512_set1_ps(0.5f));   //PRINT(f05);
 *	    __m512 f05 = _mm512_or_ps(_mm512_and_ps(x, _mm512_set1_ps(_castu32_f32(FL_SIGN_BIT))), _mm512_set1_ps(0.5f));   //PRINT(f05);
 *	    // x + 0.5 of the same sign, in round toward zero mode, exceptions suppressed
 *	    __m512 xp05 = _mm512_add_round_ps(x, f05, (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC));                    //PRINT(xp05);
 *	    // truncate, rounding toward zero, exceptions suppressed
 *	    __m512 res = _mm512_roundscale_ps(xp05, (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC));                      //PRINT(res);
 *	    return res;
 *	}
 */

	ALN_FUNC
.Ldo_avx512:
/*
 *	(%zmm1) = 0.5
 */
	vbroadcastss .cnp5_4(%rip), %zmm1
/*
 *	%zmm1 = copysignf(0.5, %zmm0)
 *	imm8 = $0xf8: OR(dest/src1, AND(src2, src3))
 *	Note: use .sgnb_8+4, not .sgnb_8.
 */
	vpternlogd $0xf8, .sgnb_8+4(%rip){1to16}, %zmm0, %zmm1
/*
 *	Round to zero, supress arithmetic exceptions
 */
	vaddps	{rz-sae}, %zmm1, %zmm0, %zmm2
/*
 *	0xb = 1011b
 *	imm8[3], if 1, precision exception not reported
 *	imm8[2], if 0, get rounding from imm8[1:0]
 *	imm8[1:0], if 3, round to nearest smallest magnitude integer
 */
	vrndscaleps $0xb, %zmm2, %zmm0
	ret

	ALN_FUNC
.Ldo_avx:
	vcvtss2sd %xmm0, %xmm0, %xmm0
	vmovsd	.sgnb_8(%rip), %xmm1
	vandpd	%xmm0, %xmm1, %xmm1
	vmovsd	.maxd_8(%rip), %xmm2
	vmovsd	.cnp5_8(%rip), %xmm4
	vmovsd	.mnp5_8(%rip), %xmm3
	vxorpd	%xmm1, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm5
	vaddsd	%xmm2, %xmm5, %xmm5
	vsubsd	%xmm2, %xmm5, %xmm5
	vmovapd	%xmm5, %xmm2
	vsubsd	%xmm0, %xmm2, %xmm2
	vmovapd	%xmm2, %xmm0
	vcmplesd %xmm3, %xmm2, %xmm2
	vmovapd	%xmm4, %xmm3
	vaddsd	%xmm3, %xmm3, %xmm3
	vandpd	%xmm3, %xmm2, %xmm2
	vcmpnlesd %xmm4, %xmm0, %xmm0
	vandpd	%xmm3, %xmm0, %xmm0
	vsubsd	%xmm0, %xmm5, %xmm5
	vaddsd	%xmm2, %xmm5, %xmm5
	vorpd	%xmm1, %xmm5, %xmm5
	vcvtsd2ss %xmm5, %xmm0, %xmm0
	ret

	ALN_FUNC
.Ldo_sse:
	cvtss2sd %xmm0, %xmm0
	movsd	.sgnb_8(%rip), %xmm1
	andpd	%xmm0, %xmm1
	movsd	.maxd_8(%rip), %xmm2
	movsd	.cnp5_8(%rip), %xmm4
	movsd	.mnp5_8(%rip), %xmm3
	xorpd	%xmm1, %xmm0
	movapd	%xmm0, %xmm5
	addsd	%xmm2, %xmm5
	subsd	%xmm2, %xmm5
	movapd	%xmm5, %xmm2
	subsd	%xmm0, %xmm2
	movapd	%xmm2, %xmm0
	cmplesd	%xmm3, %xmm2
	movapd	%xmm4, %xmm3
	addsd	%xmm3, %xmm3
	andpd	%xmm3, %xmm2
	cmpnlesd %xmm4, %xmm0
	andpd	%xmm3, %xmm0
	subsd	%xmm0, %xmm5
	addsd	%xmm2, %xmm5
	orpd	%xmm1, %xmm5
	cvtsd2ss %xmm5, %xmm0
	ret
	ALN_DBLE
.sgnb_8:
	.long 0x00000000,0x80000000
.maxd_8:
	.long 0x00000000,0x43300000
.cnp5_8:
	.long 0x00000000,0x3fe00000
.mnp5_8:
	.long 0x00000000,0xbfe00000
.cnp5_4:
	.long	0x3f000000
	ELF_FUNC(__mth_i_anint)
	ELF_SIZE(__mth_i_anint)
