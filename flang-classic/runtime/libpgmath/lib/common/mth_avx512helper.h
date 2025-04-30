
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if ! defined(TARGET_X8664)
#error "mth_avx512helper.h must have TARGET_X8664 defined"
#endif

#ifndef	MTH_AVX512HELPER_H
#define	MTH_AVX512HELPER_H

/*
 * mth_avx512helper.h - helper macros for AVX512.
 *
 * Two objectives:
 * 1)	Provide AVX/AVX2 and SKYLAKE-AVX512 compatibility.
 * 	There are instructions in the AVX/AVX2 extensions that do not exist
 * 	with SKYLAKE-AVX512.  Create macros that provide identical functionality
 * 	to AVX/AVX2 with AVX512 - though using 512-bit registers.
 *
 * 	Example:
 * 	Vector compare in the AVX/AVX2 extensions set a resulting
 * 	vector register with a -1 (32 or 64-bit) where the results of the
 * 	comparison match.  AVX512 uses the K registers for the result of the
 * 	compare. So extend _mm256_cmpeq_epi32(a,b) to _MM512_CMPEQ_EPI32 as:
 *	(__m512i) _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a, b), -1))
 *
 *
 * 2)	Provide KNC and SKYLAKE-AVX512 compatibility.
 *	Another complication is that we currently build to have a common object
 *	between KNL and AVX512F (CPUID flags AVX512F for AVX-512, KNCNI for KNC)
 *	thus AVX512 instructions.
 *
 * 	Example:
 * 	The KNC extensions do not have a "floating point" boolean AND()
 * 	instruciton.
 * 	Extend _mm512_and_ps(a,b) to _MM512_AND_PS as:
 *	(__m512) _mm512_and_si512(a, _mm512_castps_si512(b))
 *
 *	Macro FCN_AVX512(name) is used to create unique (entry point) names
 *	based upon the extensions "_knl" or "_512" depending whether KNL is
 *	targetted or not.
 *
 *	Note: Not every possible AVX/AVX2 intrinsic is currently defined.
 *	They helper macros are created as needed in porting the FMA3 version
 *	of the math intrinsics source code.
 */

/*
 * The following macros are used to have a common source between KNL and
 * SKYLAKE-AVX512.
 */

#if	defined(__knl) || defined (__knl__)
/*
 * 	KNL implementations.
 */
#define	FCN_AVX512(a)                                                         \
	a##_knl

#define	_MM512_AND_PD(a, b)                                                   \
	(__m512d) _mm512_and_si512(_mm512_castpd_si512(a),                    \
				   _mm512_castpd_si512(b))

#define	_MM512_AND_PS(a, b)                                                   \
	(__m512) _mm512_and_si512(_mm512_castps_si512(a),                     \
				  _mm512_castps_si512(b))

#define	_MM512_ANDNOT_PD(a, b)                                                \
	(__m512d) _mm512_andnot_si512(_mm512_castpd_si512(a),                 \
				      _mm512_castpd_si512(b))

#define	_MM512_ANDNOT_PS(a, b)                                                \
	(__m512) _mm512_andnot_si512(_mm512_castps_si512(a),                  \
				     _mm512_castps_si512(b))

#define	_MM512_OR_PD(a, b)                                                    \
	(__m512d) _mm512_or_si512(_mm512_castpd_si512(a),                     \
				  _mm512_castpd_si512(b))

#define	_MM512_OR_PS(a, b)                                                    \
	(__m512) _mm512_or_si512(_mm512_castps_si512(a),                      \
				 _mm512_castps_si512(b))

#define	_MM512_XOR_PD(a, b)                                                   \
	(__m512d) _mm512_xor_si512(_mm512_castpd_si512(a),                    \
				   _mm512_castpd_si512(b))

#define	_MM512_XOR_PS(a, b)                                                   \
	(__m512) _mm512_xor_si512(_mm512_castps_si512(a),                     \
				  _mm512_castps_si512(b))

#define	_MM512_EXTRACTF256_PS(a,b)                                            \
	(__m256) _mm512_extractf64x4_pd(_mm512_castps_pd(a),b)

#define	_MM512_INSERTF256_PS(a,b,c)                                           \
	 (__m512) _mm512_insertf64x4(_mm512_castps_pd(a),                     \
				     _mm256_castps_pd(b),c)

#define	_MM512_EXTRACTI256_SI512(a,b)                                         \
	_mm512_extracti64x4_epi64(a,b)

#define	_MM512_MOVM_EPI32(a)                                                  \
	_mm512_maskz_set1_epi32(a,-1)

#define	_MM512_MOVM_EPI64(a)                                                  \
	_mm512_maskz_set1_epi64(a,-1)

#else		// #if	defined(__knl) || defined (__knl__)
/*
 * 	SKYLAKE-AVX512 implementations.
 */
#define	FCN_AVX512(a)                                                         \
	a##_512

#define	_MM512_AND_PS(a, b)                                                   \
	_mm512_and_ps(a, b)

#define	_MM512_AND_PD(a, b)                                                   \
	_mm512_and_pd(a, b)

#define	_MM512_ANDNOT_PS(a, b)                                                \
	_mm512_andnot_ps(a, b)

#define	_MM512_ANDNOT_PD(a, b)                                                \
	_mm512_andnot_pd(a, b)

#define	_MM512_OR_PS(a, b)                                                    \
	_mm512_or_ps(a, b)

#define	_MM512_OR_PD(a, b)                                                    \
	_mm512_or_pd(a, b)

#define	_MM512_XOR_PS(a, b)                                                   \
	_mm512_xor_ps(a, b)

#define	_MM512_XOR_PD(a, b)                                                   \
	_mm512_xor_pd(a, b)

#define	_MM512_EXTRACTF256_PS(a,b)                                            \
	(__m256) _mm512_extractf32x8_ps(a,b)

#define	_MM512_INSERTF256_PS(a,b,c)                                           \
	 _mm512_insertf32x8(a,b,c)

#define	_MM512_EXTRACTI256_SI512(a,b)                                         \
	_mm512_extracti32x8_epi32(a,b)

#define	_MM512_MOVM_EPI32(a)                                                  \
	_mm512_movm_epi32(a)

#define	_MM512_MOVM_EPI64(a)                                                  \
	_mm512_movm_epi64(a)
#endif		// #if	defined(__knl) || defined (__knl__)


/*
 * The following macros are used to provide 512-bit compatibility with
 * intrinsics that only exist with AVX/AVX2.
 */

#define	_MM512_CMPEQ_EPI32(a, b)                                              \
	_MM512_MOVM_EPI32(_mm512_cmpeq_epi32_mask(a, b))

#define	_MM512_CMPEQ_PD(a, b)                                                 \
	_MM512_CMP_PD(a, b, _CMP_EQ_OQ)

#define	_MM512_CMPGT_EPI32(a, b)                                              \
	_MM512_MOVM_EPI32(_mm512_cmpgt_epi32_mask(a, b))

#define	_MM512_CMPEQ_EPI64(a, b)                                              \
	_MM512_MOVM_EPI64(_mm512_cmpeq_epi64_mask(a, b))

#define	_MM512_CMP_PS(a, b, c)                                                \
	(__m512) _MM512_MOVM_EPI32(_mm512_cmp_ps_mask(a, b, c))

#define	_MM512_CMP_PD(a, b, c)                                                \
	(__m512d) _MM512_MOVM_EPI64(_mm512_cmp_pd_mask(a, b, c))

#define	_MM512_BLEND_EPI32(a,b,m)                                             \
	_mm512_mask_blend_epi32(m,a,b)

#define	_MM512_BLEND_EPI64(a,b,m)                                             \
	_mm512_mask_blend_epi64(m,a,b)

#define	_MM512_BLENDV_PS(a,b,m)                                               \
	(__m512) _mm512_ternarylogic_epi32(                                   \
		_mm512_castps_si512(a),                                       \
		_mm512_castps_si512(b),                                       \
		_mm512_srai_epi32(_mm512_castps_si512(m), 31),                \
		0xd8)

#define	_MM512_BLENDV_PD(a,b,m)                                               \
	(__m512d) _mm512_ternarylogic_epi64(                                  \
		_mm512_castpd_si512(a),                                       \
		_mm512_castpd_si512(b),                                       \
		_mm512_srai_epi64(_mm512_castpd_si512(m), 63),                \
		0xd8)

#define	_MM512_MOVEMASK_EPI32(a)                                              \
	(int) _mm512_cmpneq_epi32_mask(_mm512_setzero_si512(),                \
		_mm512_and_si512(_mm512_set1_epi32(0x80000000U), a))

#define	_MM512_MOVEMASK_EPI64(a)                                              \
	(int) _mm512_cmpneq_epi64_mask(_mm512_setzero_si512(),                \
		_mm512_and_si512(_mm512_set1_epi64(0x8000000000000000ULL), a))

#define	_MM512_MOVEMASK_PS(a)                                                 \
	_MM512_MOVEMASK_EPI32(_mm512_castps_si512(a))

#define	_MM512_MOVEMASK_PD(a)                                                 \
	_MM512_MOVEMASK_EPI64(_mm512_castpd_si512(a))

#define	_MM512_ROUND_PD(a,b)                                                  \
	_mm512_roundscale_pd(a,((0<<4)|b|_MM_FROUND_NO_EXC))

#endif		// #ifndef	MTH_AVX512HELPER_H
