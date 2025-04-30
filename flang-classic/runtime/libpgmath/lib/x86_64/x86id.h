/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef X86ID_H_
#define X86ID_H_


#define X86IDFN_(l,r) l##r
#define X86IDFN(n) X86IDFN_(__Cpuid_,n)

#define	X86ID_IS_CACHED_UNDEF   (-1)

/*
 * Bit offsets for various X8664 hardware features within X86IDFN(hw_features).
 * If X86IDFN(hw_features) == 0, the variable is undefined and is
 * initialized by calling X86IDFN(init_hw_features)().
 *
 * X86IDFN(hw_features) is intended to be use by runtime routines that have
 * different execution paths depending on hardware characteristics.  In
 * particular different SSE and AVX implementations to avoid some processors'
 * expensive AVX/SSE transition penalties.
 *
 * A prototype assembly pseudo code implementation would be:
 *
 * #if  defined(TARGET_WIN_X8664)
 *      movl    ENT(X86IDFN(hw_features))(%rip), %eax
 * #else
 *      movq    ENT(X86IDFN(is_avx_cached))@GOTPCREL(%rip), %rax
 *      movl    (%rax), %eax
 * #endif
 *
 * 1:
 *      testl   $HW_AVX, %eax
 *      jnz     do_avx
 *      testl   $HW_SSE, %eax
 *      jnz     do_sse          // Can't assume do_sse on first pass
 *      subq    $8, %rsp        // Adjusted for number of local regs to save
 *      movq    I1, (%rsp)      // Or %xmm0, or I1, %xmm0, or %ymm0, ...
 *      movl    %eax, I1W       // Input to X86IDFN(init_hw_feature)()
 *      CALL    (ENT(X86IDFN(init_hw_features)))    // (%eax) = hw_features
 *      movq    (%rsp), I1      // And possibly more regs
 *      addq    $8, %rsp
 *      jmp     1b              // Restart feature tests
 *
 * Note: X86IDFN(init_hw_feature)(X86IDFN(hw_features)) will abort if
 * I1W on entry is the same as the return value.
 *
 */

#define	HW_SSE      0x00000001       // SSE, SSE2, SSE3
#define	HW_SSE4     0x00000002       // SSE4A, SSE41, SSE42
#define	HW_AVX      0x00000004
#define	HW_AVX2     0x00000008
#define	HW_AVX512   0x00000010
#define	HW_AVX512F  0x00000020
#define	HW_AVX512VL 0x00000040
#define	HW_FMA      0x00000080
#define	HW_FMA4     0x00000100
#define	HW_KNL      0x00000200
#define	HW_F16C     0x00000400
#define	HW_SSSE3    0x00000800

#if     ! defined(__ASSEMBLER__)

#include <stdint.h>

#define IS_CONCAT3_(l,m,r)  l##m##r
#define IS_CONCAT3(l,m,r)   IS_CONCAT3_(l,m,r)

#define IS_X86ID(f)                                                           \
    (X86IDFN(IS_CONCAT3(is_,f,_cached)) != X86ID_IS_CACHED_UNDEF) ?           \
        X86IDFN(IS_CONCAT3(is_,f,_cached)) : X86IDFN(IS_CONCAT3(is_,f,))()

/*
 * All the "_cached" varaibles are one of three values:
 * 1) IS_X86ID_CACHED_UNDEF:    not initialized
 * 2) false (0):                initialized and value is false
 * 3) true (1):                 initialized and value is true
 */

/*
 *  For Non-Windows based builds (Linux, OSX), the extern keyword
 *  gives the proper attribute for the global variables is_<FEATURE>_cached.
 *  But for Windows, we need to use MS' __declspec attribute.
 *  When building x86id.c which defines those global variables, we define the
 *  CPP object macro OBJ_WIN_X8664_IS_X86ID.
 */

#if     defined (TARGET_WIN_X8664) && defined(_DLL)
#   if      defined(OBJ_WIN_X8664_IS_X86ID)
#       define  DECLEXTERN  __declspec(dllexport)
#   else
#       define  DECLEXTERN  __declspec(dllimport)
#   endif
#else
#   define  DECLEXTERN  extern
#endif

#ifdef __cplusplus
extern "C" {
#endif
DECLEXTERN  uint32_t    X86IDFN(hw_features);
DECLEXTERN	int X86IDFN(is_intel_cached);
DECLEXTERN	int X86IDFN(is_amd_cached);
DECLEXTERN	int X86IDFN(is_ip6_cached);
DECLEXTERN	int X86IDFN(is_sse_cached);
DECLEXTERN	int X86IDFN(is_sse2_cached);
DECLEXTERN	int X86IDFN(is_sse3_cached);
DECLEXTERN	int X86IDFN(is_ssse3_cached);
DECLEXTERN	int X86IDFN(is_sse4a_cached);
DECLEXTERN	int X86IDFN(is_sse41_cached);
DECLEXTERN	int X86IDFN(is_sse42_cached);
DECLEXTERN	int X86IDFN(is_aes_cached);
DECLEXTERN	int X86IDFN(is_avx_cached);
DECLEXTERN	int X86IDFN(is_avx2_cached);
DECLEXTERN	int X86IDFN(is_avx512_cached);
DECLEXTERN	int X86IDFN(is_avx512f_cached);
DECLEXTERN	int X86IDFN(is_avx512vl_cached);
DECLEXTERN	int X86IDFN(is_fma_cached);
DECLEXTERN	int X86IDFN(is_fma4_cached);
DECLEXTERN	int X86IDFN(is_ht_cached);
DECLEXTERN	int X86IDFN(is_athlon_cached);
DECLEXTERN	int X86IDFN(is_hammer_cached);
DECLEXTERN	int X86IDFN(is_gh_cached);
DECLEXTERN	int X86IDFN(is_gh_a_cached);
DECLEXTERN	int X86IDFN(is_gh_b_cached);
DECLEXTERN	int X86IDFN(is_shanghai_cached);
DECLEXTERN	int X86IDFN(is_istanbul_cached);
DECLEXTERN	int X86IDFN(is_bulldozer_cached);
DECLEXTERN	int X86IDFN(is_piledriver_cached);
DECLEXTERN	int X86IDFN(is_k7_cached);
DECLEXTERN	int X86IDFN(is_ia32e_cached);
DECLEXTERN	int X86IDFN(is_p4_cached);
DECLEXTERN	int X86IDFN(is_knl_cached);
DECLEXTERN	int X86IDFN(is_x86_64_cached);
DECLEXTERN	int X86IDFN(is_f16c_cached);

DECLEXTERN	int X86IDFN(is_intel)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_amd)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_ip6)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_sse)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_sse2)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_sse3)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_ssse3)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_sse4a)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_sse41)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_sse42)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_aes)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_avx)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_avx2)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_avx512)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_avx512f)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_avx512vl)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_fma)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_fma4)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_ht)(void);	/* return 0 .. logical processor count */
DECLEXTERN	int X86IDFN(is_athlon)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_hammer)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_gh)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_gh_a)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_gh_b)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_shanghai)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_istanbul)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_bulldozer)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_piledriver)(void);/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_k7)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_ia32e)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_p4)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_knl)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(is_x86_64)(void);	/* return 0 or 1 */
DECLEXTERN	int X86IDFN(get_cachesize)(void);
DECLEXTERN	int X86IDFN(is_f16c)(void);
DECLEXTERN	char *X86IDFN(get_processor_name)(void);

extern int X86IDFN(get_cores)(void);

#ifdef __cplusplus
}
#endif

#endif          /* ! defined(__ASSEMBLER__) */

#endif /* X86ID_H_ */
/* vim: set ts=4 expandtab: */
