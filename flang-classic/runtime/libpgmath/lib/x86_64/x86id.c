/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#if     defined(TARGET_WIN_X8664)
#   if      defined(OBJ_WIN_X8664_IS_X86ID)
#       error   object macro OBJ_WIN_X8664_IS_X86ID cannot already be defined
#   else
#       define  OBJ_WIN_X8664_IS_X86ID
#   endif
#endif

#include "pgcpuid.h"
#include "x86id.h"

/*
 * Define some interesting fields in the extended control register[0].
 * xcr[0] only defines the lower 32-bits of the 64-bit register.
 */

#define	xcr0_bit_XMM	0x00000002U
#define	xcr0_bit_YMM	0x00000004U
#define	xcr0_bit_ZMMK	0x00000020U
#define	xcr0_bit_ZMMLO 	0x00000040U
#define	xcr0_bit_ZMMHI 	0x00000080U

#define	xcr0_mask_YMM	(xcr0_bit_XMM | xcr0_bit_YMM)
#define	xcr0_mask_ZMM	(xcr0_bit_ZMMK | xcr0_bit_ZMMLO | xcr0_bit_ZMMHI)


#define signature_AMD_ebx	0x68747541
#define signature_AMD_ecx	0x444d4163
#define signature_AMD_edx	0x69746e65

#define signature_INTEL_ebx	0x756e6547
#define signature_INTEL_ecx	0x6c65746e
#define signature_INTEL_edx	0x49656e69

//#define DEBUG
#if     defined(DEBUG)
#include    <string.h>
/* use DEBUG_PRINTF(format %s with any arguments %d but no endline",
 *              string, integer );
 */
#define DEBUG_PRINTF(...)                                            \
    do { fputs(__func__, stdout);                                    \
         fputs(strlen(__func__) > 7 ? ":\t" : ":\t\t", stdout);      \
         printf(__VA_ARGS__);                                        \
         fputs("\n", stdout); } while (0)
#else
#define DEBUG_PRINTF(...)
#endif

/*
 * prototypes for the test functions here
 */
static int ia_cachesize(void);
static int ia_unifiedcache(void);
static int amd_cachesize(void);
static int ia_cores(void);
static int amd_cores(void);
static int is_xcr_set(uint32_t, uint64_t);
static int is_amd_family(uint32_t, uint32_t *);

/*
 * Various routines in the runtime libraries are needing to detect what processor type/model/feature
 * they are running on.  Instead of using relatively heavy weight routines to return that information,
 * provide a mechanism to cache the data.
 *
 * The "X86IDFN(is_<TYPE/MODEL/FEATURE>)" is the routine that is called, cache that info in global
 * variable X86IDFN(is_<TYPE/MODEL/FEATURE>_cached).
 *
 * Use macro IS_X86ID(<
 */

uint32_t X86IDFN(hw_features)    	= 0;
int X86IDFN(is_intel_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_amd_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_ip6_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_sse_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_sse2_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_sse3_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_ssse3_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_sse4a_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_sse41_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_sse42_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_aes_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_avx_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_avx2_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_avx512_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_avx512f_cached)  	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_avx512vl_cached) 	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_fma_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_fma4_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_ht_cached)       	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_athlon_cached)   	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_hammer_cached)   	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_gh_cached)       	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_gh_a_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_gh_b_cached)     	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_shanghai_cached) 	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_istanbul_cached) 	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_bulldozer_cached) 	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_piledriver_cached) 	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_k7_cached)       	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_ia32e_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_p4_cached)       	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_knl_cached)      	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_x86_64_cached)    	= X86ID_IS_CACHED_UNDEF;
int X86IDFN(is_f16c_cached)      	= X86ID_IS_CACHED_UNDEF;

/*
 * Return whether extended control register has requested bits set.
 * Assumes that the processor has the xgetbv instruction.
 * Return:  0 == register does not have bit(s) set or __pgi_getbv() failed.
 *          1 == bits set.
 */

static
int is_xcr_set(uint32_t xcr_indx, uint64_t xcr_mask)
{
    uint64_t xcr;

    if( __pgi_getbv( xcr_indx, &xcr ) == 0 ) {
        DEBUG_PRINTF("_pgi_getbv() failed xcr_indx=%#8.8x, "
            "xcr_mask=%#16.16lx", xcr_indx, xcr_mask);
        return 0;
    }

    DEBUG_PRINTF("xcr[%u]=%#16.16x, xcr_mask=%#16.16lx",
        xcr_indx, xcr, xcr_mask);
    return (xcr & xcr_mask) == xcr_mask;
}

/*
 * cache values returned from __pgi_cpuid.
 * cpuid instructions on Windows are costly
 */
int
X86IDFN(idcache)(uint32_t f, uint32_t *r)
{
  int j, rv = 1;
  static struct{
    int set;
    uint32_t f;
    uint32_t i[4];
  }saved[] = {
    { 0, 0U, { 0, 0, 0, 0}}, //
    { 0, 1U, { 0, 0, 0, 0}}, //
    { 0, 2U, { 0, 0, 0, 0}}, //
    { 0, 0x80000000U, { 0, 0, 0, 0}}, //
    { 0, 0x80000001U, { 0, 0, 0, 0}}, //
    { 0, 0x80000002U, { 0, 0, 0, 0}}, //
    { 0, 0x80000003U, { 0, 0, 0, 0}}, //
    { 0, 0x80000004U, { 0, 0, 0, 0}}, //
    { 0, 0x80000006U, { 0, 0, 0, 0}}, //
    { 0, 0x80000008U, { 0, 0, 0, 0}}, //
    {-1, 0U, { 0, 0, 0, 0}} };
  for (j = 0; saved[j].set >= 0; ++j) {
    if (saved[j].f == f) {
      if (!saved[j].set) {
        /* call cpuid once, save its value */
        rv = __pgi_cpuid(f, saved[j].i);
        saved[j].set = 1;
      }
      /* return the saved value */
      r[0] = saved[j].i[0];
      r[1] = saved[j].i[1];
      r[2] = saved[j].i[2];
      r[3] = saved[j].i[3];
      break;
    } else if (saved[j].set == -1) {
      /* we're not caching this value */
      rv = __pgi_cpuid(f, r);
      break;
    }
  }
  return rv;
}

/*
 * is_amd_family(uint32_t family, uint32_t * model)
 * Return true if processor is AMD and of specific family.
 * Always return model.
 */

static
int is_amd_family(uint32_t family, uint32_t *model)
{
    ACPU1 c1;

    if ((X86IDFN(is_amd)() == 0) || (X86IDFN(idcache)( 1, c1.i ) == 0)) {
        return 0;
    }

    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    *model = c1.u.eax.model;
    return ( c1.u.eax.family == family);
}

/*
 * Check that this is a Genuine Intel processor
 */
int
X86IDFN(is_intel)(void)
// is_intel:       eax 0x00000014 ebx 0x756e6547 ecx 0x6c65746e edx 0x49656e69
// is_intel:       eax 0x00000014 ebx 0x756e6547 ecx 0x49656e69 edx 0x6c65746e
{
    unsigned int h;
    CPU0 c0;
    X86IDFN(idcache)( 0, c0.i );
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c0.i[0], c0.i[1], c0.i[2], c0.i[3] );
    X86IDFN(is_intel_cached) =
        ((signature_INTEL_ebx ^ c0.i[1]) |
         (signature_INTEL_ecx ^ c0.i[2]) |
         (signature_INTEL_edx ^ c0.i[3])) == 0;
    return X86IDFN(is_intel_cached);
}/* is_intel */

/*
 * Check that this is an Authentic AMD processor
 */
int
X86IDFN(is_amd)(void)
{
    CPU0 c0;
    unsigned int h;
    X86IDFN(idcache)( 0, c0.i );
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c0.i[0], c0.i[1], c0.i[2], c0.i[3] );
    X86IDFN(is_amd_cached) =
        ((signature_AMD_ebx ^ c0.i[1]) |
         (signature_AMD_ecx ^ c0.i[2]) |
         (signature_AMD_edx ^ c0.i[3])) == 0;
    return X86IDFN(is_amd_cached);
}/* is_amd */

/*
 * test(p6)
 *  either manufacturer
 *  cpuid(1) returns fpu and cmov flag, then must be at least p6
 */
int
X86IDFN(is_ip6)(void)
{
    ICPU1 c1;

    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
	return X86IDFN(is_ip6_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
        c1.i[0], c1.i[1], c1.i[2], c1.i[3] );

    return X86IDFN(is_ip6_cached) = ( c1.u.edx.fpu && c1.u.edx.cmov );
}/* is_ip6 */

/*
 * test(sse)
 *  call with either AMD or Intel
 *  test sse bit, same bit for either manufacturer
 */
int
X86IDFN(is_sse)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_sse_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_sse_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_sse_cached) = ( c1.u.edx.sse != 0);
}/* is_sse */

/*
 * test(sse2)
 *  call with either AMD or Intel
 *  test sse2 bit, same bit for either manufacturer
 */
int
X86IDFN(is_sse2)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_sse2_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_sse2_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_sse2_cached) = ( c1.u.edx.sse2 != 0);
}/* is_sse2 */

/*
 * test(sse3)
 *  call with either AMD or Intel
 */
int
X86IDFN(is_sse3)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_sse3_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_sse3_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_sse3_cached) = ( c1.u.ecx.sse3 != 0);
}/* is_sse3 */

/*
 * test(ssse3)
 *  call with either AMD or Intel
 *  test ssse3 bit, same bit for either manufacturer
 */
int
X86IDFN(is_ssse3)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_ssse3_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_ssse3_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
        return X86IDFN(is_ssse3_cached) = ( c1.u.ecx.ssse3 != 0);
}/* is_ssse3 */

/*
 * test(sse4a)
 *  right now, it's just the greyhound check
 */
int
X86IDFN(is_sse4a)(void)
{
    CPU80 c80;
    ACPU81 c81;
    if( !X86IDFN(is_amd)() )
        return X86IDFN(is_sse4a_cached) = 0;
    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
        return X86IDFN(is_sse4a_cached) = 0;
    if( c80.b.largest < 0x80000001 )
        return X86IDFN(is_sse4a_cached) = 0;
    if( X86IDFN(idcache)( 0x80000001, c81.i ) == 0 )
        return X86IDFN(is_sse4a_cached) = 0;
    return X86IDFN(is_sse4a_cached) = ( c81.u.ecx.sse4a != 0);
}/* is_sse4a */

/*
 * test(sse41)
 *  right now, it's just the penryn check
 */
int
X86IDFN(is_sse41)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() )
        return X86IDFN(is_sse41_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_sse41_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_sse41_cached) = ( c1.u.ecx.sse41 != 0);
}/* is_sse41 */

/*
 * test(sse42)
 */
int
X86IDFN(is_sse42)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_sse42_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_sse42_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_sse42_cached) = ( c1.u.ecx.sse42 != 0);
}/* is_sse42 */

/*
 * test(aes)
 */
int
X86IDFN(is_aes)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_aes_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_aes_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_aes_cached) = ( c1.u.ecx.aes != 0);
}/* is_aes */

/*
 * test(avx)
 */
int
X86IDFN(is_avx)(void)
{
    ICPU1 c1;
    
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_avx_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_avx_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    if( !c1.u.ecx.avx )
        return X86IDFN(is_avx_cached) = 0;
    /* see whether the OS will save the ymm state */
    if( !c1.u.ecx.osxsave )
        return X86IDFN(is_avx_cached) = 0;

    return X86IDFN(is_avx_cached) = is_xcr_set(0, xcr0_mask_YMM);
}/* is_avx */


/*
 * test(avx2)
 */
int
X86IDFN(is_avx2)(void)
{
    ICPU7 c7;
    
    if ( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_avx2_cached) = 0;

    if ( !X86IDFN(is_avx)() )
        return X86IDFN(is_avx2_cached) = 0;

    if ( __pgi_cpuid_ecx( 7, c7.i, 0 ) == 0 )
        return X86IDFN(is_avx2_cached) = 0;

    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	  c7.i[0], c7.i[1], c7.i[2], c7.i[3] );

    return X86IDFN(is_avx2_cached) = (c7.u.ebx.avx2 != 0);
}/* is_avx2 */

/*
 * test(avx512)
 * Determine whether processor and O/S support AVX512.
 */
int
X86IDFN(is_avx512)(void)
{
    if( !X86IDFN(is_intel)() )
        return X86IDFN(is_avx512_cached) = 0;

    if ( !X86IDFN(is_avx)() )
        return X86IDFN(is_avx512_cached) = 0;

    return X86IDFN(is_avx512_cached) = is_xcr_set(0, xcr0_mask_ZMM);
}

/*
 * test(avx512f)
 */
int
X86IDFN(is_avx512f)(void)
{
    ICPU7 c7;
    
    if ( !X86IDFN(is_avx512)() )
        return X86IDFN(is_avx512f_cached) = 0;
    if( __pgi_cpuid_ecx( 7, c7.i, 0 ) == 0 )
        return X86IDFN(is_avx512f_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c7.i[0], c7.i[1], c7.i[2], c7.i[3] );
    return X86IDFN(is_avx512f_cached) = ( c7.u.ebx.avx512f != 0);
}/* is_avx512f */

/*
 * test(avx512vl)
 */
int
X86IDFN(is_avx512vl)(void)
{
    ICPU7 c7;
    
    if( !X86IDFN(is_avx512f)() )
        return X86IDFN(is_avx512vl_cached) = 0;
    if( __pgi_cpuid_ecx( 7, c7.i, 0 ) == 0 )
        return X86IDFN(is_avx512vl_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c7.i[0], c7.i[1], c7.i[2], c7.i[3] );
    return X86IDFN(is_avx512vl_cached) = ( c7.u.ebx.avx512vl != 0);
}/* is_avx51vlf */

/*
 * test(f16c)
 */
int
X86IDFN(is_f16c)(void)
{
    ICPU1 c1;
    
    if ( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_f16c_cached) = 0;

    if ( !X86IDFN(is_avx)() )
        return X86IDFN(is_f16c_cached) = 0;

    if ( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_f16c_cached) = 0;

    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	  c7.i[0], c7.i[1], c7.i[2], c7.i[3] );

    return X86IDFN(is_f16c_cached) = (c1.u.ecx.f16c != 0);
}/* is_f16c */

/*
 * test(fma)
 */
int
X86IDFN(is_fma)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() && !X86IDFN(is_amd)() )
        return X86IDFN(is_fma_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_fma_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_fma_cached) = ( c1.u.ecx.fma != 0);
}/* is_fma */

/*
 * test(fma4)
 */
int
X86IDFN(is_fma4)(void)
{
    CPU80 c80;
    ACPU81 c81;
    if( !X86IDFN(is_amd)() )
        return X86IDFN(is_fma4_cached) = 0;
    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
        return X86IDFN(is_fma4_cached) = 0;
    if( c80.b.largest < 0x80000001 )
        return X86IDFN(is_fma4_cached) = 0;
    if( X86IDFN(idcache)( 0x80000001, c81.i ) == 0 )
        return X86IDFN(is_fma4_cached) = 0;
    return X86IDFN(is_fma4_cached) = ( c81.u.ecx.fma4 != 0);
}/* is_fma4 */

/*
 * test(ht)
 *  call with Intel
 *  test sse3 bit, same bit for either manufacturer
 */
int
X86IDFN(is_ht)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() )
        return X86IDFN(is_ht_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_ht_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    if( c1.u.edx.htt )
        return X86IDFN(is_ht_cached) =  c1.u.ebx.proccount;
    return X86IDFN(is_ht_cached) = 0;
}/* is_ht */

/*
 * test(athlon)
 *  test AMD
 *  test family==15, or model == 1,2,4,6
 */
int
X86IDFN(is_athlon)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_amd)() )
        return X86IDFN(is_athlon_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_athlon_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    if( c1.u.eax.family == 15 )
        return X86IDFN(is_athlon_cached) = 1;
    if( c1.u.eax.family != 6 )
        return X86IDFN(is_athlon_cached) = 0;
    switch( c1.u.eax.model ){
    case 1 :
    case 2 :
    case 4 :
    case 6 :
        return X86IDFN(is_athlon_cached) = 1;
    }
    return X86IDFN(is_athlon_cached) = 0;
}/* is_athlon */

/*
 * test(hammer)
 *  test for AMD
 *  test for family == 15
 */
int
X86IDFN(is_hammer)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_amd)() )
        return X86IDFN(is_hammer_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_hammer_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_hammer_cached) = ( c1.u.eax.family == 15 );
}/* is_hammer */

/*
 * test(gh)
 *  test for AMD
 *  test for family == 16
 */
int
X86IDFN(is_gh)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_amd)() )
        return X86IDFN(is_gh_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_gh_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_gh_cached) = ( c1.u.eax.family == 15 && c1.u.eax.extfamily == 1);
}/* is_gh */

/*
 * test(gh-a)
 *  test for gh
 *  test for model == 0
 */
int
X86IDFN(is_gh_a)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_gh)() )
        return X86IDFN(is_gh_a_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_gh_a_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_gh_a_cached) = ( c1.u.eax.model == 0 );
}/* is_gh_a */

/*
 * test(gh-b)
 *  test for gh
 *  test for model == 1
 */

/*
 *
 * Code from rte/pgc/hammer/src/cpuinfo.c
 *
 * {
 *   CPUID c1;
 *   CPUMODEL m1;
 *   ACPU81 c81;
 * 
 *   if (!__pgi_is_gh())
 *     return 0;
 * 
 *   if (X86IDFN(idcache)(1, c1.i) == 0)
 *     return 0;
 * 
 *   m1.i = c1.reg.eax;
 * 
 *   if (m1.bits.model >= 2) {
 *     if (X86IDFN(idcache)(0x80000001, c81.i) == 0)
 *       return 0;
 *     if (c81.u.ecx.mas) {
 *       return 1;
 *     }
 *   }
 * 
 *   return 0;
 * }
 */

int
X86IDFN(is_gh_b)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_gh)() )
        return X86IDFN(is_gh_b_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_gh_b_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_gh_b_cached) = ( c1.u.eax.model >= 2 );
}/* is_gh_b */

/*
 * test(shanghai)
 *  test for shanghai
 *  test for is a gh, and cache size >= 6MB
 */
int
X86IDFN(is_shanghai)(void)
{
    CPU80 c80;
    ACPU86 c86;
    if( !X86IDFN(is_gh)() )
        return X86IDFN(is_shanghai_cached) = 0;
    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
        return X86IDFN(is_shanghai_cached) = 0;
    if( c80.b.largest < 0x80000006U )
        return X86IDFN(is_shanghai_cached) = 0;
    if( X86IDFN(idcache)( 0x80000006U, c86.i ) == 0 )
        return X86IDFN(is_shanghai_cached) = 0;
    return X86IDFN(is_shanghai_cached) = ( c86.u.l3cache.size >= 6 );
}/* is_shanghai */

/*
 * test(istanbul)
 *  test for istanbul
 *  test for is a shanghai, and model > 4
 */
int
X86IDFN(is_istanbul)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_shanghai)() )
        return X86IDFN(is_istanbul_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_istanbul_cached) = 0;
    return X86IDFN(is_istanbul_cached) = ( c1.u.eax.model > 4 );
}/* is_istanbul */


/*
 * test(bulldozer)
 *  test for bulldozer
 *  test for family == 21
 */
int
X86IDFN(is_bulldozer)(void)
{
    ACPU1 c1;

    if ( (X86IDFN(is_amd)() == 0) || (X86IDFN(idcache)( 1, c1.i ) == 0)) {
        return X86IDFN(is_bulldozer_cached) = 0;
    }
    DEBUG_PRINTF("eax %8.8x ebx %8.8x ecx %8.8x edx %8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_bulldozer_cached) = ( c1.u.eax.family == 15 && c1.u.eax.extfamily == 6);
}/* is_bulldozer */

/*
 * test(piledriver)
 *  test for bulldozer & fma
 */
int
X86IDFN(is_piledriver)(void)
{
    return X86IDFN(is_piledriver_cached) = ( X86IDFN(is_bulldozer)() && X86IDFN(is_fma)() );
}/* is_piledriver */

/*
 * test(k7)
 *  test AMD
 *  test family == 6
 */
int
X86IDFN(is_k7)(void)
{
    ACPU1 c1;
    if( !X86IDFN(is_amd)() )
        return X86IDFN(is_k7_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_k7_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_k7_cached) = ( c1.u.eax.family == 6 );
}/* is_k7 */

/*
 * test(ia32e)
 *  test Intel
 *  test family == 15 and lm
 */
int
X86IDFN(is_ia32e)(void)
{
    ICPU1 c1;
    CPU80 c80;
    ICPU81 c81;
    if( !X86IDFN(is_intel)() )
        return X86IDFN(is_ia32e_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_ia32e_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    if( c1.u.eax.family != 15 )
        return X86IDFN(is_ia32e_cached) = 0;
    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
        return X86IDFN(is_ia32e_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x", c80.i[0] );
    if( c80.b.largest < 0x80000001 )
        return X86IDFN(is_ia32e_cached) = 0; /* no extended flags */
    if( X86IDFN(idcache)( 0x80000001, c81.i ) == 0 )
        return X86IDFN(is_ia32e_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c81.i[0], c81.i[1], c81.i[2], c81.i[3] );
    return X86IDFN(is_ia32e_cached) = ( c81.u.edx.lm != 0);
}/* is_ia32e */

/*
 * test(p4)
 *  test Intel
 *  test family == 15
 */
int
X86IDFN(is_p4)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() )
        return X86IDFN(is_p4_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_p4_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    return X86IDFN(is_p4_cached) = ( c1.u.eax.family == 15 );
}/* is_p4 */

/*
 * test(knl)
 *  test Intel
 *  test family == 6 && model == 0x57
 */
int
X86IDFN(is_knl)(void)
{
    ICPU1 c1;
    if( !X86IDFN(is_intel)() )
        return X86IDFN(is_knl_cached) = 0;
    if( X86IDFN(idcache)( 1, c1.i ) == 0 )
        return X86IDFN(is_knl_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c1.i[0], c1.i[1], c1.i[2], c1.i[3] );
    if( c1.u.eax.family == 6 ){
        int model = ((int)c1.u.eax.extmodel << 4) + (int)c1.u.eax.model;
	    return X86IDFN(is_knl_cached) = ( model == 0x57 );
    }
    return X86IDFN(is_knl_cached) = 0;
}/* is_knl */

/*
 * either manufacturer
 * test for lm flag in extended features
 */
int
X86IDFN(is_x86_64)(void)
{
    CPU80 c80;
    ICPU81 c81;

    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
        return X86IDFN(is_x86_64_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x", c80.i[0] );
    if( c80.b.largest < 0x80000001 )
        return X86IDFN(is_x86_64_cached) = 0;
    if( X86IDFN(idcache)( 0x80000001, c81.i ) == 0 )
        return X86IDFN(is_x86_64_cached) = 0;
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c81.i[0], c81.i[1], c81.i[2], c81.i[3] );
    return X86IDFN(is_x86_64_cached) = ( c81.u.edx.lm != 0);
}/* is_x86_64 */

/*
 * Initialize global variable X86IDFN(hw_features).
 */

uint32_t
X86IDFN(init_hw_features)(uint32_t old_hw_features)
{
    if (X86IDFN(is_sse3)()) {       // Implies SSE, SSE2, SSE3
        X86IDFN(hw_features) |= HW_SSE;
    }

    /*
     * AMD processors with SSE4A does not necessarily imply support for SSSE3.
     */
    if (X86IDFN(is_ssse3)()) {
        X86IDFN(hw_features) |= HW_SSSE3;
    }

    if (X86IDFN(is_sse42)()) {      // Implies SSE4A, SSE41, and SSE42
        X86IDFN(hw_features) |= HW_SSE4;
    }

    if (X86IDFN(is_avx)()) {
        X86IDFN(hw_features) |= HW_AVX;
    }

    if (X86IDFN(is_avx2)()) {
        X86IDFN(hw_features) |= HW_AVX2;
    }

    if (X86IDFN(is_avx512)()) {
        X86IDFN(hw_features) |= HW_AVX512;
    }

    if (X86IDFN(is_avx512f)()) {
        X86IDFN(hw_features) |= HW_AVX512F;
    }

    if (X86IDFN(is_avx512vl)()) {
        X86IDFN(hw_features) |= HW_AVX512VL;
    }

    if (X86IDFN(is_fma)()) {
        X86IDFN(hw_features) |= HW_FMA;
    }

    if (X86IDFN(is_fma4)()) {
        X86IDFN(hw_features) |= HW_FMA4;
    }

    if (X86IDFN(is_knl)()) {
        X86IDFN(hw_features) |= HW_KNL;
    }

    if (X86IDFN(is_f16c)()) {
        X86IDFN(hw_features) |= HW_F16C;
    }

    if (old_hw_features != X86IDFN(hw_features)) {
        return X86IDFN(hw_features);
    }

    /*
     * Either the processor does not have at a minimum SSE3 support, or
     * this routine has been now called twice with same input argument.
     * Abort and avoid infinite loop since nothing is going to change.
     */

#if defined(TARGET_WIN_X8664) && ! defined(_NO_CRT_STDIO_INLINE)
    /*
     * Exception! Windows - building x86id.obj for libcpuid.lib:
     * It is unclear why fprintf() can't be used when x86id.c is being
     * compiled for libcpuid.lib.
     */

    printf("Error: %s called twice with hw_features=%#x\n", __func__,
        X86IDFN(hw_features));
#else
    // All other architectures/platforms/libraries can safely use fprintf().
    fprintf(stderr, "Error: %s called twice with hw_features=%#x\n", __func__,
        X86IDFN(hw_features));
#endif
    exit(EXIT_FAILURE);     // XXX XXX - should be __abort(1, "some string");

}/* init_hw_features */

/*
 * Locally defined functions.
 */


/*
 * for Intel processors, the values returned by cpuid(2)
 * are an encoding of the cache size, as below
 * other values encode TLB sizes, etc.
 */
static int
ia_cachecode( int code )
{
    switch( code ){
    case 0x39:
    case 0x3b:
    case 0x41:
    case 0x79:
    case 0x81:
	return 128*1024; /*"128KB L2 cache"*/
    case 0x3c:
    case 0x42:
    case 0x7a:
    case 0x82:
	return 256*1024; /*"256KB L2 cache"*/
    case 0x43:
    case 0x7b:
    case 0x7f:
    case 0x83:
    case 0x86:
	return 512*1024; /*"512KB L2 cache"*/
    case 0x44:
    case 0x7c:
    case 0x84:
    case 0x87:
	return 1024*1024; /*"1MB L2 cache"*/
    case 0x45:
    case 0x7d:
    case 0x85:
	return 2048*1024; /*"2MB L2 cache"*/
    case 0x4e:
	return 6*1024*1024; /*"6MB L2 cache"*/
    case 0xe4:
	return 8*1024*1024; /*"8MB L3 cache"*/
    }
    return 0;
}/* ia_cachecode */

/*
 * return cache size for Intel processors
 */
static int
ia_cachesize(void)
{
    CPU0 c0;
    ICPU2 c2;
    CPU80 c80;
    ICPU86 c86;
    ICPU4 c4;
    int i, n, r;

    if( X86IDFN(idcache)( 0, c0.i ) == 0 )
	return 0;
    if (c0.b.largest >= 4) {
	r = ia_unifiedcache();
	if (r) {
	    return r;
	}
    }
    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
	return 0;
    DEBUG_PRINTF("eax %#8.8x", c80.i[0] );
    if( c80.b.largest >= 0x80000006 ){
	if( X86IDFN(idcache)( 0x80000006, c86.i ) ){
	    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
		c86.i[0], c86.i[1], c86.i[2], c86.i[3] );
	    return c86.u.ecx.size * 1024;
	}
    }

    DEBUG_PRINTF("largest=%d", c0.b.largest );

    if( c0.b.largest < 2 )
	return 0;

    X86IDFN(idcache)( 2, c2.i );
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c2.i[0], c2.i[1], c2.i[2], c2.i[3] );
    n = c2.u[0].c1;
    while( n-- ){
	for( i = 0; i < 4; ++i ){
	    if( c2.u[i].invalid == 0 ){
		if( i > 0 ){	/* 1st byte in eax is something else */
		    r = ia_cachecode( c2.u[i].c1 );
		    if( r )
			return r;
		}
		r = ia_cachecode( c2.u[i].c2 );
		if( r )
		    return r;
		r = ia_cachecode( c2.u[i].c3 );
		if( r )
		    return r;
		r = ia_cachecode( c2.u[i].c4 );
		if( r )
		    return r;
	    }
	}
	X86IDFN(idcache)( 2, c2.i );
	DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c2.i[0], c2.i[1], c2.i[2], c2.i[3] );
    }
    return 0;
}/* ia_cachesize */

static int
ia_unifiedcache(void) {
    ICPU4        c4;
    int          n;
    int          i;
    int          r, r2, r3;
    /* cache size information available */

    r2 = r3 = 0;
    for (i = 0; i <= 3; i++) {
	__pgi_cpuid_ecx( 4, c4.i, i );
	DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	    c4.i[0], c4.i[1], c4.i[2], c4.i[3] );
	switch (c4.u.eax.cachetype) {
	default:
	    goto done;
	case 1:
	    /*
	    printf("Data Cache\n");
	    printf("+++ level %d\n", c4.u.eax.cachelevel);
	    printf("+++ #bytes %d\n",
		( (c4.u.ebx.assoc+1) * 
		  (c4.u.ebx.partitions+1) *
		  (c4.u.ebx.linesize+1) *
		  (c4.u.nsets+1) ) ;
	    );
	    */
	    break;
	case 2:
	    /*
	    printf("Instruction Cache\n");
	    printf("+++ level %d\n", c4.u.eax.cachelevel);
	    */
	    break;
	case 3:
	    /*
	    printf("Unified Cache\n");
	    printf("+++ level %d\n", c4.u.eax.cachelevel);
	    printf("+++ #bytes %d\n",
		( (c4.u.ebx.assoc+1) * 
		  (c4.u.ebx.partitions+1) *
		  (c4.u.ebx.linesize+1) *
		  (c4.u.nsets+1) )
	    );
	    */
	    r =  (c4.u.ebx.assoc+1) * 
		 (c4.u.ebx.partitions+1) *
		 (c4.u.ebx.linesize+1) *
		 (c4.u.nsets+1);
	    if (c4.u.eax.cachelevel == 2)
		r2 = r;
	    else if (c4.u.eax.cachelevel == 3) {
		r3 = r;
	    }
	    break;
	}
    }
done:
    if (r3)
	return r3;
    return r2;
}

/*
 * return cache size for AMD processors
 */
static int
amd_cachesize(void)
{
    CPU80 c80;
    ACPU86 c86;

    if( X86IDFN(idcache)( 0x80000000U, c80.i ) == 0 )
	return 0;
    DEBUG_PRINTF("largest=%#8.8x", c80.b.largest );
    if( c80.b.largest < 0x80000006U )
	return 0;
    if( X86IDFN(idcache)( 0x80000006U, c86.i ) == 0 )
	return 0;
    if( c86.u.l3cache.size ) {
	return c86.u.l3cache.size * 512 * 1024;
    }
    return c86.u.l2cache.size * 1024;
}/* amd_cachesize */

/*
 * test(cachesize)
 *  return intel or amd cache size
 */
int
X86IDFN(get_cachesize)(void)
{
    if( X86IDFN(is_intel)() )
	return ia_cachesize();
    if( X86IDFN(is_amd)() )
	return amd_cachesize();
    return 0;
}/* get_cachesize */

/*
 * return cores for Intel processors
 */
static int
ia_cores(void)
{
    CPU0 c0;
    ICPU4 c4;
    int i, n, r;

    if( X86IDFN(idcache)( 0, c0.i ) == 0 )
	return 0;
    DEBUG_PRINTF("largest=%d", c0.b.largest );

    if( c0.b.largest < 4 )
	return 0;

    __pgi_cpuid_ecx( 4, c4.i, 0 );
    DEBUG_PRINTF("eax %#8.8x ebx %#8.8x ecx %#8.8x edx %#8.8x",
	c4.i[0], c4.i[1], c4.i[2], c4.i[3] );
    return c4.u.eax.ncores + 1;
}/* ia_cores */

/*
 * return cores for AMD processors
 */
static int
amd_cores(void)
{
    CPU80 c80;
    ACPU88 c88;

    if( X86IDFN(idcache)( 0x80000000U, c80.i ) == 0 )
	return 0;
    DEBUG_PRINTF("largest=%d", c80.b.largest );
    if( c80.b.largest < 0x80000008U )
	return 0;
    if( X86IDFN(idcache)( 0x80000008U, c88.i ) == 0 )
	return 0;
    return c88.u.ecx.cores + 1;
}/* amd_cores */

/*
 * test(cpuname)
 *  return processor name string
 */
static char processor_name[50];
char *
X86IDFN(get_processor_name)(void)
{
    CPU80 c80;
    int i;
    if( X86IDFN(idcache)( 0x80000000, c80.i ) == 0 )
	return 0;
    DEBUG_PRINTF("eax %#8.8x", c80.i[0] );
    if( c80.b.largest < 0x80000004 ){
	processor_name[0] = '\0';
	return processor_name;	/* no processor name string */
    }
    if( X86IDFN(idcache)( 0x80000002, (unsigned int*)(processor_name+0) ) == 0 ){
	processor_name[0] = '\0';
	return processor_name;	/* no processor name string */
    }
    if( X86IDFN(idcache)( 0x80000003, (unsigned int*)(processor_name+16) ) == 0 ){
	processor_name[0] = '\0';
	return processor_name;	/* no processor name string */
    }
    if( X86IDFN(idcache)( 0x80000004, (unsigned int*)(processor_name+32) ) == 0 ){
	processor_name[0] = '\0';
	return processor_name;	/* no processor name string */
    }
    processor_name[48] = '\0';
    for( i = 0; i < 48; ++i ){
	if( processor_name[i] != ' ' )
	    return processor_name+i;
    }
    return processor_name;
}/* get_processor_name */
