/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef	CPUIDX8664_H
#define	CPUIDX8664_H

#include <stdio.h>
#include <stdint.h>
#include <cpuid.h>

#if	! defined(TARGET_X8664)
#error	To use cpuid8664.h CPP macro TARGET_X8664 must be defined.
#endif

/*
 * Define some interesting fields in the extended control register[0].
 * Currently xcr[0] only defines bits in the lower 32-bits of the 64-bit
 * register.
 */

#define	xcr0_bit_XMM	0x00000002U
#define	xcr0_bit_YMM	0x00000004U
#define	xcr0_bit_ZMMK	0x00000020U
#define	xcr0_bit_ZMMLO 	0x00000040U
#define	xcr0_bit_ZMMHI 	0x00000080U

#define	xcr0_mask_YMM	(xcr0_bit_XMM | xcr0_bit_YMM)
#define	xcr0_mask_ZMM	(xcr0_bit_ZMMK | xcr0_bit_ZMMLO | xcr0_bit_ZMMHI)

/*
 * Simple macro to add an arbitrary prefix to the internal (static) functions
 * defined in this header.
 */

#ifndef	CPUIDX8664
#define CPUIDX8664(a)  __cpuid_##a
#endif		// #ifndef CPUIDX8664


static	int CPUIDX8664(is_avx512)();
static	int CPUIDX8664(is_avx512vl)();
static	int CPUIDX8664(is_avx512f)();
static	int CPUIDX8664(is_avx2)();
static	int CPUIDX8664(is_avx)();
static	int CPUIDX8664(is_intel)();
static	int CPUIDX8664(is_amd)();
static	int CPUIDX8664(is_fma4)();
static	int CPUIDX8664(is_sse4a)();
static	int CPUIDX8664(is_sse41)();
static	int CPUIDX8664(is_f16c)();

/*
 * Check that this is a Genuine Intel processor
 */
static int
CPUIDX8664(is_intel)(void)
{
    uint32_t eax, ebx, ecx, edx;

    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    return ((signature_INTEL_ebx ^ ebx) | (signature_INTEL_ecx ^ ecx) | (signature_INTEL_edx ^ edx)) == 0;
}/* is_intel */

/*
 * Check that this is a Genuine AMD processor
 */
static int
CPUIDX8664(is_amd)(void)
{
    uint32_t eax, ebx, ecx, edx;

    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    return ((signature_AMD_ebx ^ ebx) | (signature_AMD_ecx ^ ecx) | (signature_AMD_edx ^ edx)) == 0;
}/* is_amd */

/*
 * Check that this is a Genuine AMD processor that supports FMA4 instructions.
 */
static int
CPUIDX8664(is_fma4)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if (CPUIDX8664(is_amd)() == 0) {
        return 0;
    }

    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    if (eax < 0x80000001) {
        return 0;       // No extended flags
    }

    if (__get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ecx & bit_FMA4) != 0;
}/* is_fma4 */

/*
 * Check that this is a Genuine AMD processor that supports SSE4a instructions.
 * Note: right now, it's just really the greyhound check.
 */
static int
CPUIDX8664(is_sse4a)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if (CPUIDX8664(is_amd)() == 0) {
        return 0;
    }

    if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    if (eax < 0x80000001) {
        return 0;       // No extended flags
    }

    if (__get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ecx & bit_SSE4a) != 0;
}/* is_sse4a */

/*
 * Check that this is a Genuine Intel processor that supports SSE4.1
 * instructions.
 * Note: right now, it's just really the penryn check.
 */
static int
CPUIDX8664(is_sse41)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if (CPUIDX8664(is_intel)() == 0) {
        return 0;
    }

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ecx & bit_SSE4_1) != 0;
}/* is_sse41 */

/*
 * Check that this is either a Genuine Intel or AMD processor that supports
 * AVX instructions.
 */
static int
CPUIDX8664(is_avx)(void)
{
    uint32_t eax, ebx, ecx, edx;
    uint32_t xcr0;		// Right now only lower 32-bit are interesting

    if ((CPUIDX8664(is_intel)() == 0) && (CPUIDX8664(is_amd)() == 0)) {
        return 0;
    }

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    if ((ecx & bit_AVX) == 0) {
        return 0;
    }
    if ((ecx & bit_OSXSAVE) == 0) {	// Is xgetbv available
        return 0;
    }

    /*
     * Now check to see whether O/S has enabled AVX intructions.
     * asm("xgetbv\n\tshlq\t$32,%%rdx\n\torq\t%%rdx,%%rax"
     *      : "=a"(xcr0) : "c"(0) : "%rdx");
     */
    asm("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");

    return ((xcr0 & xcr0_mask_YMM) == xcr0_mask_YMM);
}/* is_avx */

/*
 * Check that this is either a Genuine Intel or AMD processor that supports
 * AVX2 instructions.
 */
static int
CPUIDX8664(is_avx2)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if ((CPUIDX8664(is_intel)() == 0) && (CPUIDX8664(is_amd)() == 0)) {
        return 0;
    }

    if (CPUIDX8664(is_avx)() == 0) {
        return 0;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ebx & bit_AVX2) != 0;
}/* is_avx2 */


/*
 * Check that this is a Genuine Intel processor that supports
 * AVX512 instructions.
 * This routine should not be used standalone - should be used as
 * a helper function to is_avx512f() and is_avx512vl().
 */
static int
CPUIDX8664(is_avx512)(void)
{
    uint32_t eax, ebx, ecx, edx;
    uint32_t xcr0;		// Right now only lower 32-bit are interesting

    /*
     * Must check for is_intel() explicitly since is_avx()
     * returns true if AVX is enable for either Intel or AMD.
     */
    if (CPUIDX8664(is_intel)() == 0) {
        return 0;
    }

    if (CPUIDX8664(is_avx)() == 0) {
        return 0;
    }

    /*
     * Now check to see whether O/S has enabled AVX512 intructions.
     * asm("xgetbv\n\tshlq\t$32,%%rdx\n\torq\t%%rdx,%%rax"
     *      : "=a"(xcr0) : "c"(0) : "%rdx");
     */
    asm("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");

    return ((xcr0 & xcr0_mask_ZMM) == xcr0_mask_ZMM);
}

/*
 * Check that this is a Genuine Intel processor that supports
 * AVX512f (foundation) instructions.
 */
static int
CPUIDX8664(is_avx512f)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if (CPUIDX8664(is_avx512)() == 0) {
        return 0;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ebx & bit_AVX512F) != 0;
}/* is_avx512f */

/*
 * Check that this is a Genuine Intel processor that supports
 * AVX512vl (vector length) instructions.
 */
static int
CPUIDX8664(is_avx512vl)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if (CPUIDX8664(is_avx512f)() == 0) {
        return 0;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ebx & bit_AVX512VL) != 0;
}/* is_avx512vl */

/*
 * Check that this is either a Genuine Intel or AMD processor that supports
 * f16c instructions.
 */
static int
CPUIDX8664(is_f16c)(void)
{
    uint32_t eax, ebx, ecx, edx;

    if ((CPUIDX8664(is_intel)() == 0) && (CPUIDX8664(is_amd)() == 0)) {
        return 0;
    }

    if (CPUIDX8664(is_avx)() == 0) {
        return 0;
    }

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return 0;
    }

    return (ecx & bit_F16C) != 0;
}/* is_f16c */

#ifdef  UNIT_TEST
int
main()
{
  printf("is_intel()=%d\n", CPUIDX8664(is_intel)());
  printf("is_amd()=%d\n", CPUIDX8664(is_amd)());
  printf("is_fma4()=%d\n", CPUIDX8664(is_fma4)());
  printf("is_sse4a()=%d\n", CPUIDX8664(is_sse4a)());
  printf("is_sse41()=%d\n", CPUIDX8664(is_sse41)());
  printf("is_avx()=%d\n", CPUIDX8664(is_avx)());
  printf("is_avx2()=%d\n", CPUIDX8664(is_avx2)());
  printf("is_avx512f()=%d\n", CPUIDX8664(is_avx512f)());
  printf("is_avx512vl()=%d\n", CPUIDX8664(is_avx512vl)());
  printf("is_f16c()=%d\n", CPUIDX8664(is_f16c)());
}
#endif
#endif		// #ifndef	CPUIDX8664_H
