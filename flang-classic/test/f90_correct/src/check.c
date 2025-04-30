/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern int __hpf_lcpu;

void
checkh_(short* res, short* exp, int* np)
{
    int i;
    int n = *np;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
      if (exp[i] & (~ res[i])) {
            tests_failed ++;
	    if( tests_failed < 100 )
            printf(
	    "test number %d FAILED. res %u(%04x)  exp %u(%04x)\n",
	     i+1,res[i], res[i], exp[i], exp[i] );
        } else {
	    tests_passed ++;
        }
    }
    if (tests_failed == 0) {
	    printf(
	"%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
}

void
check_(int* res, int* exp, int* np)
{
    int i;
    int n = *np;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
        } else {
            tests_failed ++;
	    if( tests_failed < 100 )
            printf(
	    "test number %d FAILED. res %d(%08x)  exp %d(%08x)\n",
	     i+1,res[i], res[i], exp[i], exp[i] );
        }
    }
    if (tests_failed == 0) {
	    printf(
	"%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
    fflush(stdout);
}

void
check(int* res, int* exp, int* np)
{
    check_(res, exp, np);
}

void
checkll_(long long *res, long long *exp, int *np)
{
    int i;
    int n = *np;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (exp[i] == res[i]) {
	    tests_passed ++;
        } else {
             tests_failed ++;
	    if( tests_failed < 100 )
             printf( "test number %d FAILED. res %lld(%0llx)  exp %lld(%0llx)\n",
	     i+1,res[i], res[i], exp[i], exp[i] );
        }
    }
    if (tests_failed == 0) {
	    printf(
	"%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    } else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
    fflush(stdout);
}

void
checkll(long long *res, long long *exp, int *np)
{
    checkll_(res, exp, np);
    fflush(stdout);
}

/* maximum allowed difference in units in the last place */
#ifndef MAX_DIFF_ULPS
#define MAX_DIFF_ULPS 2
#endif

void
checkf_(float* res, float* exp, int* np)
{
    int i;
    int n = *np;
    int tests_passed = 0;
    int tests_failed = 0;
    int ires, iexp, diff;

    assert(sizeof(int) == 4);
    assert(sizeof(float) == 4);
    for (i = 0; i < n; i++) {
	ires = *(int *)(res + i);
	iexp = *(int *)(exp + i);
	if (ires < 0)
	    ires = 0x80000000 - ires;
	if (iexp < 0)
	    iexp = 0x80000000 - iexp;
	diff = abs(ires - iexp);
        if (diff <= MAX_DIFF_ULPS)
	    tests_passed++;
        else {
            tests_failed++;
	    if (tests_failed < 100) 
		printf("test number %d FAILED. diff in last place units: %d\n",
			i+1, diff);
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    }
    else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
    fflush(stdout);
}

void
checkf(float* res, float* exp, int* np)
{
    checkf_(res, exp, np);
}

void
checkd_(double* res, double* exp, int* np)
{
    int i;
    int n = *np;
    int tests_passed = 0;
    int tests_failed = 0;
    int resh, exph, diffh;
    unsigned int resl, expl, diffl;
    int borrow;

    assert(sizeof(int) == 4);
    assert(sizeof(double) == 8);
    for (i = 0; i < n; i++) {
#ifdef BIG_ENDIAN
	resh = *(int *)(res + i);
	resl = *((unsigned int *)(res + i) + 1);
	exph = *(int *)(exp + i);
	expl = *((unsigned int *)(exp + i) + 1);
#else
	resl = *(unsigned int *)(res + i);
	resh = *((int *)(res + i) + 1);
	expl = *(unsigned int *)(exp + i);
	exph = *((int *)(exp + i) + 1);
#endif
	/* if (res < 0) res = 0x8000000000000000 - res; */
	if (resh < 0) {
	    resl = 0 - resl;
	    borrow = (resl != 0);
	    resh = 0x80000000 - resh - borrow;
	}
	/* if (exp < 0) exp = 0x8000000000000000 - exp; */
	if (exph < 0) {
	    expl = 0 - expl;
	    borrow = (expl != 0);
	    exph = 0x80000000 - exph - borrow;
	}
	/* diff = llabs(res - exp); */
	diffl = resl - expl;
	borrow = (int)((resl >> 31) - (expl >> 31)) < (int)(diffl >> 31);
	diffh = resh - exph - borrow;
	if (diffh < 0) {
	    diffl = -diffl;
	    borrow = (diffl != 0);
	    diffh = -diffh - borrow;
	}
        if (diffh == 0 && diffl <= MAX_DIFF_ULPS)
	    tests_passed++;
        else {
            tests_failed++;
	    if (tests_failed < 100) 
		printf("test number %d FAILED. diff in last place units: %d %d\n",
			i+1, diffh, diffl);
        }
    }
    if (tests_failed == 0) {
	printf("%3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    }
    else {
	printf("%3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
    }
    fflush(stdout);
}

void
checkd(double* res, double* exp, int* np)
{
    checkd_(res, exp, np);
}

void
fcpyf_(float *r, float f)
{
*r = f;
}

void
fcpyf(float *r, float f)
{
    fcpyf_(r, f);
}

void
fcpyi_(int *r, int f)
{
    *r = f;
}

void
fcpyi(int *r, int f)
{
    fcpyi_(r, f);
}

#if defined(WINNT) || defined(WIN32)
void
__stdcall CHECK(int* res, int* exp, int* np)
{
    check_(res, exp, np);
}

void
__stdcall CHECKF(float* res, float* exp, int* np)
{
    checkf_(res, exp, np);
}

void
__stdcall CHECKD(double* res, double* exp, int* np)
{
    checkd_(res, exp, np);
}

void
__stdcall CHECKLL(long long *res, long long *exp, int *np)
{
    checkll_(res, exp, np);
}

void
__stdcall FCPYF(float *r, float f)
{
    fcpyf_(r, f);
}

void
__stdcall FCPYI(int *r, int f)
{
    fcpyi_(r, f);
}
#endif
