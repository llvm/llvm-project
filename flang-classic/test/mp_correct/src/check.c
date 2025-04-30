/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void
check_(int * res, int * exp, int * np)
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
            printf(
	    "---- test number %d FAILED. res %d(%08x)  exp %d(%08x)\n",
	     i+1, res[i], res[i], exp[i], exp[i] );
        }
    }
    if (tests_failed == 0)
	printf(
	"---- %3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    else
	printf("---- %3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
}

void
check(int * res, int * exp, int * np)
{
    check_(res, exp, np);
}

int
chkalt_(int * ir, int * ie, int * np)
{
    int i;
    int n = *np;
    for (i = 0; i < n; i++) {
	if (ir[i] != ie[i])
	    return 0;
    }
    return 1;
}

int
chkalt(int * ir, int * ie, int * np)
{
    return chkalt_(ir, ie, np);
}

#if defined(WINNT) || defined(WIN32)
void
__stdcall CHECK(res, exp, np)
    int *res, *exp, *np;
{
    check_(res, exp, np);
}

int __stdcall CHKALT (int *ir, int *ie, int *np) {
    return chkalt_(ir, ie, np);
}
#endif

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
}

void
checkf(float* res, float* exp, int* np)
{
    checkf_(res, exp, np);
}

#ifdef __cplusplus
}
#endif
