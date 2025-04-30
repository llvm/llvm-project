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
}

void
check(int* res, int* exp, int* np)
{
    check_(res, exp, np);
}

#if defined(WINNT) || defined(WIN32)
void
__stdcall CHECK(int* res, int* exp, int* np)
{
    check_(res, exp, np);
}
#endif
