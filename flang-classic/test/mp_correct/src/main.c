/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdio.h>
static int vect_error = 0;
static int dummy();

#ifdef __cplusplus
extern "C" void check(int *, int *, int);
extern "C" void test();
#else
extern void check(int *, int *, int);
#endif

int
main() {
    test();
    if (vect_error == -1)
	printf("---------------------- vector test PASSED\n");
    else if (vect_error == 1)
	printf("---------------------- vector test FAILED\n");
    fclose(stdout);
    return 0;
}

#ifdef __cplusplus
extern "C"
#endif
void
check(int result[], int expect[], int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (expect[i] == result[i])  tests_passed ++;
        else {
            tests_failed ++;
            printf("---- test number %d failed. result %d  expected %d\n",
                      i, result[i], expect[i] );
        }
    }
    if (tests_failed == 0)
	printf("---- %3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    else
	printf("---- %3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
}

#ifdef __cplusplus
extern "C"
#endif
void
checki8( unsigned int res[], unsigned int exp[], int np)
{
    int i;
    int n = np * 2;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i+=2) {
        if (exp[i] == res[i] && exp[i+1] == res[i+1])
           tests_passed ++;
        else {
            tests_failed ++;
            printf( "---- test number %d FAILED. res %d(%08x) %d(%08x)\n",
                         i/2, res[i], res[i], res[i+1], res[i+1]);
            printf( "                            exp %d(%08x) %d(%08x)\n",
                         exp[i], exp[i], exp[i+1], exp[i+1] );
        }
    }
    if (tests_failed == 0)
        printf("---- %3d tests completed. %d tests PASSED. %d tests failed.\n",
                      np, tests_passed, tests_failed);
    else
        printf("---- %3d tests completed. %d tests passed. %d tests FAILED.\n",
                      np, tests_passed, tests_failed);
}

#ifdef __cplusplus
extern "C"
#endif
void
checkf( float result[], float expect[], int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (expect[i] == result[i])  tests_passed ++;
        else {
            int j = (result[i] - expect[i]) * 1000.0;
            tests_failed ++;
            printf("---- test number %d failed. single prec diff: %d\n",
                      i, j/*,  result[i], expect[i] */ );
        }
    }
    if (tests_failed == 0)
	printf("---- %3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    else
	printf("---- %3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
}

#ifdef __cplusplus
extern "C"
#endif
void
checkd( double result[], double expect[], int n)
{
    int i;
    int tests_passed = 0;
    int tests_failed = 0;

    for (i = 0; i < n; i++) {
        if (expect[i] == result[i])  tests_passed ++;
        else {
            int j = (result[i] - expect[i]) * 1000.0;
            tests_failed ++;
            printf("---- test number %d failed. double prec diff: %d\n",
                      i, j/*,  result[i], expect[i] */ );
        }
    }
    if (tests_failed == 0)
	printf("---- %3d tests completed. %d tests PASSED. %d tests failed.\n",
                      n, tests_passed, tests_failed);
    else
	printf("---- %3d tests completed. %d tests passed. %d tests FAILED.\n",
                      n, tests_passed, tests_failed);
}

/* --------------- vector test support utilities ------------------------- */

static double d = 2.0;
static int   sign = 1;

#ifdef __cplusplus
extern "C"
#endif
void
initdv(double *a, int n, int stride)
{
    int i;

    vect_error = -1;
    for (i = 0; n-- > 0; i += stride) {
	a[i] = sign * d;
	sign = -sign;
	d += .51;
    }

}

#ifdef __cplusplus
extern "C"
#endif
void
initfv(float *a, int n, int stride)
{
    int i;

    vect_error = -1;
    for (i = 0; n-- > 0; i += stride) {
	a[i] = sign * d;
	sign = -sign;
	d += .51;
    }
}

/* ----------------------------------------------------------------------- */

#define EPS 8.0e-16

#ifdef __cplusplus
extern "C"
#endif
void
dvcheck(double *a, char *txt, int n, int stride, double val, int dump)
{
    double sum = 0, x;
    int i, m = n;

    for (i = 0; n-- > 0; i += stride)
	dummy(),
	sum += a[i];

    if (val == 0.0)	val = 1.0;
    x = (sum - val)/val;
    if (x < 0.0)  x = -x;

    if (x >= EPS) {
	fprintf(stdout, "---- bad checksum for %s. exp: %1.17g  res: %1.17g\n",
		txt, val, sum);
	vect_error = 1;
    }
    else if (vect_error == 0)
	vect_error = -1;		/* set flag for test driver */

    if (dump)
	for (i = 0; m-- > 0; i += stride)
	    fprintf(stdout, " %s[%d] = %g\n", txt, i, a[i]);
}

/* ----------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C"
#endif
void
fvcheck(float *a, char *txt, int n, int stride, double val, int dump)
{
    double sum = 0, x;
    int i, m = n;
    /* 2.9802326E-8 */

    for (i = 0; n-- > 0; i += stride)
	dummy(),
	sum += a[i];

    if (val == 0.0)	val = 1.0;
    x = (sum - val)/val;
    if (x < 0.0)  x = -x;

    if (x >= EPS) {
	fprintf(stdout, "---- bad checksum for %s. exp: %1.17g  res: %1.17g\n",
		txt, val, sum);
	vect_error = 1;
    }
    else if (vect_error == 0)
	vect_error = -1;		/* set flag for test driver */

    if (dump)
	for (i = 0; m-- > 0; i += stride)
	    fprintf(stdout, " %s[%d] = %g\n", txt, i, a[i]);
}

static int
dummy() { return 1; }
