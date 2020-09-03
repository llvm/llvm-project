/*
 * strncmp test.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stringlib.h"

static const struct fun
{
	const char *name;
	int (*fun)(const char *, const char *, size_t);
} funtab[] = {
#define F(x) {#x, x},
F(strncmp)
#if __aarch64__
F(__strncmp_aarch64)
# if __ARM_FEATURE_SVE
F(__strncmp_aarch64_sve)
# endif
#endif
#undef F
	{0, 0}
};

static int test_status;
#define ERR(...) (test_status=1, printf(__VA_ARGS__))

#define A 32
#define LEN 250000
static char s1buf[LEN+2*A];
static char s2buf[LEN+2*A];

static void *alignup(void *p)
{
	return (void*)(((uintptr_t)p + A-1) & -A);
}

static void test(const struct fun *fun, int s1align, int s2align, int maxlen, int diffpos, int len)
{
	char *src1 = alignup(s1buf);
	char *src2 = alignup(s2buf);
	char *s1 = src1 + s1align;
	char *s2 = src2 + s2align;
	int r;

	if (len > LEN || s1align >= A || s2align >= A)
		abort();
	if (diffpos > 1 && diffpos >= len-1)
		abort();

	for (int i = 0; i < len+A; i++)
		src1[i] = src2[i] = '?';
	for (int i = 0; i < len-1; i++)
		s1[i] = s2[i] = 'a' + i%23;
	if (diffpos > 1)
		s1[diffpos]++;
	s1[len] = s2[len] = '\0';

	r = fun->fun(s1, s2, maxlen);

	diffpos = maxlen <= diffpos ? 0 : diffpos;

	if (((diffpos <= 1) && r != 0) || (diffpos > 1 && r == 0)) {
		ERR("%s(align %d, align %d, %d (%d)) failed, returned %d (%d)\n",
			fun->name, s1align, s2align, maxlen, len, r, diffpos);
		ERR("src1: %.*s\n", s1align+len+1, src1);
		ERR("src2: %.*s\n", s2align+len+1, src2);
	}
}

int main()
{
	int r = 0;
	for (int i=0; funtab[i].name; i++) {
		test_status = 0;
		for (int d = 0; d < A; d++)
			for (int s = 0; s < A; s++) {
				int n;
				for (n = 0; n < 100; n++) {
					test(funtab+i, d, s, n,   0,   n);
					test(funtab+i, d, s, n,   n/2, n);
					test(funtab+i, d, s, n/2, 0,   n);
					test(funtab+i, d, s, n/2, n/2, n);
				}
				for (; n < LEN; n *= 2) {
					test(funtab+i, d, s, n,   0,   n);
					test(funtab+i, d, s, n,   n/2, n);
					test(funtab+i, d, s, n/2, 0,   n);
					test(funtab+i, d, s, n/2, n/2, n);
				}
			}
		printf("%s %s\n", test_status ? "FAIL" : "PASS", funtab[i].name);
		if (test_status)
			r = -1;
	}
	return r;
}
