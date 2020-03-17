/*
 * memcmp test.
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
	int (*fun)(const void *s1, const void *s2, size_t n);
} funtab[] = {
#define F(x) {#x, x},
F(memcmp)
#if __aarch64__
F(__memcmp_aarch64)
# if __ARM_FEATURE_SVE
F(__memcmp_aarch64_sve)
# endif
#endif
#undef F
	{0, 0}
};

static int test_status;
#define ERR(...) (test_status=1, printf(__VA_ARGS__))

#define A 32
#define LEN 250000
static unsigned char s1buf[LEN+2*A];
static unsigned char s2buf[LEN+2*A];

static void *alignup(void *p)
{
	return (void*)(((uintptr_t)p + A-1) & -A);
}

static void test(const struct fun *fun, int s1align, int s2align, int len, int diffpos)
{
	unsigned char *src1 = alignup(s1buf);
	unsigned char *src2 = alignup(s2buf);
	unsigned char *s1 = src1 + s1align;
	unsigned char *s2 = src2 + s2align;
	int r;

	if (len > LEN || s1align >= A || s2align >= A)
		abort();
	if (diffpos && diffpos >= len)
		abort();

	for (int i = 0; i < len+A; i++)
		src1[i] = src2[i] = '?';
	for (int i = 0; i < len; i++)
		s1[i] = s2[i] = 'a' + i%23;
	if (diffpos)
		s1[diffpos]++;

	r = fun->fun(s1, s2, len);

	if ((!diffpos && r != 0) || (diffpos && r == 0)) {
		ERR("%s(align %d, align %d, %d) failed, returned %d\n",
			fun->name, s1align, s2align, len, r);
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
					test(funtab+i, d, s, n, 0);
					test(funtab+i, d, s, n, n / 2);
				}
				for (; n < LEN; n *= 2) {
					test(funtab+i, d, s, n, 0);
					test(funtab+i, d, s, n, n / 2);
				}
			}
		printf("%s %s\n", test_status ? "FAIL" : "PASS", funtab[i].name);
		if (test_status)
			r = -1;
	}
	return r;
}
