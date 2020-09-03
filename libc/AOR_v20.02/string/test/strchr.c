/*
 * strchr test.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "stringlib.h"

static const struct fun
{
	const char *name;
	char *(*fun)(const char *s, int c);
} funtab[] = {
#define F(x) {#x, x},
F(strchr)
#if __aarch64__
F(__strchr_aarch64)
F(__strchr_aarch64_mte)
# if __ARM_FEATURE_SVE
F(__strchr_aarch64_sve)
# endif
#endif
#undef F
	{0, 0}
};

static int test_status;
#define ERR(...) (test_status=1, printf(__VA_ARGS__))

#define A 32
#define SP 512
#define LEN 250000
static char sbuf[LEN+2*A];

static void *alignup(void *p)
{
	return (void*)(((uintptr_t)p + A-1) & -A);
}

static void test(const struct fun *fun, int align, int seekpos, int len)
{
	char *src = alignup(sbuf);
	char *s = src + align;
	char *f = seekpos != -1 ? s + seekpos : 0;
	int seekchar = 0x1;
	void *p;

	if (len > LEN || seekpos >= len - 1 || align >= A)
		abort();
	if (seekchar >= 'a' && seekchar <= 'a' + 23)
		abort();

	for (int i = 0; i < len + A; i++)
		src[i] = '?';
	for (int i = 0; i < len - 2; i++)
		s[i] = 'a' + i%23;
	if (seekpos != -1)
		s[seekpos] = seekchar;
	s[len - 1] = '\0';

	p = fun->fun(s, seekchar);

	if (p != f) {
		ERR("%s(%p,0x%02x,%d) returned %p\n", fun->name, s, seekchar, len, p);
		ERR("expected: %p\n", f);
		abort();
	}
}

int main()
{
	int r = 0;
	for (int i=0; funtab[i].name; i++) {
		test_status = 0;
		for (int a = 0; a < A; a++) {
			int n;
			for (n = 1; n < 100; n++) {
				for (int sp = 0; sp < n - 1; sp++)
					test(funtab+i, a, sp, n);
				test(funtab+i, a, -1, n);
			}
			for (; n < LEN; n *= 2) {
				test(funtab+i, a, -1, n);
				test(funtab+i, a, n / 2, n);
			}
		}
		printf("%s %s\n", test_status ? "FAIL" : "PASS", funtab[i].name);
		if (test_status)
			r = -1;
	}
	return r;
}
