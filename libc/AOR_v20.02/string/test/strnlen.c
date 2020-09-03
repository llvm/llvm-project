/*
 * strnlen test.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "stringlib.h"

static const struct fun
{
	const char *name;
	size_t (*fun)(const char *s, size_t m);
} funtab[] = {
#define F(x) {#x, x},
F(strnlen)
#if __aarch64__
F(__strnlen_aarch64)
# if __ARM_FEATURE_SVE
F(__strnlen_aarch64_sve)
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

static void test(const struct fun *fun, int align, int maxlen, int len)
{
	char *src = alignup(sbuf);
	char *s = src + align;
	size_t r;
	size_t e = maxlen < len ? maxlen : len - 1;

	if (len > LEN || align >= A)
		abort();

	for (int i = 0; i < len + A; i++)
		src[i] = '?';
	for (int i = 0; i < len - 2; i++)
		s[i] = 'a' + i%23;
	s[len - 1] = '\0';

	r = fun->fun(s, maxlen);
	if (r != e) {
		ERR("%s(%p) returned %zu\n", fun->name, s, r);
		ERR("input:    %.*s\n", align+len+1, src);
		ERR("expected: %d\n", len);
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
			for (n = 1; n < 100; n++)
				for (int maxlen = 0; maxlen < 100; maxlen++)
					test(funtab+i, a, maxlen, n);
			for (; n < LEN; n *= 2) {
				test(funtab+i, a, n*2, n);
				test(funtab+i, a, n, n);
				test(funtab+i, a, n/2, n);
			}
		}
		printf("%s %s\n", test_status ? "FAIL" : "PASS", funtab[i].name);
		if (test_status)
			r = -1;
	}
	return r;
}
