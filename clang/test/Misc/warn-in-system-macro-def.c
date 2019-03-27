// RUN: %clang_cc1 -triple x86_64-apple-darwin -isystem %S -Wdouble-promotion -fsyntax-only %s  2>&1 | FileCheck -allow-empty %s
// CHECK: warning:
// CHECK: expanded from macro 'ISNAN'
// CHECK: expanded from macro 'isnan'

#include <warn-in-system-macro-def.c.inc>

#define isnan(x) \
	(sizeof (x) == sizeof (float)                \
	? __isnanf (x)                    \
	: sizeof (x) == sizeof (double)               \
	? __isnan (x) : __isnanl (x))

int main(void)
{
	double foo = 1.0;

	if (ISNAN(foo))
		return 1;
	return 0;
}
