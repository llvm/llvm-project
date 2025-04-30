/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Change for long double by Ulrich Drepper <drepper@cygnus.com>.
 * Intel i387 specific version.
 * Public domain.
 */

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: $";
#endif

/*
 * isinfl(x) returns 1 if x is inf, -1 if x is -inf, else 0;
 * no branching!
 */

#include <math.h>
#include <math_private.h>

int __isinfl(long double x)
{
	int32_t se,hx,lx;
	GET_LDOUBLE_WORDS(se,hx,lx,x);
	/* This additional ^ 0x80000000 is necessary because in Intel's
	   internal representation of the implicit one is explicit.  */
	lx |= (hx ^ 0x80000000) | ((se & 0x7fff) ^ 0x7fff);
	lx |= -lx;
	se &= 0x8000;
	return ~(lx >> 31) & (1 - (se >> 14));
}
hidden_def (__isinfl)
weak_alias (__isinfl, isinfl)
