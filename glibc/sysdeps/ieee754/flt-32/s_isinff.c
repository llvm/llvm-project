/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Public domain.
 */

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_isinff.c,v 1.3 1995/05/11 23:20:21 jtc Exp $";
#endif

/*
 * isinff(x) returns 1 if x is inf, -1 if x is -inf, else 0;
 * no branching!
 */

#include <math.h>
#include <math_private.h>

int
__isinff (float x)
{
	int32_t ix,t;
	GET_FLOAT_WORD(ix,x);
	t = ix & 0x7fffffff;
	t ^= 0x7f800000;
	t |= -t;
	return ~(t >> 31) & (ix >> 30);
}
hidden_def (__isinff)
weak_alias (__isinff, isinff)
