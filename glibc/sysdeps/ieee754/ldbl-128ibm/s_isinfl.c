/*
 * Written by J.T. Conklin <jtc@netbsd.org>.
 * Change for long double by Jakub Jelinek <jj@ultra.linux.cz>
 * Public domain.
 */

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: $";
#endif

/*
 * isinfl(x) returns 1 if x is inf, -1 if x is -inf, else 0;
 * no branching!
 * slightly dodgy in relying on signed shift right copying sign bit
 */

#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

int
___isinfl (long double x)
{
  double xhi;
  int64_t hx, mask;

  xhi = ldbl_high (x);
  EXTRACT_WORDS64 (hx, xhi);

  mask = (hx & 0x7fffffffffffffffLL) ^ 0x7ff0000000000000LL;
  mask |= -mask;
  mask >>= 63;
  return ~mask & (hx >> 62);
}
hidden_ver (___isinfl, __isinfl)
#if !IS_IN (libm)
weak_alias (___isinfl, ____isinfl)
long_double_symbol (libc, ___isinfl, isinfl);
long_double_symbol (libc, ____isinfl, __isinfl);
#endif
