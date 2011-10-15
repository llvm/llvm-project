/* This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 */

/* double __floatdidf(di_int a); */

#ifdef __x86_64__

#include <stdint.h>

double __floatdidf(int64_t a)
{
	return (double)a;
}

#endif /* __x86_64__ */
