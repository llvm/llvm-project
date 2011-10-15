/* This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 */

#include "DD.h"
#include <math.h>

#if !defined(INFINITY) && defined(HUGE_VAL)
#define INFINITY HUGE_VAL
#endif /* INFINITY */

#define makeFinite(x)	{ \
							(x).s.hi = __builtin_copysign(isinf((x).s.hi) ? 1.0 : 0.0, (x).s.hi); \
							(x).s.lo = 0.0; \
						}

#define zeroNaN(x)		{ \
							if (isnan((x).s.hi)) { \
								(x).s.hi = __builtin_copysign(0.0, (x).s.hi); \
								(x).s.lo = 0.0; \
							} \
						}

long double __gcc_qadd(long double, long double);
long double __gcc_qsub(long double, long double);
long double __gcc_qmul(long double, long double);

long double _Complex
__multc3(long double a, long double b, long double c, long double d)
{
	long double ac = __gcc_qmul(a,c);
	long double bd = __gcc_qmul(b,d);
	long double ad = __gcc_qmul(a,d);
	long double bc = __gcc_qmul(b,c);
	
	DD real = { .ld = __gcc_qsub(ac,bd) };
	DD imag = { .ld = __gcc_qadd(ad,bc) };
	
	if (isnan(real.s.hi) && isnan(imag.s.hi))
	{
		int recalc = 0;
		
		DD aDD = { .ld = a };
		DD bDD = { .ld = b };
		DD cDD = { .ld = c };
		DD dDD = { .ld = d };
		
		if (isinf(aDD.s.hi) || isinf(bDD.s.hi))
		{
			makeFinite(aDD);
			makeFinite(bDD);
			zeroNaN(cDD);
			zeroNaN(dDD);
			recalc = 1;
		}
		
		if (isinf(cDD.s.hi) || isinf(dDD.s.hi))
		{
			makeFinite(cDD);
			makeFinite(dDD);
			zeroNaN(aDD);
			zeroNaN(bDD);
			recalc = 1;
		}
		
		if (!recalc)
		{
			DD acDD = { .ld = ac };
			DD bdDD = { .ld = bd };
			DD adDD = { .ld = ad };
			DD bcDD = { .ld = bc };
			
			if (isinf(acDD.s.hi) || isinf(bdDD.s.hi) || isinf(adDD.s.hi) || isinf(bcDD.s.hi))
			{
				zeroNaN(aDD);
				zeroNaN(bDD);
				zeroNaN(cDD);
				zeroNaN(dDD);
				recalc = 1;
			}
		}
		
		if (recalc)
		{
			real.s.hi = INFINITY * (aDD.s.hi*cDD.s.hi - bDD.s.hi*dDD.s.hi);
			real.s.lo = 0.0;
			imag.s.hi = INFINITY * (aDD.s.hi*dDD.s.hi + bDD.s.hi*cDD.s.hi);
			imag.s.lo = 0.0;
		}
	}
	
	long double _Complex z;
	__real__ z = real.ld;
	__imag__ z = imag.ld;
	
	return z;
}
