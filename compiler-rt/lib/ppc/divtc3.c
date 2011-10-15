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

long double __gcc_qadd(long double, long double);
long double __gcc_qsub(long double, long double);
long double __gcc_qmul(long double, long double);
long double __gcc_qdiv(long double, long double);

long double _Complex
__divtc3(long double a, long double b, long double c, long double d)
{
	DD cDD = { .ld = c };
	DD dDD = { .ld = d };
	
	int ilogbw = 0;
	const double logbw = logb(__builtin_fmax( __builtin_fabs(cDD.s.hi), __builtin_fabs(dDD.s.hi) ));
	
	if (isfinite(logbw))
	{
		ilogbw = (int)logbw;
		
		cDD.s.hi = scalbn(cDD.s.hi, -ilogbw);
		cDD.s.lo = scalbn(cDD.s.lo, -ilogbw);
		dDD.s.hi = scalbn(dDD.s.hi, -ilogbw);
		dDD.s.lo = scalbn(dDD.s.lo, -ilogbw);
	}
	
	const long double denom = __gcc_qadd(__gcc_qmul(cDD.ld, cDD.ld), __gcc_qmul(dDD.ld, dDD.ld));
	const long double realNumerator = __gcc_qadd(__gcc_qmul(a,cDD.ld), __gcc_qmul(b,dDD.ld));
	const long double imagNumerator = __gcc_qsub(__gcc_qmul(b,cDD.ld), __gcc_qmul(a,dDD.ld));
	
	DD real = { .ld = __gcc_qdiv(realNumerator, denom) };
	DD imag = { .ld = __gcc_qdiv(imagNumerator, denom) };
	
	real.s.hi = scalbn(real.s.hi, -ilogbw);
	real.s.lo = scalbn(real.s.lo, -ilogbw);
	imag.s.hi = scalbn(imag.s.hi, -ilogbw);
	imag.s.lo = scalbn(imag.s.lo, -ilogbw);
	
	if (isnan(real.s.hi) && isnan(imag.s.hi))
	{
		DD aDD = { .ld = a };
		DD bDD = { .ld = b };
		DD rDD = { .ld = denom };
		
		if ((rDD.s.hi == 0.0) && (!isnan(aDD.s.hi) || !isnan(bDD.s.hi)))
		{
			real.s.hi = __builtin_copysign(INFINITY,cDD.s.hi) * aDD.s.hi;
			real.s.lo = 0.0;
			imag.s.hi = __builtin_copysign(INFINITY,cDD.s.hi) * bDD.s.hi;
			imag.s.lo = 0.0;
		}
		
		else if ((isinf(aDD.s.hi) || isinf(bDD.s.hi)) && isfinite(cDD.s.hi) && isfinite(dDD.s.hi))
		{
			makeFinite(aDD);
			makeFinite(bDD);
			real.s.hi = INFINITY * (aDD.s.hi*cDD.s.hi + bDD.s.hi*dDD.s.hi);
			real.s.lo = 0.0;
			imag.s.hi = INFINITY * (bDD.s.hi*cDD.s.hi - aDD.s.hi*dDD.s.hi);
			imag.s.lo = 0.0;
		}
		
		else if ((isinf(cDD.s.hi) || isinf(dDD.s.hi)) && isfinite(aDD.s.hi) && isfinite(bDD.s.hi))
		{
			makeFinite(cDD);
			makeFinite(dDD);
			real.s.hi = __builtin_copysign(0.0,(aDD.s.hi*cDD.s.hi + bDD.s.hi*dDD.s.hi));
			real.s.lo = 0.0;
			imag.s.hi = __builtin_copysign(0.0,(bDD.s.hi*cDD.s.hi - aDD.s.hi*dDD.s.hi));
			imag.s.lo = 0.0;
		}
	}
	
	long double _Complex z;
	__real__ z = real.ld;
	__imag__ z = imag.ld;
	
	return z;
}
