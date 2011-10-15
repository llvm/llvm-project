#ifndef __DD_HEADER
#define __DD_HEADER

#include <stdint.h>

typedef union {
	long double ld;
	struct {
		double hi;
		double lo;
	}s;
}DD;

typedef union { 
	double d;
	uint64_t x;
} doublebits;

#define LOWORDER(xy,xHi,xLo,yHi,yLo) \
	(((((xHi)*(yHi) - (xy)) + (xHi)*(yLo)) + (xLo)*(yHi)) + (xLo)*(yLo))

static inline double __attribute__((always_inline))
fabs(double x)
{
	doublebits result = { .d = x };
	result.x &= UINT64_C(0x7fffffffffffffff);
	return result.d;
}

static inline double __attribute__((always_inline))
high26bits(double x)
{
	doublebits result = { .d = x };
	result.x &= UINT64_C(0xfffffffff8000000);
	return result.d;
}

static inline int __attribute__((always_inline))
different_sign(double x, double y)
{
	doublebits xsignbit = { .d = x }, ysignbit = { .d = y };
	int result = (int)(xsignbit.x >> 63) ^ (int)(ysignbit.x >> 63);
	return result;
}

#endif /* __DD_HEADER */
