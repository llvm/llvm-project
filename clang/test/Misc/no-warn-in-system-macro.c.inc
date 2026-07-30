extern int __isnanf(float f);
extern int __isnan(double f);
extern int __isnanl(long double f);
#define isnan(x) \
	(sizeof (x) == sizeof (float)                \
	? __isnanf (x)                    \
	: sizeof (x) == sizeof (double)               \
	? __isnan (x) : __isnanl (x))

