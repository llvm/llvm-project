/* The inline __ieee754_sqrt is not correctly rounding; it's OK for
   most internal uses in glibc, but not for sqrt itself.  */
#define NO_MATH_REDIRECT
#define __ieee754_sqrt __avoid_ieee754_sqrt
#include <math.h>
#include <math_private.h>
#undef __ieee754_sqrt
extern double __ieee754_sqrt (double);
#include <math-type-macros-double.h>
#include <w_sqrt_template.c>
