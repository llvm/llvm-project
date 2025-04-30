#include <math.h>
#define __pow __ieee754_pow_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/e_pow.c>
