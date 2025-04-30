#include <math.h>
#define __exp __ieee754_exp_fma4
#define SECTION __attribute__ ((section (".text.fma4")))

#include <sysdeps/ieee754/dbl-64/e_exp.c>
