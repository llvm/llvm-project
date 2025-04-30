#include <math.h>
#define __exp __ieee754_exp_avx
#define SECTION __attribute__ ((section (".text.avx")))

#include <sysdeps/ieee754/dbl-64/e_exp.c>
