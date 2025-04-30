#include <math.h>
#define __log __ieee754_log_avx
#define SECTION __attribute__ ((section (".text.avx")))

#include <sysdeps/ieee754/dbl-64/e_log.c>
