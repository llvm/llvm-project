#include <math.h>
#define __log __ieee754_log_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/e_log.c>
