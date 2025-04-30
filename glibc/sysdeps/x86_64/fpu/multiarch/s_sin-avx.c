#define __cos __cos_avx
#define __sin __sin_avx
#define SECTION __attribute__ ((section (".text.avx")))

#include <sysdeps/ieee754/dbl-64/s_sin.c>
