#define __ieee754_acos __ieee754_acos_fma4
#define __ieee754_asin __ieee754_asin_fma4
#define __doasin __doasin_fma4
#define __docos __docos_fma4
#define __dubcos __dubcos_fma4
#define __dubsin __dubsin_fma4
#define SECTION __attribute__ ((section (".text.fma4")))

#include <sysdeps/ieee754/dbl-64/e_asin.c>
