#define __ieee754_acos __ieee754_acos_fma
#define __ieee754_asin __ieee754_asin_fma
#define __doasin __doasin_fma
#define __docos __docos_fma
#define __dubcos __dubcos_fma
#define __dubsin __dubsin_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/e_asin.c>
