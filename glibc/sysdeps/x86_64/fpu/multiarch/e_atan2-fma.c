#define __ieee754_atan2 __ieee754_atan2_fma
#define __add __add_fma
#define __dbl_mp __dbl_mp_fma
#define __dvd __dvd_fma
#define __mpatan2 __mpatan2_fma
#define __mul __mul_fma
#define __sub __sub_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/e_atan2.c>
