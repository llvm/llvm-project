#define __ieee754_atan2 __ieee754_atan2_fma4
#define __add __add_fma4
#define __dbl_mp __dbl_mp_fma4
#define __dvd __dvd_fma4
#define __mpatan2 __mpatan2_fma4
#define __mul __mul_fma4
#define __sub __sub_fma4
#define SECTION __attribute__ ((section (".text.fma4")))

#include <sysdeps/ieee754/dbl-64/e_atan2.c>
