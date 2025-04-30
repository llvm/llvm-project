#define __atan __atan_fma4
#define __add __add_fma4
#define __dbl_mp __dbl_mp_fma4
#define __mpatan __mpatan_fma4
#define __mul __mul_fma4
#define __sub __sub_fma4
#define SECTION __attribute__ ((section (".text.fma4")))

#include <sysdeps/ieee754/dbl-64/s_atan.c>
