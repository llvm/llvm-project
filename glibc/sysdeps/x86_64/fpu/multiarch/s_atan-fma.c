#define __atan __atan_fma
#define __add __add_fma
#define __dbl_mp __dbl_mp_fma
#define __mpatan __mpatan_fma
#define __mul __mul_fma
#define __sub __sub_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/s_atan.c>
