#define __atan __atan_avx
#define __add __add_avx
#define __dbl_mp __dbl_mp_avx
#define __mul __mul_avx
#define __sub __sub_avx
#define SECTION __attribute__ ((section (".text.avx")))

#include <sysdeps/ieee754/dbl-64/s_atan.c>
