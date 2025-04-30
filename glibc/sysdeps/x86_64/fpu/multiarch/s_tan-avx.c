#define __tan __tan_avx
#define __dbl_mp __dbl_mp_avx
#define __sub __sub_avx
#define SECTION __attribute__ ((section (".text.avx")))

#include <sysdeps/ieee754/dbl-64/s_tan.c>
