#define __tan __tan_fma
#define __dbl_mp __dbl_mp_fma
#define __mpranred __mpranred_fma
#define __mptan __mptan_fma
#define __sub __sub_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/s_tan.c>
