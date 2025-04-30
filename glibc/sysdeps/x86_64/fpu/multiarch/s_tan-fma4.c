#define __tan __tan_fma4
#define __dbl_mp __dbl_mp_fma4
#define __mpranred __mpranred_fma4
#define __mptan __mptan_fma4
#define __sub __sub_fma4
#define SECTION __attribute__ ((section (".text.fma4")))

#include <sysdeps/ieee754/dbl-64/s_tan.c>
