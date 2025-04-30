#define __cos __cos_fma
#define __sin __sin_fma
#define __docos __docos_fma
#define __dubsin __dubsin_fma
#define __mpcos __mpcos_fma
#define __mpcos1 __mpcos1_fma
#define __mpsin __mpsin_fma
#define __mpsin1 __mpsin1_fma
#define SECTION __attribute__ ((section (".text.fma")))

#include <sysdeps/ieee754/dbl-64/s_sin.c>
