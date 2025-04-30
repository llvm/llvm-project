#define __cos __cos_fma4
#define __sin __sin_fma4
#define __docos __docos_fma4
#define __dubsin __dubsin_fma4
#define __mpcos __mpcos_fma4
#define __mpcos1 __mpcos1_fma4
#define __mpsin __mpsin_fma4
#define __mpsin1 __mpsin1_fma4
#define SECTION __attribute__ ((section (".text.fma4")))

#include <sysdeps/ieee754/dbl-64/s_sin.c>
