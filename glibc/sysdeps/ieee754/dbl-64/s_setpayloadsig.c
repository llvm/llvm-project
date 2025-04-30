#define SIG 1
#define FUNC __setpayloadsig
#include <s_setpayload_main.c>
libm_alias_double (__setpayloadsig, setpayloadsig)
