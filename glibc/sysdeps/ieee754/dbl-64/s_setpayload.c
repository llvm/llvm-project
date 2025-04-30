#define SIG 0
#define FUNC __setpayload
#include <s_setpayload_main.c>
libm_alias_double (__setpayload, setpayload)
