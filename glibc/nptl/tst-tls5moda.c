#include <tst-tls5.h>

#ifdef TLS_REGISTER
static __thread char a [32] __attribute__ ((aligned (64)));
TLS_REGISTER (a)
#endif
