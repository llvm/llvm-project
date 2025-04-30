#include <tst-tls5.h>

#ifdef TLS_REGISTER
static __thread int e1 = 24;
static __thread char e2 [32] __attribute__ ((aligned (64)));
TLS_REGISTER (e1)
TLS_REGISTER (e2)
#endif
