#include <tst-tls5.h>

#ifdef TLS_REGISTER
static __thread int c;
TLS_REGISTER (c)
#endif
