#include <sgidefs.h>

#if _MIPS_SIM == _ABIO32
# error "long double fma being compiled for o32 ABI"
#endif

#include <sysdeps/ieee754/soft-fp/s_fmal.c>
