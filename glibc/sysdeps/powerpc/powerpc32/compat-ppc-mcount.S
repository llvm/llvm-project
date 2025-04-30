#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_15)

	compat_text_section
# define _mcount __compat_mcount
# include "ppc-mcount.S"
# undef _mcount

compat_symbol (libc, __compat_mcount, _mcount, GLIBC_2_0)
#endif
