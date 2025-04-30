#include <shlib-compat.h>

#define __mcount_internal ___mcount_internal

#include <gmon/mcount.c>

#undef __mcount_internal

/* __mcount_internal was added in glibc 2.15 with version GLIBC_PRIVATE,
   but it should have been put in version GLIBC_2.15.  Mark the
   GLIBC_PRIVATE version obsolete and add it to GLIBC_2.16 instead.  */
versioned_symbol (libc, ___mcount_internal, __mcount_internal, GLIBC_2_16);

#if SHLIB_COMPAT (libc, GLIBC_2_15, GLIBC_2_16)
strong_alias (___mcount_internal, ___mcount_internal_private);
symbol_version (___mcount_internal_private, __mcount_internal, GLIBC_PRIVATE);
#endif
