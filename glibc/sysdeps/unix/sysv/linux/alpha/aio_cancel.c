#include <shlib-compat.h>

#define aio_cancel64 XXX
#include <aio.h>
#undef aio_cancel64
#include <errno.h>

extern __typeof (aio_cancel) __new_aio_cancel;
extern __typeof (aio_cancel) __old_aio_cancel;

#define __aio_cancel	__new_aio_cancel

#include <rt/aio_cancel.c>

#undef __aio_cancel
versioned_symbol (libc, __new_aio_cancel, aio_cancel, GLIBC_2_34);
versioned_symbol (libc, __new_aio_cancel, aio_cancel64, GLIBC_2_34);
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3, GLIBC_2_34)
compat_symbol (librt, __new_aio_cancel, aio_cancel, GLIBC_2_3);
compat_symbol (librt, __new_aio_cancel, aio_cancel64, GLIBC_2_3);
#endif

#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_3)

#undef ECANCELED
#define __aio_cancel	__old_aio_cancel
#define ECANCELED	125

#include <rt/aio_cancel.c>

#undef __aio_cancel
compat_symbol (librt, __old_aio_cancel, aio_cancel, GLIBC_2_1);
compat_symbol (librt, __old_aio_cancel, aio_cancel64, GLIBC_2_1);

#endif
