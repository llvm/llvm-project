#include <shlib-compat.h>

#include <sysdeps/unix/sysv/linux/send.c>

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_17, GLIBC_2_34)
/* libpthread compat symbol: AArch64 used the generic version without the
   libc_hidden_def which lead in a non existent __send symbol in libc.so.  */
compat_symbol (libc, __libc_send, __send, GLIBC_2_17);
#endif
