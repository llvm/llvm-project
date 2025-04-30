#ifndef _FCNTL_H
#include <io/fcntl.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern int __open64 (const char *__file, int __oflag, ...);
libc_hidden_proto (__open64)
extern int __libc_open64 (const char *file, int oflag, ...);
extern int __libc_open (const char *file, int oflag, ...);
libc_hidden_proto (__libc_open)
extern int __libc_fcntl (int fd, int cmd, ...);
libc_hidden_proto (__libc_fcntl)
extern int __fcntl64_nocancel_adjusted (int fd, int cmd, void *arg)
   attribute_hidden;
extern int __libc_fcntl64 (int fd, int cmd, ...);
libc_hidden_proto (__libc_fcntl64)
extern int __open (const char *__file, int __oflag, ...);
libc_hidden_proto (__open)
extern int __fcntl (int __fd, int __cmd, ...);
libc_hidden_proto (__fcntl)
extern int __fcntl64 (int __fd, int __cmd, ...) attribute_hidden;
libc_hidden_proto (__fcntl64)
extern int __openat (int __fd, const char *__file, int __oflag, ...)
  __nonnull ((2));
libc_hidden_proto (__openat)
extern int __openat64 (int __fd, const char *__file, int __oflag, ...)
  __nonnull ((2));
libc_hidden_proto (__openat64)

extern int __open_2 (const char *__path, int __oflag);
extern int __open64_2 (const char *__path, int __oflag);
extern int __openat_2 (int __fd, const char *__path, int __oflag);
extern int __openat64_2 (int __fd, const char *__path, int __oflag);


#if IS_IN (rtld)
#  include <dl-fcntl.h>
#endif

/* Flag determining whether the *at system calls are available.  */
extern int __have_atfcts attribute_hidden;
#endif

#endif
