#ifndef _SYS_STATFS_H
#include <io/sys/statfs.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern int __statfs (const char *__file, struct statfs *__buf);
libc_hidden_proto (__statfs)
extern int __fstatfs (int __fildes, struct statfs *__buf);
libc_hidden_proto (__fstatfs)
extern int __statfs64 (const char *__file, struct statfs64 *__buf)
     attribute_hidden;
extern int __fstatfs64 (int __fildes, struct statfs64 *__buf);

# endif /* !_ISOMAC */
#endif
