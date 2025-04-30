#ifndef _MNTENT_H
#include <misc/mntent.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern FILE *__setmntent (const char *__file, const char *__mode);
extern struct mntent *__getmntent_r (FILE *__stream,
				     struct mntent *__result,
				     char *__buffer, int __bufsize);
extern int __addmntent (FILE *__stream, const struct mntent *__mnt);
extern int __endmntent (FILE *__stream);
extern char *__hasmntopt (const struct mntent *__mnt, const char *__opt);

libc_hidden_proto (__setmntent)
libc_hidden_proto (__getmntent_r)
libc_hidden_proto (__endmntent)
libc_hidden_proto (__hasmntopt)

# endif /* !_ISOMAC */
#endif
