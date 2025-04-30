#ifndef _SYS_FILE_H
#include <misc/sys/file.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */
extern int __flock (int __fd, int __operation);

# endif /* !_ISOMAC */
#endif
