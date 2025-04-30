#ifndef	_STDIO_EXT_H
#include <stdio-common/stdio_ext.h>

# ifndef _ISOMAC

libc_hidden_proto (__fsetlocking)

#define __fsetlocking(fp, type) \
  ({ int __result = ((fp->_flags & _IO_USER_LOCK)			\
		     ? FSETLOCKING_BYCALLER : FSETLOCKING_INTERNAL);	\
									\
     if (type != FSETLOCKING_QUERY)					\
       {								\
	 fp->_flags &= ~_IO_USER_LOCK;					\
	 if (type == FSETLOCKING_BYCALLER)				\
	   fp->_flags |= _IO_USER_LOCK;					\
       }								\
									\
     __result;								\
  })

# endif /* !_ISOMAC */
#endif /* stdio_ext.h */
