#ifndef _SYS_UIO_H
#include <misc/sys/uio.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern ssize_t __readv (int __fd, const struct iovec *__iovec,
			int __count);
libc_hidden_proto (__readv)
extern ssize_t __writev (int __fd, const struct iovec *__iovec,
			 int __count);
libc_hidden_proto (__writev)

/* Used for p{read,write}{v64}v2 implementation.  */
libc_hidden_proto (preadv)
libc_hidden_proto (preadv64)
libc_hidden_proto (pwritev)
libc_hidden_proto (pwritev64)
#endif
#endif
