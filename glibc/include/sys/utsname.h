#ifndef	_SYS_UTSNAME_H
#include <posix/sys/utsname.h>

#ifndef _ISOMAC
extern int __uname (struct utsname *__name);

libc_hidden_proto (uname)
libc_hidden_proto (__uname)

# if IS_IN (rtld)
extern __typeof (__uname) __uname attribute_hidden;
# endif
#endif
#endif
