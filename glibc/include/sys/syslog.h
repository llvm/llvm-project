#ifndef _LIBC_SYS_SYSLOG_H
#define _LIBC_SYS_SYSLOG_H 1
#include <misc/sys/syslog.h>
#ifndef _ISOMAC

libc_hidden_ldbl_proto (syslog)

/* __vsyslog_internal uses the same mode_flags bits as
   __v*printf_internal; see libio/libioP.h.  */
extern void __vsyslog_internal (int pri, const char *fmt, __gnuc_va_list ap,
				unsigned int mode_flags)
     attribute_hidden
     __attribute__ ((__format__ (__printf__, 2, 0)));

#endif /* _ISOMAC */
#endif /* syslog.h */
