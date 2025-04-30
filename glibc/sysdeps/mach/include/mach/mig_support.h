#ifndef _MACH_MIG_SUPPORT_H
#include_next <mach/mig_support.h>
#ifndef _ISOMAC
libc_hidden_proto (__mig_get_reply_port)
libc_hidden_proto (__mig_dealloc_reply_port)
libc_hidden_proto (__mig_init)

#ifdef _LIBC
# include <libc-symbols.h>

# if defined USE_MULTIARCH && (IS_IN (libmachuser) || IS_IN (libhurduser))
/* Avoid directly calling ifunc-enabled memcpy or strpcpy,
   because they would introduce a relocation loop between lib*user and
   libc.so.  */
#  define memcpy(dest, src, n) __mig_memcpy(dest, src, n)
# endif
#endif

#endif
#endif
