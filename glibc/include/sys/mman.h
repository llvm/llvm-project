#ifndef _SYS_MMAN_H
#include <misc/sys/mman.h>
#include <_ns_unmigratable.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern void *__mmap (void *__addr, size_t __len, int __prot,
		     int __flags, int __fd, __off_t __offset);
libc_hidden_proto (__mmap)
extern void *__mmap64 (void *__addr, size_t __len, int __prot,
		       int __flags, int __fd, __off64_t __offset);
libc_hidden_proto (__mmap64)
extern int __munmap (void *__addr, size_t __len);
libc_hidden_proto (__munmap)
extern int __mprotect (void *__addr, size_t __len, int __prot);
libc_hidden_proto (__mprotect)

extern int __madvise (void *__addr, size_t __len, int __advice);
libc_hidden_proto (__madvise)

/* This one is Linux specific.  */
extern void *__mremap (void *__addr, size_t __old_len,
		       size_t __new_len, int __flags, ...);
libc_hidden_proto (__mremap)

# if IS_IN (rtld)
#  include <dl-mman.h>
# endif
#endif

#endif
