#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sysdep.h>

__attribute__((noinline))
int __mprotect_internal(void *start, size_t len, int prot)
{
  return INLINE_SYSCALL_CALL (mprotect, start, len, prot);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
int __mprotect_nextsilicon(void *start, size_t len, int prot)
{
  return __mprotect_internal(start, len, prot);
}

int __mprotect(void *start, size_t len, int prot)
{
  return __mprotect_nextsilicon(start, len, prot);
}

libc_hidden_def (__mprotect_internal)

libc_hidden_def (__mprotect)
weak_alias (__mprotect, mprotect)
