#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sysdep.h>

__attribute__((noinline))
int __madvise_internal(void *start, size_t len, int advice)
{
  return INLINE_SYSCALL_CALL (madvise, start, len, advice);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
int __madvise_nextsilicon(void *addr, size_t len, int advice)
{
  return __madvise_internal(addr, len, advice);
}

int __madvise(void *addr, size_t len, int advice)
{
  return __madvise_nextsilicon(addr, len, advice);
}

libc_hidden_def (__madvise_internal)

libc_hidden_def (__madvise)
weak_alias (__madvise, madvise)
