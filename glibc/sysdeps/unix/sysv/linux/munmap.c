#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sysdep.h>

__attribute__((noinline))
int __munmap_internal(void *start, size_t len)
{
  return INLINE_SYSCALL_CALL (munmap, start, len);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
int __munmap_nextsilicon(void *start, size_t len)
{
  return __munmap_internal(start, len);
}

int __munmap(void *start, size_t len)
{
  return __munmap_nextsilicon(start, len);
}

libc_hidden_def (__munmap_internal)

libc_hidden_def (__munmap)
weak_alias (__munmap, munmap)
