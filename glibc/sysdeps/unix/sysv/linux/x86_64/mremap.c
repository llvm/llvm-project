#include <errno.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sysdep.h>

__attribute__((noinline))
void *__mremap_internal(void *old_addr, size_t old_len, size_t new_len, int flags, void *new_addr)
{
  return (void *)INLINE_SYSCALL_CALL (mremap, old_addr, old_len, new_len, flags, new_addr);
}

/* This symbol will be hijacked by nextsilicon when offloaded */
__attribute__((noinline))
void *__mremap_nextsilicon(void *old_addr, size_t old_len, size_t new_len, int flags, void *new_addr)
{
  return __mremap_internal(old_addr, old_len, new_len, flags, new_addr);
}

void *__mremap(void *old_addr, size_t old_len, size_t new_len, int flags, ...)
{
  void *new_addr = NULL;
  va_list ap;

  if (flags & MREMAP_FIXED) {
    va_start(ap, flags);
    new_addr = va_arg(ap, void *);
    va_end(ap);
  }

  return __mremap_nextsilicon(old_addr, old_len, new_len, flags, new_addr);
}

libc_hidden_def (__mremap_internal)

libc_hidden_def (__mremap)
weak_alias (__mremap, mremap)
