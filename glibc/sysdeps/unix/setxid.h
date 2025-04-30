#include <sysdep.h>

#define INLINE_SETXID_SYSCALL(name, nr, args...) \
  INLINE_SYSCALL (name, nr, args)
