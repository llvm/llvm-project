#include <stdarg.h>
#include <sys/syscall.h> // For syscall numbers.

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/sys/ioctl/ioctl.h"

#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, ioctl, (int fd, int req, ...)) {
  void *arg;
  va_list ap;
  va_start(ap, req);
  arg = va_arg(ap, void *);
  va_end(ap);
  int ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_ioctl, fd, req, arg);
  // FIXME(@izaakschroeder): There is probably more to do here.
  // See: https://github.com/kraj/musl/blob/kraj/master/src/misc/ioctl.c
  return ret;
}

} // namespace LIBC_NAMESPACE
