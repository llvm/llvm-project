#include "src/sys/time/utimes.h"

#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, utimes, (const char *, const struct timeval[2])) {
  return EINVAL;
}

} // namespace LIBC_NAMESPACE
