#ifndef LLVM_SYS_MMAN_LINUX_GENERIC_PKEY_COMMON_H_
#define LLVM_SYS_MMAN_LINUX_GENERIC_PKEY_COMMON_H_

#include "hdr/errno_macros.h" // For ENOSYS
#include "src/__support/common.h"
#include "src/__support/error_or.h"

namespace LIBC_NAMESPACE_DECL {
namespace pkey_common {

LIBC_INLINE ErrorOr<int> pkey_get(int pkey) {
  (void)pkey;
  return Error(ENOSYS);
}

LIBC_INLINE ErrorOr<int> pkey_set(int pkey, unsigned int access_rights) {
  (void)pkey;
  (void)access_rights;
  return Error(ENOSYS);
}

} // namespace pkey_common
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_SYS_MMAN_LINUX_GENERIC_PKEY_COMMON_H_
