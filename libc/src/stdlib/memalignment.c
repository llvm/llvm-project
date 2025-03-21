#include "src/stdlib/memalignment.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, memalignment, (const void *p)) {
  if (p == NULL) {
    return 0;
  }

  uintptr_t addr = (uintptr_t)p;

  // Find the rightmost set bit, which represents the maximum alignment
  // The alignment is a power of two, so we need to find the largest
  // power of two that divides the address
  return addr & (~addr + 1);
}

} // namespace LIBC_NAMESPACE_DECL
