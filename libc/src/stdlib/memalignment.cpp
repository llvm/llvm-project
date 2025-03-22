#include "src/stdlib/memalignment.h"
#include "src/__support/macros/config.h"
#include "src/__support/CPP/bit.h"
namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, memalignment, (const void *p)) {
  if (p == nullptr)
    return 0;

  uintptr_t addr = reinterpret_cast<uintptr_t>(p);

  return 1 << cpp::countr_zero(addr);
}

} // namespace LIBC_NAMESPACE_DECL
