#include "src/stdlib/aligned_alloc.h"
#include "src/__support/alloc/alloc.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, aligned_alloc, (size_t alignment, size_t size)) {
  return allocator->alloc(alignment, size);
}

} // namespace LIBC_NAMESPACE_DECL
