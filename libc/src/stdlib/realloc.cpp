#include "src/stdlib/realloc.h"
#include "src/__support/alloc/alloc.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, realloc, (void *ptr, size_t size)) {
  return allocator->expand(ptr, allocator->default_alignment, size);
}

} // namespace LIBC_NAMESPACE_DECL
