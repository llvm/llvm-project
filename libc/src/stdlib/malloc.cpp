#include "src/stdlib/malloc.h"
#include "src/__support/alloc/alloc.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/aligned_alloc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  return aligned_alloc(allocator->default_alignment, size);
}

} // namespace LIBC_NAMESPACE_DECL
