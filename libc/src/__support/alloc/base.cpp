#include "src/__support/alloc/base.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

void *BaseAllocator::alloc(size_t alignment, size_t size) {
  return impl_alloc(this, alignment, size);
}

void *BaseAllocator::expand(void *ptr, size_t alignment, size_t size) {
  return impl_expand(this, ptr, alignment, size);
}

bool BaseAllocator::free(void *ptr) { return impl_free(this, ptr); }

} // namespace LIBC_NAMESPACE_DECL
