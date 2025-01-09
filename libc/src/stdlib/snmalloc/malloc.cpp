#include "src/stdlib/malloc.h"
#include "snmalloc/snmalloc.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void *, malloc, (size_t size)) {
  return snmalloc::libc::malloc(size);
}
} // namespace LIBC_NAMESPACE_DECL
