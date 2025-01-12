#include "src/stdlib/calloc.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/malloc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, calloc, (size_t nmeb, size_t size)) {
  return malloc(nmeb * size);
}

} // namespace LIBC_NAMESPACE_DECL
