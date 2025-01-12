#include "src/__support/alloc/page.h"
#include "src/__suport/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" void *__llvm_libc_page_allocate(size_t n_pages);
extern "C" void *__llvm_libc_page_expand(void *ptr, size_t n_pages);
extern "C" bool __llvm_libc_page_free(void *ptr, size_t n_pages);

void *page_allocate(size_t n_pages) {
  return __llvm_libc_page_allocate(n_pages);
}

void *page_expand(void *ptr, size_t n_pages) {
  return __llvm_libc_page_expand(ptr, n_pages);
}

bool page_free(void *ptr, size_t n_pages) {
  return __llvm_libc_page_free(ptr, n_pages);
}

} // namespace LIBC_NAMESPACE_DECL
