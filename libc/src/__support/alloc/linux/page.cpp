#include "src/__support/alloc/page.h"
#include "include/llvm-libc-macros/stdlib-macros.h"
#include "src/__support/macros/config.h"
#include "src/__support/memory_size.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/mremap.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/getpagesize.h"
#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {

static void *alloc_hint = NULL;

void *page_allocate(size_t n_pages) {
  size_t page_size = getpagesize();
  size_t size = n_pages * page_size;
  size_t aligned_size = internal::SafeMemSize(size).align_up(page_size);

  void *ptr = mmap(&alloc_hint, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == NULL)
    return nullptr;

  alloc_hint = (void *)(((uintptr_t)ptr) + aligned_size);
  return ptr;
}

void *page_expand(void *ptr, size_t n_pages) {
  (void)ptr;
  (void)n_pages;
  return nullptr;
}

bool page_free(void *ptr, size_t n_pages) {
  (void)ptr;
  (void)n_pages;
  return false;
}

} // namespace LIBC_NAMESPACE_DECL
