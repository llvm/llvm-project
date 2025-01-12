#include "src/__support/alloc/arena.h"
#include "src/__support/alloc/page.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/page_size.h"
#include "src/__support/memory_size.h"
#include "src/string/memmove.h"
#include "src/unistd/getpagesize.h"

namespace LIBC_NAMESPACE_DECL {

void *arena_allocate(BaseAllocator *base, size_t alignment, size_t size) {
  ArenaAllocator *self = reinterpret_cast<ArenaAllocator *>(base);

  if (self->buffer == nullptr) {
    self->buffer = reinterpret_cast<uint8_t *>(page_allocate(1));
    self->buffer_size = self->get_page_size();
  }

  uintptr_t curr_ptr = (uintptr_t)self->buffer + (uintptr_t)self->curr_offset;
  uintptr_t offset = internal::align_forward<uintptr_t>(curr_ptr, alignment);
  offset -= (uintptr_t)self->buffer;

  if (offset + size > self->buffer_size) {
    self->buffer = reinterpret_cast<uint8_t *>(
        page_expand(self->buffer, self->buffer_size / self->get_page_size()));
    self->buffer_size += self->get_page_size();
  }

  if (offset + size <= self->buffer_size) {
    void *ptr = &self->buffer[offset];
    self->prev_offset = offset;
    self->curr_offset = offset + size;
    return ptr;
  }
  return nullptr;
}

void *arena_expand(BaseAllocator *base, void *ptr, size_t alignment,
                   size_t size) {
  ArenaAllocator *self = reinterpret_cast<ArenaAllocator *>(base);

  if (self->buffer + self->prev_offset == ptr) {
    self->curr_offset = self->prev_offset + size;
    return ptr;
  } else {
    void *new_mem = arena_allocate(base, alignment, size);
    memmove(new_mem, ptr, size);
    return new_mem;
  }
  return nullptr;
}

bool arena_free(BaseAllocator *base, void *ptr) {
  (void)base;
  (void)ptr;
  return true;
}

size_t ArenaAllocator::get_page_size() {
  if (page_size == LIBC_PAGE_SIZE_SYSTEM) {
    page_size = getpagesize();
  }
  return page_size;
}

static ArenaAllocator default_arena_allocator(LIBC_PAGE_SIZE,
                                              2 * sizeof(void *));
BaseAllocator *arena_allocator = &default_arena_allocator;

} // namespace LIBC_NAMESPACE_DECL
