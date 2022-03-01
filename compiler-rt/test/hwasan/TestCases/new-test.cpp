// Test basic new functionality.
// RUN: %clangxx_hwasan -std=c++17 %s -o %t
// RUN: %run %t

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <sanitizer/allocator_interface.h>
#include <sanitizer/hwasan_interface.h>

int main() {
  __hwasan_enable_allocator_tagging();

  size_t volatile n = 0;
  char *a1 = new char[n];
  assert(a1 != nullptr);
  assert(__sanitizer_get_allocated_size(a1) == 0);
  delete[] a1;

#if defined(__cpp_aligned_new) &&                                              \
    (!defined(__GLIBCXX__) ||                                                  \
     (defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 7))
  // Aligned new/delete
  constexpr auto kAlign = std::align_val_t{8};
  void *a2 = ::operator new(4, kAlign);
  assert(a2 != nullptr);
  assert(reinterpret_cast<uintptr_t>(a2) % static_cast<uintptr_t>(kAlign) == 0);
  assert(__sanitizer_get_allocated_size(a2) >= 4);
  ::operator delete(a2, kAlign);
#endif
}
