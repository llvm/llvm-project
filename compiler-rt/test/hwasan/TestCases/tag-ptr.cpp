// RUN: %clangxx_hwasan -O0 %s -o %t && %run %t

#include <assert.h>
#include <memory>
#include <sanitizer/hwasan_interface.h>
#include <set>
#include <stdio.h>

int main() {
  auto p = std::make_unique<char>();
  std::set<void *> ptrs;
  for (unsigned i = 0;; ++i) {
    void *ptr = __hwasan_tag_pointer(p.get(), i);
    if (!ptrs.insert(ptr).second)
      break;
    fprintf(stderr, "%p, %u, %u\n", ptr, i, __hwasan_get_tag_from_pointer(ptr));
    assert(__hwasan_get_tag_from_pointer(ptr) == i);
  }
#ifdef __x86_64__
  assert(ptrs.size() == 8 || ptrs.size() == 64);
#else
  assert(ptrs.size() == 256);
#endif
}
