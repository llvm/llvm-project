// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -g -DPOSITIVE %s -o %t && not %run %t |& FileCheck %s

#include <obstack.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>

static void *obstack_chunk_alloc(size_t sz) {
  return malloc(sz);
}

static void obstack_chunk_free(void *p) {
  free(p);
}

int main(void) {
  obstack obs;
  obstack_init(&obs);
  for (size_t sz = 16; sz < 0xFFFF; sz *= 2) {
    void *p = obstack_alloc(&obs, sz);
    int data[10] = {0};
    obstack_grow(&obs, &data, sizeof(data));
    obstack_blank(&obs, sz);
    obstack_grow(&obs, &data, sizeof(data));
    obstack_int_grow(&obs, 13);
    p = obstack_finish(&obs);
#ifdef POSITIVE
    if (sz == 4096) {
      __msan_check_mem_is_initialized(p, sizeof(data));
      __msan_check_mem_is_initialized(p, sizeof(data) + 1);
    }
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: #0 0x{{.*}} in main{{.*}}obstack.cc:[[@LINE-3]]
#endif
  }
  obstack_free(&obs, 0);
}
