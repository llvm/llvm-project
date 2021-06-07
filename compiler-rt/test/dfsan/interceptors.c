// RUN: %clang_dfsan -mllvm -dfsan-fast-16-labels -mllvm -dfsan-combine-pointer-labels-on-load=false %s -o %t && %run %t
// RUN: %clang_dfsan -DORIGIN_TRACKING -mllvm -dfsan-fast-16-labels -mllvm -dfsan-track-origins=1 -mllvm -dfsan-combine-pointer-labels-on-load=false %s -o %t && %run %t
//
// Tests custom implementations of various glibc functions.
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <malloc.h>
#include <stdlib.h>
#include <sys/mman.h>

#define ASSERT_ZERO_LABEL(data) \
  assert(0 == dfsan_get_label((long) (data)))

#define ASSERT_READ_ZERO_LABEL(ptr, size) \
  assert(0 == dfsan_read_label(ptr, size))

const int kAlignment = 8;
const int kSize = 16;

void test_aligned_alloc() {
  char *p = (char *) aligned_alloc(kAlignment, kSize);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test_calloc() {
  char *p = (char *) calloc(kSize, 1);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test_cfree() {
  // The current glibc does not support cfree.
}

void test_free() {
  char *p = (char *) malloc(kSize);
  dfsan_set_label(1, p, kSize);
  free(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
}

void test_mallinfo() {
  struct mallinfo mi = mallinfo();
  for (int i = 0; i < sizeof(struct mallinfo); ++i) {
    char c = ((char *)(&mi))[i];
    assert(!c);
    ASSERT_ZERO_LABEL(c);
  }
}

void test_malloc() {
  char *p = (char *) malloc(kSize);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test_malloc_stats() {
  // Only ensures it does not crash. Our interceptor of malloc_stats is empty.
  malloc_stats();
}

void test_malloc_usable_size() {
  char *p = (char *) malloc(kSize);
  size_t s = malloc_usable_size(p);
  assert(s == kSize);
  ASSERT_ZERO_LABEL(s);
  free(p);
}

void test_mallopt() {
  int r = mallopt(0, 0);
  assert(!r);
  ASSERT_ZERO_LABEL(r);
}

void test_memalign() {
  char *p = (char *) memalign(kAlignment, kSize);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test_mmap() {
  char *p = mmap(NULL, kSize, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  char val = 0xff;
  dfsan_set_label(1, &val, sizeof(val));
  memset(p, val, kSize);
  p = mmap(p, kSize, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  munmap(p, kSize);
}

void test_mmap64() {
  // The current glibc does not support mmap64.
}

void test_unmmap() {
  char *p = mmap(NULL, kSize, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  char val = 0xff;
  dfsan_set_label(1, &val, sizeof(val));
  memset(p, val, kSize);
  munmap(p, kSize);
  ASSERT_READ_ZERO_LABEL(p, kSize);
}

void test_posix_memalign() {
  char *p;
  dfsan_set_label(1, &p, sizeof(p));
  int r = posix_memalign((void **)&p, kAlignment, kSize);
  assert(!r);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test_pvalloc() {
  char *p = (char *) pvalloc(kSize);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test_realloc() {
  char *p = (char *) malloc(kSize);

  char *q = (char *) realloc(p, kSize * 2);
  ASSERT_ZERO_LABEL(q);
  ASSERT_READ_ZERO_LABEL(q, kSize * 2);

  char *x = (char *) realloc(q, kSize);
  ASSERT_ZERO_LABEL(x);
  ASSERT_READ_ZERO_LABEL(x, kSize);

  free(x);
}

void test_reallocarray() {
  // The current glibc does not support reallocarray.
}

void test_valloc() {
  char *p = (char *) valloc(kSize);
  ASSERT_ZERO_LABEL(p);
  ASSERT_READ_ZERO_LABEL(p, kSize);
  free(p);
}

void test___libc_memalign() {
  // The current glibc does not support __libc_memalign.
}

void test___tls_get_addr() {
  // The current glibc does not support __tls_get_addr.
}

int main(void) {
  // With any luck this sequence of calls will cause allocators to return the
  // same pointer. This is probably the best we can do to test these functions.
  test_aligned_alloc();
  test_calloc();
  test_cfree();
  test_free();
  test_mallinfo();
  test_malloc();
  test_malloc_stats();
  test_malloc_usable_size();
  test_mallopt();
  test_memalign();
  test_mmap();
  test_mmap64();
  test_unmmap();
  test_posix_memalign();
  test_pvalloc();
  test_realloc();
  test_reallocarray();
  test_valloc();
  test___libc_memalign();
  test___tls_get_addr();
}
