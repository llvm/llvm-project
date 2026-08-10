// Test that LargeAllocator unpoisons memory before releasing it to the OS.
// RUN: %clangxx_asan %s -o %t
// The memory is released only when the deallocated chunk leaves the quarantine,
// otherwise the mmap(p, ...) call overwrites the malloc header.
// RUN: %env_asan_opts=quarantine_size_mb=0 %run %t

#include <assert.h>
#include <string.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <malloc.h>
void *my_memalign(size_t boundary, size_t size) {
  return memalign(boundary, size);
}
#else
void *my_memalign(size_t boundary, size_t size) {
  void *p;
  posix_memalign(&p, boundary, size);
  return p;
}
#endif

int main() {
  const long kPageSize = sysconf(_SC_PAGESIZE);
  void *p = my_memalign(kPageSize, 1024 * 1024);
  free(p);

  char *q = (char *)mmap(p, kPageSize, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
  assert(q == p);

  memset(q, 42, kPageSize);

  munmap(q, kPageSize);
  return 0;
}
