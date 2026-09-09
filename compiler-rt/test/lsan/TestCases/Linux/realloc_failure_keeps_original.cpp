// Verifies that a failing realloc() leaves the original allocation untouched:
// it keeps its contents, its reported size and its allocation stack, and no
// free hook is reported for it.
//
// Linux only: the setup below reads /proc/self/statm to size the address-space
// cap that makes the replacement allocation fail.
//
// RUN: %clangxx_lsan %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone

#include <sanitizer/allocator_interface.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>

static void *g_freed;
static void *g_alloced;

static void OnMalloc(const volatile void *ptr, size_t) {
  g_alloced = (void *)ptr;
}

static void OnFree(const volatile void *ptr) { g_freed = (void *)ptr; }

// Caps the address space just above the current usage so that the next large
// mmap fails inside the allocator instead of being rejected up front by
// max_allocation_size_mb.
static void CapAddressSpace() {
  FILE *f = fopen("/proc/self/statm", "r");
  assert(f);
  unsigned long vsz_pages = 0;
  int scanned = fscanf(f, "%lu", &vsz_pages);
  fclose(f);
  assert(scanned == 1);

  rlimit rl;
  int res = getrlimit(RLIMIT_AS, &rl);
  assert(res == 0);
  rl.rlim_cur = (rlim_t)vsz_pages * getpagesize() + (64UL << 20);
  res = setrlimit(RLIMIT_AS, &rl);
  assert(res == 0);
}

int main() {
  const size_t kSize = 100;
  char *p = (char *)malloc(kSize);
  assert(p);
  memset(p, 'a', kSize);

  // Install the hooks only after CapAddressSpace(), which itself allocates.
  CapAddressSpace();
  int installed = __sanitizer_install_malloc_and_free_hooks(OnMalloc, OnFree);
  assert(installed);

  void *q = realloc(p, 512UL << 20);
  assert(q == NULL);

  // The failed realloc must report neither a free of p nor a new allocation.
  assert(g_freed == nullptr);
  assert(g_alloced == nullptr);

  assert(__sanitizer_get_ownership(p));
  assert(__sanitizer_get_allocated_size(p) == kSize);
  for (size_t i = 0; i < kSize; ++i)
    assert(p[i] == 'a');

  fprintf(stderr, "original allocation survived realloc failure\n");
  free(p);
  assert(g_freed == p);
  fprintf(stderr, "freed once\n");
  return 0;
}

// CHECK: original allocation survived realloc failure
// CHECK: freed once
