// Enabling double-free detection must not change what a free hook observes.
// The hook has to run before the chunk state changes, so the pointer is still
// reported as owned, exactly as when the feature is disabled.
//
// RUN: %clangxx_lsan -O0 %s -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=0 %run %t 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 %run %t 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <sanitizer/allocator_interface.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

static void *g_expected;
static int g_free_hooks;

static void OnMalloc(const volatile void *, size_t) {}

static void OnFree(const volatile void *ptr) {
  if ((void *)ptr != g_expected)
    return;
  ++g_free_hooks;
  fprintf(stderr, "free hook: owned=%d size=%zu\n",
          __sanitizer_get_ownership((void *)ptr),
          __sanitizer_get_allocated_size((void *)ptr));
}

int main() {
  assert(__sanitizer_install_malloc_and_free_hooks(OnMalloc, OnFree));

  g_expected = malloc(64);
  free(g_expected);
  assert(g_free_hooks == 1);

  fprintf(stderr, "done\n");
  return 0;
}

// CHECK: free hook: owned=1 size=64
// CHECK: done
