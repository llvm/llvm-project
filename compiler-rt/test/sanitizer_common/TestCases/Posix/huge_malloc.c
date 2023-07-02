// RUN: %clang -O0 %s -o %t && %env_tool_opts=allocator_may_return_null=1:hard_rss_limit_mb=50:quarantine_size_mb=1 %run %t

#include <stdlib.h>
#include <stdio.h>

// https://github.com/google/sanitizers/issues/981
// UNSUPPORTED: android-26

// FIXME: Hangs.
// UNSUPPORTED: tsan

// Hwasan requires tagging of new allocations, so needs RSS for shadow.
// UNSUPPORTED: hwasan

// FIXME: Something wrong with MADV_FREE or MAP_NORESERVE there.
// UNSUPPORTED: target={{.*solaris.*}}

void *p;

int main(int argc, char **argv) {
  for (int i = 0; i < sizeof(void *) * 8; ++i) {
    // Calloc avoids MSAN shadow poisoning.
    p = calloc(1ull << i, 1);
    fprintf(stderr, "%d %llu: %p\n", i, (1ull << i), p);
    free(p);
  }
  return 0;
}
