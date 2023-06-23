// RUN: %clang -O0 %s -o %t && %env_tool_opts=allocator_may_return_null=1:hard_rss_limit_mb=50:quarantine_size_mb=1 %run %t

#include <stdlib.h>
#include <stdio.h>

// https://github.com/google/sanitizers/issues/981
// UNSUPPORTED: android-26

// FIXME: Hangs.
// UNSUPPORTED: tsan

// FIXME: Make it work. Don't xfail to avoid excessive memory usage.
// UNSUPPORTED: msan

// Hwasan requires tagging of new allocations, so needs RSS for shadow.
// UNSUPPORTED: hwasan

void *p;

int main(int argc, char **argv) {
  for (int i = 0; i < sizeof(void *) * 8; ++i) {
    p = malloc(1ull << i);
    fprintf(stderr, "%llu: %p\n", (1ull << i), p);
    free(p);
  }
  return 0;
}
