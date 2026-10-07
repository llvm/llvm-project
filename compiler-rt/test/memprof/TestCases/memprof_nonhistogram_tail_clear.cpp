// The non-histogram shadow keeps one counter per MEM_GRANULARITY block. An
// allocation smaller than SHADOW_GRANULARITY used to round its size down to 0,
// which skipped ClearShadow entirely, so a recycled chunk reported the previous
// allocation's access count.

// RUN: %clangxx_memprof -O0 -mllvm -memprof-use-callbacks=true %s -o %t
// RUN: %env_memprof_opts=print_text=1:log_path=stdout %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int main() {
  char *first = nullptr;
  // A single call site, so both allocations merge into one MIB.
  for (int i = 0; i < 2; ++i) {
    char *p = (char *)malloc(4);
    if (!p)
      return 1;
    if (i == 0) {
      first = p;
      for (int j = 0; j < 42; ++j)
        p[0] = 'A';
    } else if (p != first) {
      // The stale count only manifests on a recycled chunk. memprof has no
      // quarantine, so with no intervening allocation the just-freed chunk
      // comes back; require it so the test cannot pass vacuously.
      return 1;
    }
    free(p);
  }

  printf("Test completed successfully\n");
  return 0;
}

// The second allocation never touches the chunk, so it must contribute 0:
// 42 accesses over 2 allocations gives an average of 21.
// CHECK: access_count (ave/min/max): 21.00 / 0 / 42
// CHECK: Test completed successfully
