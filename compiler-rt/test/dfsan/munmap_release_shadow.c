// RUN: %clang_dfsan %s -o %t && %run %t

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

size_t get_rss_kb() {
  long rss = 0L;
  FILE *f = NULL;
  assert((f = fopen("/proc/self/statm", "r")));
  assert(fscanf(f, "%*s%ld", &rss) == 1);
  fclose(f);
  return ((size_t)rss * (size_t)sysconf(_SC_PAGESIZE)) >> 10;
}

int main(int argc, char **argv) {
  const size_t map_size = 100 << 20;
  size_t before = get_rss_kb();

  char *p = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  const dfsan_label label = dfsan_create_label("l", 0);
  char val = 0xff;
  dfsan_set_label(label, &val, sizeof(val));
  memset(p, val, map_size);
  size_t after_mmap = get_rss_kb();

  munmap(p, map_size);
  size_t after_munmap = get_rss_kb();

  fprintf(stderr, "RSS at start: %td, after mmap: %td, after mumap: %td\n",
          before, after_mmap, after_munmap);

  // The memory after mmap increases 3 times of map_size because the overhead of
  // shadow memory is 2x.
  const size_t mmap_cost_kb = 3 * (map_size >> 10);
  assert(after_mmap >= before + mmap_cost_kb);
  // OS does not release memory to the same level as the start of the program.
  // The assert checks the memory after munmap up to a delta.
  const size_t delta = 50000;
  assert(after_munmap + delta <= after_mmap);
  return 0;
}
