// DFSAN_OPTIONS=no_huge_pages_for_shadow=false RUN: %clang_dfsan %s -o %t && %run %t
// DFSAN_OPTIONS=no_huge_pages_for_shadow=true RUN: %clang_dfsan %s -o %t && %run %t
// DFSAN_OPTIONS=no_huge_pages_for_shadow=false RUN: %clang_dfsan %s -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 -o %t && %run %t
// DFSAN_OPTIONS=no_huge_pages_for_shadow=true RUN: %clang_dfsan %s -DORIGIN_TRACKING -mllvm -dfsan-track-origins=1 -o %t && %run %t

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

size_t get_rss_kb() {
  size_t ret = 0;
  pid_t pid = getpid();

  char fname[256];
  sprintf(fname, "/proc/%ld/task/%ld/smaps", (long)pid, (long)pid);
  FILE *f = fopen(fname, "r");
  assert(f);

  char buf[256];
  while (fgets(buf, sizeof(buf), f) != NULL) {
    int64_t rss;
    if (sscanf(buf, "Rss: %ld kB", &rss) == 1)
      ret += rss;
  }
  assert(feof(f));
  fclose(f);

  return ret;
}

int main(int argc, char **argv) {
  const size_t map_size = 100 << 20;
  size_t before = get_rss_kb();

  // mmap and touch all addresses. The overhead is 1x.
  char *p = mmap(NULL, map_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  memset(p, 0xff, map_size);
  size_t after_mmap = get_rss_kb();

  // store labels to all addresses. The overhead is 2x.
  const dfsan_label label = 8;
  char val = 0xff;
  dfsan_set_label(label, &val, sizeof(val));
  memset(p, val, map_size);
  size_t after_mmap_and_set_label = get_rss_kb();

  // fixed-mmap the same address. OS recyles pages and reinitializes data at the
  // address. This should be the same to calling munmap.
  p = mmap(p, map_size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  size_t after_fixed_mmap = get_rss_kb();

  // store labels to all addresses.
  memset(p, val, map_size);
  size_t after_mmap_and_set_label2 = get_rss_kb();

  // munmap the addresses.
  munmap(p, map_size);
  size_t after_munmap = get_rss_kb();

  fprintf(
      stderr,
      "RSS at start: %zu, after mmap: %zu, after mmap+set label: %zu, after "
      "fixed map: %zu, after another mmap+set label: %zu, after munmap: %zu\n",
      before, after_mmap, after_mmap_and_set_label, after_fixed_mmap,
      after_mmap_and_set_label2, after_munmap);

  const size_t mmap_cost_kb = map_size >> 10;
  // Shadow space (1:1 with application memory)
  const size_t mmap_shadow_cost_kb = sizeof(dfsan_label) * mmap_cost_kb;
#ifdef ORIGIN_TRACKING
  // Origin space (1:1 with application memory)
  const size_t mmap_origin_cost_kb = mmap_cost_kb;
#else
  const size_t mmap_origin_cost_kb = 0;
#endif
  assert(after_mmap >= before + mmap_cost_kb);
  assert(after_mmap_and_set_label >=
         after_mmap + mmap_shadow_cost_kb + mmap_origin_cost_kb);
  assert(after_mmap_and_set_label2 >=
         before + mmap_cost_kb + mmap_shadow_cost_kb + mmap_origin_cost_kb);

#ifdef ORIGIN_TRACKING
  // This value is chosen based on observed difference.
  const size_t mmap_origin_chain_kb = 4000;
#else
  const size_t mmap_origin_chain_kb = 0;
#endif

  // RSS may not change memory amount after munmap to the same level as the
  // start of the program. The assert checks the memory up to a delta.
  const size_t delta = 5000;
  // Origin chains are not freed, even when the origin space which refers to
  // them is freed, so mmap_origin_chain_kb is added to account for this.
  assert(after_fixed_mmap <= before + delta + mmap_origin_chain_kb);
  assert(after_munmap <= before + delta + mmap_origin_chain_kb);

  return 0;
}
