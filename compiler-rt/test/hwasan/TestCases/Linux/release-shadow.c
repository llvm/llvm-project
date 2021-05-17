// Test that tagging a large region to 0 reduces RSS.
// RUN: %clang_hwasan -mllvm -hwasan-globals=0 -mllvm -hwasan-instrument-stack=0 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sanitizer/hwasan_interface.h>

const unsigned char kTag = 42;
const size_t kNumShadowPages = 256;
const size_t kNumPages = 16 * kNumShadowPages;
const size_t kPageSize = 4096;
const size_t kMapSize = kNumPages * kPageSize;

void sync_rss() {
  char *page = (char *)mmap(0, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  // Linux kernel updates RSS counters after a set number of page faults.
  for (int i = 0; i < 1000; ++i) {
    page[0] = 42;
    madvise(page, kPageSize, MADV_DONTNEED);
  }
  munmap(page, kPageSize);
}

size_t current_rss() {
  sync_rss();
  int statm_fd = open("/proc/self/statm", O_RDONLY);
  assert(statm_fd >= 0);

  char buf[100];
  assert(read(statm_fd, &buf, sizeof(buf)) > 0);
  size_t size, rss;
  assert(sscanf(buf, "%zu %zu", &size, &rss) == 2);

  close(statm_fd);
  return rss;
}

void test_rss_difference(void *p) {
  __hwasan_tag_memory(p, kTag, kMapSize);
  size_t rss_before = current_rss();
  __hwasan_tag_memory(p, 0, kMapSize);
  size_t rss_after = current_rss();
  fprintf(stderr, "%zu -> %zu\n", rss_before, rss_after);
  assert(rss_before > rss_after);
  size_t diff = rss_before - rss_after;
  fprintf(stderr, "diff %zu\n", diff);
  // Check that the difference is at least close to kNumShadowPages.
  assert(diff > kNumShadowPages / 4 * 3);
}

int main() {
  fprintf(stderr, "starting rss %zu\n", current_rss());
  fprintf(stderr, "shadow pages: %zu\n", kNumShadowPages);

  void *p = mmap(0, kMapSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  fprintf(stderr, "p = %p\n", p);

  test_rss_difference(p);
  test_rss_difference(p);
  test_rss_difference(p);

  return 0;
}
