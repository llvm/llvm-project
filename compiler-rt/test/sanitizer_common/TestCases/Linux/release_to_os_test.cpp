// RUN: %clangxx %s -o %t
// RUN: env %tool_options=allocator_release_to_os_interval_ms=-1 %run %t

// Temporarily disable test
// UNSUPPORTED: tsan
// UNSUPPORTED: target=powerpc64{{.*}}

// Not needed, no allocator.
// UNSUPPORTED: ubsan

// FIXME: This mode uses 32bit allocator without purge.
// UNSUPPORTED: hwasan-aliasing

#include <algorithm>
#include <assert.h>
#include <fcntl.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sanitizer/allocator_interface.h>

const size_t kPageSize = 4096;

void sync_rss() {
  char *page =
      (char *)mmap((void *)&sync_rss, kPageSize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
  // Linux kernel updates RSS counters after a set number of page faults.
  for (int i = 0; i < 10000; ++i) {
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

size_t MallocReleaseStress() {
  const size_t kNumChunks = 10000;
  const size_t kAllocSize = 100;
  const size_t kNumIter = 100;
  uintptr_t *chunks[kNumChunks] = {0};
  std::mt19937 r;

  for (size_t iter = 0; iter < kNumIter; iter++) {
    std::shuffle(chunks, chunks + kNumChunks, r);
    size_t to_replace = rand() % kNumChunks;
    for (size_t i = 0; i < kNumChunks; i++) {
      if (chunks[i])
        assert(chunks[i][0] == (uintptr_t)chunks[i]);
      if (i < to_replace) {
        delete[] chunks[i];
        chunks[i] = new uintptr_t[kAllocSize];
        chunks[i][0] = (uintptr_t)chunks[i];
      }
    }
  }
  fprintf(stderr, "before delete: %zu\n", current_rss());
  for (auto p : chunks)
    delete[] p;
  return kNumChunks * kAllocSize * sizeof(uintptr_t);
}

int main(int argc, char **argv) {
  // 32bit asan allocator is unsupported.
  if (sizeof(void *) < 8)
    return 0;
  auto a = current_rss();
  auto total = MallocReleaseStress() >> 10;
  auto b = current_rss();
  __sanitizer_purge_allocator();
  auto c = current_rss();
  fprintf(stderr, "a:%zu b:%zu c:%zu total:%zu\n", a, b, c, total);
  assert(a + total / 8 < b);
  assert(c + total / 8 < b);
}
