// Regression test for
// https://code.google.com/p/chromium/issues/detail?id=446692
// where asan consumed too much RAM due to transparent hugetables.
//
// RUN: %clangxx_asan -g %s -o %t
// RUN: %env_asan_opts=no_huge_pages_for_shadow=1 %run %t 2>&1 | FileCheck %s
// RUN: %run %t 2>&1 | FileCheck %s
//
// Would be great to run the test with no_huge_pages_for_shadow=0, but
// the result will depend on the OS version and settings...
//
// REQUIRES: x86_64-target-arch
//
// WARNING: this test is very subtle and may nto work on some systems.
// If this is the case we'll need to futher improve it or disable it.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sanitizer/asan_interface.h>

char FileContents[1 << 16];

void FileToString(const char *path) {
  FileContents[0] = 0;
  int fd = open(path, 0);
  if (fd < 0) return;
  char *p = FileContents;
  ssize_t size = sizeof(FileContents) - 1;
  ssize_t res = 0;
  do {
    ssize_t got = read (fd, p, size);
    if (got == 0)
      break;
    else if (got > 0)
      {
        p += got;
        res += got;
        size -= got;
      }
    else if (errno != EINTR)
      break;
  } while (size > 0 && res < sizeof(FileContents));
  if (res >= 0)
    FileContents[res] = 0;
}

long ReadShadowRss() {
  const char *path = "/proc/self/smaps";
  FileToString(path);
  char *s = strstr(FileContents, "2008fff7000-10007fff8000");
  if (!s) return 0;

  s = strstr(s, "Rss:");
  if (!s) return 0;
  s = s + 4;
  return atol(s);
}

const int kAllocSize = 1 << 28;  // 256Mb
const int kTwoMb = 1 << 21;
const int kAsanShadowGranularity = 8;

char *x;

__attribute__((no_sanitize_address)) void TouchNoAsan(size_t i) { x[i] = 0; }

int main() {
  long rss[5];
  rss[0] = ReadShadowRss();
  // use mmap directly to avoid asan touching the shadow.
  x = (char *)mmap(0, kAllocSize, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, 0, 0);
  fprintf(stderr, "X: %p-%p\n", x, x + kAllocSize);
  rss[1] = ReadShadowRss();

  // Touch the allocated region, but not the shadow.
  for (size_t i = 0; i < kAllocSize; i += kTwoMb * kAsanShadowGranularity)
    TouchNoAsan(i);
  rss[2] = ReadShadowRss();

  // Touch the shadow just a bit, in 2Mb*Granularity steps.
  for (size_t i = 0; i < kAllocSize; i += kTwoMb * kAsanShadowGranularity)
    __asan_poison_memory_region(x + i, kAsanShadowGranularity);
  rss[3] = ReadShadowRss();

  // Touch all the shadow.
  __asan_poison_memory_region(x, kAllocSize);
  rss[4] = ReadShadowRss();

  // Print the differences.
  for (int i = 0; i < 4; i++) {
    assert(rss[i] > 0);
    assert(rss[i+1] >= rss[i]);
    long diff = rss[i+1] / rss[i];
    fprintf(stderr, "RSS CHANGE IS %d => %d: %s (%ld vs %ld)\n", i, i + 1,
            diff < 10 ? "SMALL" : "LARGE", rss[i], rss[i + 1]);
  }
}
// CHECK: RSS CHANGE IS 2 => 3: SMALL
// CHECK: RSS CHANGE IS 3 => 4: LARGE
