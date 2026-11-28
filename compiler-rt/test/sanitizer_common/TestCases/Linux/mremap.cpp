// Tests that sanitizers properly update shadow memory on Linux mremap()
// (relocation, shrink, expand, MREMAP_DONTUNMAP, and unaligned sizes) without
// false positives or missed races on retained memory.
//
// RUN: %clangxx -O1 -pthread %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %if tsan %{ not %run %t race-shrink 2>&1 | FileCheck %s --check-prefix=CHECK-RACE %}
// RUN: %if tsan %{ not %run %t race-subpage 2>&1 | FileCheck %s --check-prefix=CHECK-RACE %}

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MREMAP_DONTUNMAP
#  define MREMAP_DONTUNMAP 4
#endif

#if !defined(__has_feature)
#  define __has_feature(x) 0
#endif

#if __has_feature(thread_sanitizer)
typedef unsigned invisible_barrier_t;
extern "C" {
void __tsan_testonly_barrier_init(invisible_barrier_t *barrier, unsigned count);
void __tsan_testonly_barrier_wait(invisible_barrier_t *barrier);
void __tsan_read1(void *addr);
void __tsan_write4(void *addr);
}
static invisible_barrier_t barrier;
static void barrier_init(invisible_barrier_t *b, unsigned count) {
  __tsan_testonly_barrier_init(b, count);
}
static void barrier_wait(invisible_barrier_t *b) {
  __tsan_testonly_barrier_wait(b);
}
#else
static pthread_barrier_t barrier;
static void barrier_init(pthread_barrier_t *b, unsigned count) {
  assert(pthread_barrier_init(b, nullptr, count) == 0);
}
static void barrier_wait(pthread_barrier_t *b) {
  int res = pthread_barrier_wait(b);
  assert(res == 0 || res == PTHREAD_BARRIER_SERIAL_THREAD);
}
static void __tsan_read1(void *addr) { (void)addr; }
static void __tsan_write4(void *addr) { (void)addr; }
#endif

struct MoveParams {
  char *src;
  size_t src_size;
  char *dst;
  size_t dst_size;
};

static void *move_worker(void *arg) {
  MoveParams *p = (MoveParams *)arg;
  p->src[0] = 0x11;
  p->src[p->src_size - 1] = 0x22;
  // Record unsynchronized accesses in shadow across the target range (start,
  // middle, and end) before it is replaced as mremap destination.
  p->dst[0] = 0x33;
  p->dst[p->dst_size / 2] = 0x44;
  p->dst[p->dst_size - 1] = 0x55;
  barrier_wait(&barrier);
  return nullptr;
}

static void test_mremap_move(size_t page_size, size_t src_size,
                             size_t dst_size) {
  char *src = (char *)mmap(nullptr, src_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(src != MAP_FAILED);
  char *dst = (char *)mmap(nullptr, dst_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(dst != MAP_FAILED);

  MoveParams params = {src, src_size, dst, dst_size};
  pthread_t th;
  pthread_create(&th, nullptr, move_worker, &params);
  barrier_wait(&barrier);

  // Move and expand src into dst via mremap (MREMAP_FIXED unmaps previous dst).
  char *remapped = (char *)mremap(src, src_size, dst_size,
                                  MREMAP_MAYMOVE | MREMAP_FIXED, dst);
  assert(remapped == dst);

  // Old mapping's shadow must be cleared.
  __tsan_read1(&src[0]);
  __tsan_read1(&src[src_size - 1]);

  // New mapping's shadow must be reset across the entire range.
  memset(remapped, 0x66, dst_size);

  pthread_join(th, nullptr);
  assert(munmap(remapped, dst_size) == 0);
}

struct DontunmapParams {
  char *src;
  size_t size;
};

static void *dontunmap_worker(void *arg) {
  DontunmapParams *p = (DontunmapParams *)arg;
  p->src[0] = 0x11;
  p->src[p->size - 1] = 0x22;
  barrier_wait(&barrier);
  return nullptr;
}

static void test_mremap_dontunmap(size_t page_size) {
  const size_t size = 2 * page_size;
  char *src = (char *)mmap(nullptr, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(src != MAP_FAILED);

  DontunmapParams params = {src, size};
  pthread_t th;
  pthread_create(&th, nullptr, dontunmap_worker, &params);
  barrier_wait(&barrier);

  char *dst =
      (char *)mremap(src, size, size, MREMAP_MAYMOVE | MREMAP_DONTUNMAP);
  if (dst == MAP_FAILED) {
    assert(errno == EINVAL);
    pthread_join(th, nullptr);
    assert(munmap(src, size) == 0);
    return;
  }
  assert(dst != src);

  // Both the relocated mapping and the zero-filled old mapping must have
  // their shadow reset.
  memset(dst, 0x33, size);
  memset(src, 0x44, size);

  pthread_join(th, nullptr);
  assert(munmap(src, size) == 0);
  assert(munmap(dst, size) == 0);
}

struct ShrinkExpandParams {
  char *buf;
  size_t page_size;
};

static void *shrink_expand_worker(void *arg) {
  ShrinkExpandParams *p = (ShrinkExpandParams *)arg;
  p->buf[0] = 0x11;
  p->buf[p->page_size] = 0x22;
  p->buf[2 * p->page_size - 1] = 0x33;
  barrier_wait(&barrier);
  // Wait for main to shrink the mapping before dirtying the unmapped second
  // page's shadow, so UnmapShadow does not clear this write prematurely.
  barrier_wait(&barrier);
  __tsan_write4(&p->buf[p->page_size + 64]);
  __tsan_write4(&p->buf[2 * p->page_size - 16]);
  barrier_wait(&barrier);
  return nullptr;
}

static void test_mremap_shrink_and_expand(size_t page_size) {
  // Reserve 2 pages so we can shrink to an unaligned size (100 bytes) and then
  // expand in-place to 200 bytes and page_size + 1 (2 pages).
  char *buf = (char *)mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf != MAP_FAILED);

  ShrinkExpandParams params = {buf, page_size};
  pthread_t th;
  pthread_create(&th, nullptr, shrink_expand_worker, &params);
  barrier_wait(&barrier);

  // Shrink in-place from 2 pages to 100 bytes (still 1 page mapped).
  char *shrunk = (char *)mremap(buf, 2 * page_size, 100, 0);
  assert(shrunk == buf);

  // Shadow of the unmapped second page must be cleared.
  __tsan_read1(&buf[page_size]);
  __tsan_read1(&buf[2 * page_size - 1]);

  // Allow worker to dirty the shadow of the unmapped second page, then wait for
  // it before expanding in-place.
  barrier_wait(&barrier);
  barrier_wait(&barrier);

  // Sub-page expansion (100 -> 200) keeps the same single mapped page.
  char *subpage = (char *)mremap(buf, 100, 200, 0);
  assert(subpage == buf);

  // Expanding to page_size + 1 maps the entire second page, so its full
  // page-aligned shadow must be reset (including bytes past page_size + 1).
  char *expanded = (char *)mremap(buf, 200, page_size + 1, 0);
  assert(expanded == buf);

  buf[page_size + 64] = 0x44;
  buf[2 * page_size - 16] = 0x55;
  buf[2 * page_size - 1] = 0x66;

  pthread_join(th, nullptr);
  assert(munmap(buf, 2 * page_size) == 0);
}

static void *retained_write_worker(void *arg) {
  char *addr = (char *)arg;
  *addr = 0x11;
  barrier_wait(&barrier);
  return nullptr;
}

static void test_mremap_race_shrink_expand(size_t page_size) {
  char *buf = (char *)mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf != MAP_FAILED);

  pthread_t th;
  pthread_create(&th, nullptr, retained_write_worker, &buf[0]);
  barrier_wait(&barrier);

  assert(mremap(buf, 2 * page_size, page_size, 0) == buf);
  assert(mremap(buf, page_size, 2 * page_size, 0) == buf);

  // Retained first page must keep shadow history so this race is reported.
  buf[0] = 0x22;

  pthread_join(th, nullptr);
  assert(munmap(buf, 2 * page_size) == 0);
}

static void test_mremap_race_subpage_expand(size_t page_size) {
  char *buf = (char *)mmap(nullptr, page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf != MAP_FAILED);

  pthread_t th;
  pthread_create(&th, nullptr, retained_write_worker, &buf[150]);
  barrier_wait(&barrier);

  // Sub-page expansion (100 -> 200) must not reset live shadow on the page.
  assert(mremap(buf, 100, 200, 0) == buf);

  buf[150] = 0x22;

  pthread_join(th, nullptr);
  assert(munmap(buf, page_size) == 0);
}

int main(int argc, char **argv) {
  barrier_init(&barrier, 2);
  const size_t page_size = sysconf(_SC_PAGESIZE);

  if (argc > 1 && strcmp(argv[1], "race-shrink") == 0) {
    test_mremap_race_shrink_expand(page_size);
    return 0;
  }
  if (argc > 1 && strcmp(argv[1], "race-subpage") == 0) {
    test_mremap_race_subpage_expand(page_size);
    return 0;
  }

  // Test both below and above clear_shadow_mmap_threshold (64KB), plus
  // unaligned sizes.
  test_mremap_move(page_size, page_size - 1, 2 * page_size);
  test_mremap_move(page_size, page_size + 1, 2 * page_size + 1);
  test_mremap_move(page_size, 64 * 1024, 256 * 1024);
  test_mremap_dontunmap(page_size);
  test_mremap_shrink_and_expand(page_size);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
// CHECK-RACE: WARNING: ThreadSanitizer: data race
