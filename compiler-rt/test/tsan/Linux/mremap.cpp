// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../test.h"

extern "C" {
void __tsan_read1(void *addr);
void __tsan_write4(void *addr);
}

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
  // middle, and end) before it is unmapped and reused as mremap destination.
  p->dst[0] = 0x33;
  p->dst[p->dst_size / 2] = 0x44;
  p->dst[p->dst_size - 1] = 0x55;
  barrier_wait(&barrier);
  return nullptr;
}

static void test_mremap_move(size_t page_size, size_t src_size,
                             size_t dst_size) {
  barrier_init(&barrier, 2);
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

  // Unmap dst, then move and expand src into dst via mremap.
  assert(munmap(dst, dst_size) == 0);
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
  // Dirty a distinct shadow cell in the unmapped second page before in-place
  // expansion.
  __tsan_write4(&p->buf[p->page_size + 64]);
  barrier_wait(&barrier);
  return nullptr;
}

static void test_mremap_shrink_and_expand(size_t page_size) {
  barrier_init(&barrier, 2);
  // Reserve 2 pages so we can shrink to 1 page and then expand in-place back to
  // 2 pages.
  char *buf = (char *)mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(buf != MAP_FAILED);

  ShrinkExpandParams params = {buf, page_size};
  pthread_t th;
  pthread_create(&th, nullptr, shrink_expand_worker, &params);
  barrier_wait(&barrier);

  // Shrink in-place from 2 pages to 1 page.
  char *shrunk = (char *)mremap(buf, 2 * page_size, page_size, 0);
  assert(shrunk == buf);

  // Shadow of the unmapped second page must be cleared.
  __tsan_read1(&buf[page_size]);
  __tsan_read1(&buf[2 * page_size - 1]);

  // Let worker dirty the shadow of the unmapped second page, then expand
  // in-place back to 2 pages.
  barrier_wait(&barrier);
  char *expanded = (char *)mremap(buf, page_size, 2 * page_size, 0);
  assert(expanded == buf);

  // Newly mapped tail page must have its shadow reset.
  buf[page_size + 64] = 0x44;
  buf[2 * page_size - 1] = 0x55;

  pthread_join(th, nullptr);
  assert(munmap(buf, 2 * page_size) == 0);
}

int main() {
  const size_t page_size = sysconf(_SC_PAGESIZE);
  // Test both below and above clear_shadow_mmap_threshold (64KB).
  test_mremap_move(page_size, page_size - 1, 2 * page_size);
  test_mremap_move(page_size, 64 * 1024, 256 * 1024);
  test_mremap_shrink_and_expand(page_size);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer: data race
// CHECK: DONE
