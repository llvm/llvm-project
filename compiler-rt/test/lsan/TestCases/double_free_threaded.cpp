// Concurrent frees of the same pointer must be reported exactly once: the
// chunk state machine lets a single thread claim the chunk, and every other
// thread observes the double free.
//
// The "large" case covers the secondary allocator, where releasing a chunk
// unmaps it together with its metadata: a losing thread must never end up
// reading, or compare-exchanging on, memory the winning thread has already
// unmapped. That window is a handful of instructions wide, so this is a smoke
// test rather than a reliable reproducer; a regression surfaces as a SEGV
// report from inside the runtime instead of the double-free report below.
//
// RUN: %clangxx_lsan -O0 %s -pthread -o %t
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t small 2>&1 | FileCheck %s
// RUN: %env_lsan_opts=detect_leaks=0:detect_double_free=1 not %run %t large 2>&1 | FileCheck %s
// REQUIRES: lsan-standalone
// UNSUPPORTED: darwin, target={{.*netbsd.*}}

#include <pthread.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const int kThreads = 16;
static void *g_chunk;
static pthread_barrier_t g_barrier;

static void *Racer(void *) {
  pthread_barrier_wait(&g_barrier);
  free(g_chunk);
  return nullptr;
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  g_chunk = malloc(!strcmp(argv[1], "large") ? (2 << 20) : 64);
  pthread_barrier_init(&g_barrier, nullptr, kThreads);

  pthread_t threads[kThreads];
  for (int i = 0; i < kThreads; ++i)
    pthread_create(&threads[i], nullptr, Racer, nullptr);
  for (int i = 0; i < kThreads; ++i)
    pthread_join(threads[i], nullptr);

  fprintf(stderr, "not reached\n");
  return 0;
}

// CHECK: ERROR: LeakSanitizer: attempting double-free on
// CHECK: The second free occurred here:
// CHECK: SUMMARY: LeakSanitizer: double-free
// CHECK-NOT: not reached
