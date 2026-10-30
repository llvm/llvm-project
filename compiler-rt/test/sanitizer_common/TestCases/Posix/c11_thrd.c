// RUN: %clang -pthread %s -o %t %if freebsd %{ -lstdthreads %}
// RUN: %run %t 2>&1 | FileCheck %s

// <threads.h> is missing on Darwin and before glibc 2.28.
// UNSUPPORTED: darwin, glibc && !glibc-2.28
// https://github.com/llvm/llvm-project/issues/199585
// UNSUPPORTED: glibc && (asan || hwasan || lsan || msan || tsan)
// UNSUPPORTED: freebsd && msan

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <time.h>
#include <unistd.h>

enum { kNumThreads = 4 };

typedef struct {
  int id;
  thrd_t parent; // Set by main before thrd_create.
  thrd_t self;   // Set by the thread.
  int sum;       // Set by the thread.
} Work;

static atomic_int detached_done;

// Main hands the block over to the holder thread, which clears this pointer.
// The block is not the thrd_create argument because a sanitizer may keep that
// reachable on its own while the thread runs.
static void *handoff;
static atomic_int holder_ready;

static void check(int res, const char *call) {
  if (res != thrd_success) {
    fprintf(stderr, "%s failed: %d\n", call, res);
    abort();
  }
}

static void fail(const char *msg, int id) {
  fprintf(stderr, "thread %d: %s\n", id, msg);
  abort();
}

// Runs on each new thread: records its identity, sleeps briefly, then
// allocates and frees.
static void do_work(Work *w) {
  w->self = thrd_current();
  if (!thrd_equal(w->self, thrd_current()))
    fail("thrd_current is not stable", w->id);
  if (thrd_equal(w->self, w->parent))
    fail("thrd_current equals the parent thread", w->id);

  // thrd_sleep returns 0 on success, not thrd_success.
  struct timespec ts = {.tv_sec = 0, .tv_nsec = 1000};
  if (thrd_sleep(&ts, NULL) != 0)
    fail("thrd_sleep failed", w->id);

  int n = 100 + w->id;
  int *v = malloc(n * sizeof(*v));
  if (!v)
    fail("malloc failed", w->id);
  for (int i = 0; i < n; ++i)
    v[i] = i;
  w->sum = 0;
  for (int i = 0; i < n; ++i)
    w->sum += v[i];
  free(v);
}

static int returning_thread(void *arg) {
  Work *w = arg;
  do_work(w);
  return 1000 + w->id;
}

static int exiting_thread(void *arg) {
  Work *w = arg;
  do_work(w);
  thrd_exit(2000 + w->id);
}

static int detached_thread(void *arg) {
  Work *w = arg;
  do_work(w);
  atomic_store_explicit(&detached_done, 1, memory_order_release);
  return 0;
}

static void check_work(const Work *w) {
  int n = 100 + w->id;
  if (w->sum != n * (n - 1) / 2)
    fail("wrong sum", w->id);
}

// Keeps the only pointer to the block on its stack until the process exits.
static int holder_thread(void *arg) {
  (void)arg;
  void *volatile block = handoff;
  handoff = NULL;
  atomic_store_explicit(&holder_ready, 1, memory_order_release);
  // block is never null: this sleeps until the process exits.
  while (block)
    pause();
  return 0;
}

// Main allocates the block, not the holder thread: the leak check ignores a
// chunk whose allocation stack has no caller frame, and that is all an
// allocation gets on a thread that the sanitizer does not know about.
// noinline keeps the pointer out of main's frame.
__attribute__((noinline)) static void allocate_block(void) {
  handoff = malloc(1337);
  if (!handoff) {
    fprintf(stderr, "malloc failed\n");
    abort();
  }
}

int main(void) {
  // Start the holder first, so that the rest of main overwrites any stale copy
  // of the block's address on main's stack. The holder is neither joined nor
  // detached: it is still running at the leak check.
  allocate_block();
  thrd_t holder;
  check(thrd_create(&holder, holder_thread, NULL), "thrd_create");
  while (!atomic_load_explicit(&holder_ready, memory_order_acquire))
    thrd_yield();

  // Threads [0, kNumThreads) return their result, threads
  // [kNumThreads, 2 * kNumThreads) deliver it through thrd_exit, and the last
  // Work is for a detached thread.
  Work work[2 * kNumThreads + 1];
  thrd_t threads[2 * kNumThreads];
  thrd_t main_thread = thrd_current();
  for (int i = 0; i < 2 * kNumThreads + 1; ++i) {
    work[i].id = i;
    work[i].parent = main_thread;
  }

  thrd_t detached;
  check(thrd_create(&detached, detached_thread, &work[2 * kNumThreads]),
        "thrd_create");
  check(thrd_detach(detached), "thrd_detach");

  for (int i = 0; i < 2 * kNumThreads; ++i)
    check(thrd_create(&threads[i],
                      i < kNumThreads ? returning_thread : exiting_thread,
                      &work[i]),
          "thrd_create");

  // Wait for the detached thread without mtx_* or cnd_*, so that this test
  // depends only on the thrd_* interceptors. Waiting before the joins gives it
  // time to finish exiting; if it is still exiting at the leak check, that is
  // harmless.
  while (!atomic_load_explicit(&detached_done, memory_order_acquire))
    thrd_yield();
  check_work(&work[2 * kNumThreads]);

  for (int i = 0; i < 2 * kNumThreads; ++i) {
    // thrd_join accepts a null result pointer.
    if (i == kNumThreads - 1) {
      check(thrd_join(threads[i], NULL), "thrd_join");
    } else {
      int res;
      check(thrd_join(threads[i], &res), "thrd_join");
      if (res != (i < kNumThreads ? 1000 : 2000) + i)
        fail("wrong thrd_join result", i);
    }
    if (!thrd_equal(work[i].self, threads[i]))
      fail("thrd_current differs from the thrd_t set by thrd_create", i);
    check_work(&work[i]);
  }

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
