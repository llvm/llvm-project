// RUN: %clang -pthread %s -o %t %if freebsd %{ -lstdthreads %}
// RUN: %run %t 2>&1 | FileCheck %s

// The threads come from pthread_create, not thrd_create, and yield with
// sched_yield, not thrd_yield, so that a failure here is due to the mtx_*
// functions alone; thrd_* has its own test.

// <threads.h> is missing on Darwin and before glibc 2.28.
// UNSUPPORTED: darwin, glibc && !glibc-2.28
// https://github.com/llvm/llvm-project/issues/199585
// UNSUPPORTED: glibc && tsan

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <time.h>

enum { kThreads = 4, kIterations = 1000 };

static mtx_t plain_mtx;     // Guards plain_count.
static mtx_t timed_mtx;     // Guards timed_count.
static mtx_t recursive_mtx; // Guards recursive_count.
static long plain_count, timed_count, recursive_count;

// An absolute TIME_UTC (that is, CLOCK_REALTIME) deadline that is never
// reached.
static struct timespec future;
// An absolute TIME_UTC deadline that has passed.
static const struct timespec epoch = {0, 0};

static void expect(int res, int expected, const char *what) {
  if (res != expected) {
    fprintf(stderr, "%s returned %d, expected %d\n", what, res, expected);
    abort();
  }
}

static void check(int res, const char *what) {
  expect(res, thrd_success, what);
}

static void start(pthread_t *thread, void *(*func)(void *)) {
  if (pthread_create(thread, NULL, func, NULL)) {
    fprintf(stderr, "pthread_create failed\n");
    abort();
  }
}

static void join(pthread_t thread) {
  if (pthread_join(thread, NULL)) {
    fprintf(stderr, "pthread_join failed\n");
    abort();
  }
}

// Runs while main holds plain_mtx and timed_mtx, so it must not get either.
static void *contender(void *arg) {
  (void)arg;
  expect(mtx_trylock(&plain_mtx), thrd_busy, "mtx_trylock (held)");
  expect(mtx_timedlock(&timed_mtx, &epoch), thrd_timedout,
         "mtx_timedlock (held)");
  return NULL;
}

static void *worker(void *arg) {
  (void)arg;
  for (int i = 0; i < kIterations; ++i) {
    check(mtx_lock(&plain_mtx), "mtx_lock");
    ++plain_count;
    check(mtx_unlock(&plain_mtx), "mtx_unlock");

    int res;
    while ((res = mtx_trylock(&plain_mtx)) == thrd_busy)
      sched_yield();
    check(res, "mtx_trylock");
    ++plain_count;
    check(mtx_unlock(&plain_mtx), "mtx_unlock");

    check(mtx_timedlock(&timed_mtx, &future), "mtx_timedlock");
    ++timed_count;
    check(mtx_unlock(&timed_mtx), "mtx_unlock");

    // The second lock checks that mtx_recursive is honoured: a non-recursive
    // mutex would deadlock here.
    check(mtx_lock(&recursive_mtx), "mtx_lock");
    check(mtx_lock(&recursive_mtx), "mtx_lock (recursive)");
    ++recursive_count;
    check(mtx_unlock(&recursive_mtx), "mtx_unlock (recursive)");
    check(mtx_unlock(&recursive_mtx), "mtx_unlock");
  }
  return NULL;
}

int main(void) {
  check(mtx_init(&plain_mtx, mtx_plain), "mtx_init(mtx_plain)");
  check(mtx_init(&timed_mtx, mtx_timed), "mtx_init(mtx_timed)");
  check(mtx_init(&recursive_mtx, mtx_plain | mtx_recursive),
        "mtx_init(mtx_plain | mtx_recursive)");

  if (clock_gettime(CLOCK_REALTIME, &future) != 0) {
    fprintf(stderr, "clock_gettime failed\n");
    abort();
  }
  future.tv_sec += 3600;

  // Failed attempts to lock, made deterministic by holding the mutexes.
  check(mtx_lock(&plain_mtx), "mtx_lock");
  check(mtx_lock(&timed_mtx), "mtx_lock");
  pthread_t thread;
  start(&thread, contender);
  join(thread);
  check(mtx_unlock(&timed_mtx), "mtx_unlock");
  check(mtx_unlock(&plain_mtx), "mtx_unlock");

  // Contended locking: the counters are only correct, and race-free, if the
  // mutexes provide mutual exclusion.
  pthread_t threads[kThreads];
  for (int i = 0; i < kThreads; ++i)
    start(&threads[i], worker);
  for (int i = 0; i < kThreads; ++i)
    join(threads[i]);

  long expected = (long)kThreads * kIterations;
  if (plain_count != 2 * expected || timed_count != expected ||
      recursive_count != expected) {
    fprintf(stderr, "wrong counts: %ld %ld %ld\n", plain_count, timed_count,
            recursive_count);
    abort();
  }

  mtx_destroy(&plain_mtx);
  mtx_destroy(&timed_mtx);
  mtx_destroy(&recursive_mtx);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
