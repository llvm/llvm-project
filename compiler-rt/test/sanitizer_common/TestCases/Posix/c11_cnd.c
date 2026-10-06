// RUN: %clang -pthread %s -o %t %if freebsd %{ -lstdthreads %}
// RUN: %run %t 2>&1 | FileCheck %s

// The threads come from pthread_create, not thrd_create, so that a failure
// here is due to the cnd_* (and mtx_*) functions alone; thrd_* has its own
// test.

// UNSUPPORTED: darwin, android, glibc && !glibc-2.28
// UNSUPPORTED: glibc && tsan

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <time.h>

static void check(int res, const char *what) {
  if (res != thrd_success) {
    fprintf(stderr, "%s failed: %d\n", what, res);
    abort();
  }
}

static void fail(const char *what) {
  fprintf(stderr, "%s\n", what);
  abort();
}

static pthread_t start_thread(void *(*func)(void *)) {
  pthread_t t;
  if (pthread_create(&t, NULL, func, NULL) != 0)
    fail("pthread_create failed");
  return t;
}

static void join_thread(pthread_t t) {
  if (pthread_join(t, NULL) != 0)
    fail("pthread_join failed");
}

// Protects all the state below.
static mtx_t mtx;

// cnd_signal and cnd_wait: a producer hands items to the main thread through
// a one-slot buffer.
enum { kItems = 100 };
static cnd_t not_empty, not_full;
static int slot_full;
static int slot;

static void *producer(void *arg) {
  (void)arg;
  for (int i = 0; i < kItems; ++i) {
    check(mtx_lock(&mtx), "mtx_lock");
    while (slot_full)
      check(cnd_wait(&not_full, &mtx), "cnd_wait");
    slot = i;
    slot_full = 1;
    check(cnd_signal(&not_empty), "cnd_signal");
    check(mtx_unlock(&mtx), "mtx_unlock");
  }
  return NULL;
}

static void test_signal_wait(void) {
  pthread_t t = start_thread(producer);
  for (int i = 0; i < kItems; ++i) {
    check(mtx_lock(&mtx), "mtx_lock");
    while (!slot_full)
      check(cnd_wait(&not_empty, &mtx), "cnd_wait");
    if (slot != i)
      fail("wrong item");
    slot_full = 0;
    check(cnd_signal(&not_full), "cnd_signal");
    check(mtx_unlock(&mtx), "mtx_unlock");
  }
  join_thread(t);
}

// cnd_broadcast: one broadcast wakes every thread waiting on go_cnd. A waiter
// that the broadcast misses never returns, so the test then hangs in
// join_thread rather than failing.
enum { kWaiters = 4 };
static cnd_t ready_cnd, go_cnd;
static int waiting, go;

static void *waiter(void *arg) {
  (void)arg;
  check(mtx_lock(&mtx), "mtx_lock");
  ++waiting;
  check(cnd_signal(&ready_cnd), "cnd_signal");
  while (!go)
    check(cnd_wait(&go_cnd, &mtx), "cnd_wait");
  check(mtx_unlock(&mtx), "mtx_unlock");
  return NULL;
}

static void test_broadcast(void) {
  pthread_t t[kWaiters];
  for (int i = 0; i < kWaiters; ++i)
    t[i] = start_thread(waiter);
  check(mtx_lock(&mtx), "mtx_lock");
  // A waiter holds mtx from its increment until cnd_wait releases it, so once
  // all have counted themselves, all are blocked in cnd_wait.
  while (waiting < kWaiters)
    check(cnd_wait(&ready_cnd, &mtx), "cnd_wait");
  go = 1;
  check(cnd_broadcast(&go_cnd), "cnd_broadcast");
  check(mtx_unlock(&mtx), "mtx_unlock");
  for (int i = 0; i < kWaiters; ++i)
    join_thread(t[i]);
}

// cnd_timedwait: an absolute TIME_UTC deadline, first one that expires, then
// one met by a signal from another thread.
static cnd_t timed_cnd;
static int timed_ready;

// C11 timeouts are TIME_UTC, which is CLOCK_REALTIME. MSan intercepts
// clock_gettime, so the deadline it returns is not reported as uninitialized.
static struct timespec deadline_after_ms(long ms) {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
    fail("clock_gettime failed");
  ts.tv_sec += ms / 1000;
  ts.tv_nsec += (ms % 1000) * 1000000L;
  if (ts.tv_nsec >= 1000000000L) {
    ++ts.tv_sec;
    ts.tv_nsec -= 1000000000L;
  }
  return ts;
}

static void *timed_signaler(void *arg) {
  (void)arg;
  check(mtx_lock(&mtx), "mtx_lock");
  timed_ready = 1;
  check(cnd_signal(&timed_cnd), "cnd_signal");
  check(mtx_unlock(&mtx), "mtx_unlock");
  return NULL;
}

static void test_timedwait(void) {
  check(mtx_lock(&mtx), "mtx_lock");

  // Nothing signals timed_cnd yet, so the wait must time out; an earlier
  // wakeup is spurious.
  struct timespec deadline = deadline_after_ms(10);
  int res;
  do
    res = cnd_timedwait(&timed_cnd, &mtx, &deadline);
  while (res == thrd_success);
  if (res != thrd_timedout)
    fail("cnd_timedwait did not time out");

  // The signaler starts while this thread holds mtx, so it can only set
  // timed_ready while this thread waits in cnd_timedwait. Loop on the
  // predicate: a timeout is not an error if the predicate has become true,
  // and if it has not (only on a very slow machine), wait with a new deadline.
  pthread_t t = start_thread(timed_signaler);
  deadline = deadline_after_ms(1000);
  while (!timed_ready) {
    res = cnd_timedwait(&timed_cnd, &mtx, &deadline);
    if (res == thrd_timedout)
      deadline = deadline_after_ms(1000);
    else
      check(res, "cnd_timedwait");
  }
  check(mtx_unlock(&mtx), "mtx_unlock");
  join_thread(t);
}

int main(void) {
  check(mtx_init(&mtx, mtx_plain), "mtx_init");
  check(cnd_init(&not_empty), "cnd_init");
  check(cnd_init(&not_full), "cnd_init");
  check(cnd_init(&ready_cnd), "cnd_init");
  check(cnd_init(&go_cnd), "cnd_init");
  check(cnd_init(&timed_cnd), "cnd_init");

  test_signal_wait();
  test_broadcast();
  test_timedwait();

  cnd_destroy(&timed_cnd);
  cnd_destroy(&go_cnd);
  cnd_destroy(&ready_cnd);
  cnd_destroy(&not_full);
  cnd_destroy(&not_empty);
  mtx_destroy(&mtx);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
