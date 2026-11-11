// RUN: %clang -pthread %s -o %t %if freebsd %{ -lstdthreads %}
// RUN: %run %t 2>&1 | FileCheck %s

// The threads come from pthread_create, not thrd_create, so that a failure
// here is due to the tss_* functions alone; thrd_* has its own test.

// <threads.h> is missing on Darwin and before glibc 2.28.
// UNSUPPORTED: darwin, glibc && !glibc-2.28
// https://github.com/llvm/llvm-project/issues/199585
// UNSUPPORTED: glibc && msan
// TSan false positive in FreeBSD libthr's internal allocator, used by
// pthread_setspecific.
// UNSUPPORTED: freebsd && tsan

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>

enum { kNumThreads = 4 };

// The keys are in heap memory: a global tss_t is zero-initialized, so MSan
// would see it as initialized whatever tss_create does.
typedef struct {
  tss_t with_dtor;
  tss_t without_dtor;
} Keys;

static Keys *keys;
static int sentinel; // The value stored under keys->without_dtor.
static atomic_int dtor_calls;
static atomic_uint dtor_ids; // Bit i is set when thread i's value is destroyed.

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

static void dtor(void *value) {
  atomic_fetch_or_explicit(&dtor_ids, 1u << *(int *)value,
                           memory_order_relaxed);
  free(value);
  atomic_fetch_add_explicit(&dtor_calls, 1, memory_order_relaxed);
}

// Sets and reads back a value under each key on the calling thread.
static void use_keys(int id) {
  if (tss_get(keys->with_dtor) || tss_get(keys->without_dtor))
    fail("tss_get on a new key is not null", id);

  int *value = malloc(sizeof(*value));
  if (!value)
    fail("malloc failed", id);
  *value = id;
  check(tss_set(keys->with_dtor, value), "tss_set");
  check(tss_set(keys->without_dtor, &sentinel), "tss_set");

  int *got = tss_get(keys->with_dtor);
  if (got != value || *got != id)
    fail("tss_get returned the wrong value", id);
  if (tss_get(keys->without_dtor) != &sentinel)
    fail("tss_get returned the wrong value", id);
}

static void *thread_func(void *arg) {
  use_keys((int)(long)arg);
  // Returning runs the destructor for keys->with_dtor.
  return NULL;
}

int main(void) {
  keys = malloc(sizeof(*keys));
  if (!keys) {
    fprintf(stderr, "malloc failed\n");
    abort();
  }
  check(tss_create(&keys->with_dtor, dtor), "tss_create");
  check(tss_create(&keys->without_dtor, NULL), "tss_create");
  // Two live keys are distinct. Comparing them in instrumented code also means
  // MSan sees the keys used whether or not it checks call arguments eagerly.
  if (keys->with_dtor == keys->without_dtor) {
    fprintf(stderr, "tss_create returned the same key twice\n");
    abort();
  }

  pthread_t threads[kNumThreads];
  for (int i = 0; i < kNumThreads; ++i) {
    if (pthread_create(&threads[i], NULL, thread_func, (void *)(long)(i + 1))) {
      fprintf(stderr, "pthread_create failed\n");
      abort();
    }
  }
  for (int i = 0; i < kNumThreads; ++i) {
    if (pthread_join(threads[i], NULL)) {
      fprintf(stderr, "pthread_join failed\n");
      abort();
    }
  }
  int calls = atomic_load_explicit(&dtor_calls, memory_order_relaxed);
  unsigned ids = atomic_load_explicit(&dtor_ids, memory_order_relaxed);
  unsigned expected_ids = (1u << (kNumThreads + 1)) - 2; // Threads 1 to N.
  if (calls != kNumThreads || ids != expected_ids) {
    fprintf(stderr, "destructor: %d calls, ids 0x%x; expected %d, 0x%x\n",
            calls, ids, kNumThreads, expected_ids);
    abort();
  }

  // Destructors run neither on tss_delete nor at program exit, so the main
  // thread frees its own value. Setting it to null exercises tss_set with null.
  use_keys(0);
  free(tss_get(keys->with_dtor));
  check(tss_set(keys->with_dtor, NULL), "tss_set");
  tss_delete(keys->with_dtor);
  tss_delete(keys->without_dtor);

  // A key created after tss_delete, which may reuse a deleted slot, starts out
  // null.
  tss_t fresh;
  check(tss_create(&fresh, NULL), "tss_create");
  if (tss_get(fresh)) {
    fprintf(stderr, "tss_get on a key created after tss_delete is not null\n");
    abort();
  }
  tss_delete(fresh);
  free(keys);

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE
