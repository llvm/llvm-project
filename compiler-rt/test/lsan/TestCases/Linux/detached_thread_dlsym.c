// Test that TLS allocations (such as glibc's internal dlerror buffer from a
// failed dlsym call, freed in __libc_thread_freeres) on a dying detached
// thread are not falsely reported as leaks if leak checking runs after the
// sanitizer's TSD destructor has executed.
//
// Disable symbolization so that even if libc debug symbols are installed,
// LSan's default `leak:*dlerror*` suppression cannot match the non-exported
// `_dlerror_run` frame and mask the leak.
// RUN: %clang_lsan %s -ldl -pthread -o %t && %env_lsan_opts=symbolize=0 %run %t

#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>

static pthread_key_t key;
static atomic_int ready = 0;

// Disable HWASan instrumentation because on the final TSD destruction pass
// this destructor runs after HwasanTSDDtor has zeroed __hwasan_tls.
__attribute__((no_sanitize("hwaddress"))) static void
key_destructor(void *arg) {
  // ASan and LSan defer thread unregistration to the final
  // (PTHREAD_DESTRUCTOR_ITERATIONS) TSD destruction pass by re-setting their
  // pthread_key_t on earlier passes. Because the sanitizer creates its key
  // during .preinit_array (key index 0) and main() creates `key` later (key
  // index 1), also re-setting `key` for PTHREAD_DESTRUCTOR_ITERATIONS passes
  // ensures that on the final pass glibc's __nptl_deallocate_tsd() invokes the
  // sanitizer destructor (index 0) first and this destructor (index 1)
  // immediately after, before __libc_thread_freeres() runs.
  uintptr_t iter = (uintptr_t)arg;
  if (iter > 1) {
    int res = pthread_setspecific(key, (void *)(iter - 1));
    assert(res == 0);
    return;
  }
  atomic_store(&ready, 1);
  sleep(10);
}

static void *thread_func(void *arg) {
  dlsym(RTLD_DEFAULT, "nonexistent_symbol_12345");
  int res = pthread_setspecific(key, (void *)PTHREAD_DESTRUCTOR_ITERATIONS);
  assert(res == 0);
  return NULL;
}

int main(void) {
  int res = pthread_key_create(&key, key_destructor);
  assert(res == 0);

  pthread_attr_t attr;
  res = pthread_attr_init(&attr);
  assert(res == 0);
  res = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  assert(res == 0);

  pthread_t th;
  res = pthread_create(&th, &attr, thread_func, NULL);
  assert(res == 0);
  pthread_attr_destroy(&attr);

  while (!atomic_load(&ready))
    sched_yield();

  return 0;
}
