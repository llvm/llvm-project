// Test that glibc's per-thread resolver state (_res._u._ext.nsaddrs, normally
// freed in __libc_thread_freeres -> res_thread_freeres) on a dying detached
// thread is not falsely reported as a leak if leak checking runs after the
// sanitizer's TSD destructor has executed.
//
// Disable symbolization so that LSan's default suppressions cannot match
// internal libresolv frames and mask the leak.
// REQUIRES: glibc
// RUN: %clang_lsan %s -pthread -o %t && %env_lsan_opts=symbolize=0 %run %t

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <assert.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <resolv.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static pthread_key_t key;
static pthread_key_t late_resolv_key;
static atomic_int ready = 0;
static struct sockaddr_in closed_ns_addr;

// Disable HWASan instrumentation because late_resolv_destructor calls this on
// the final TSD destruction pass after HwasanTSDDtor has zeroed __hwasan_tls.
__attribute__((no_sanitize("hwaddress"))) static void do_lookup(void) {
  struct addrinfo *ai = NULL;
  if (getaddrinfo("example.invalid.", NULL, NULL, &ai) == 0)
    freeaddrinfo(ai);
}

static void init_loopback_res(void) {
  // Initialize _res cleanly (in case NPTL recycled a cached thread stack/TCB),
  // detach any /etc/resolv.conf state, and point at our closed loopback port so
  // __res_context_send allocates _res._u._ext.nsaddrs[0] hermetically and fails
  // immediately with ECONNREFUSED without external network access.
  res_init();
  res_nclose(&_res);
  _res.nscount = 1;
  _res.nsaddr_list[0] = closed_ns_addr;
  _res.retrans = 1;
  _res.retry = 1;
  _res.options = RES_INIT | RES_DEFAULT;
}

// Disable HWASan instrumentation because on the final TSD destruction pass
// this destructor runs after HwasanTSDDtor has zeroed __hwasan_tls.
__attribute__((no_sanitize("hwaddress"))) static void
late_resolv_destructor(void *arg) {
  uintptr_t iter = (uintptr_t)arg;
  if (iter > 1) {
    int res = pthread_setspecific(late_resolv_key, (void *)(iter - 1));
    assert(res == 0);
    return;
  }
  // Verify that another TSD destructor running after ClearDlerror() can still
  // invoke the resolver without crashing before res_thread_freeres() runs.
  do_lookup();
}

// Disable HWASan instrumentation because on the final TSD destruction pass
// this destructor runs after HwasanTSDDtor has zeroed __hwasan_tls.
__attribute__((no_sanitize("hwaddress"))) static void
key_destructor(void *arg) {
  uintptr_t iter = (uintptr_t)arg;
  if (iter > 1) {
    int res = pthread_setspecific(key, (void *)(iter - 1));
    assert(res == 0);
    return;
  }
  atomic_store(&ready, 1);
  sleep(10);
}

static void *joined_thread_func(void *arg) {
  init_loopback_res();
  do_lookup();
  int res = pthread_setspecific(late_resolv_key,
                                (void *)PTHREAD_DESTRUCTOR_ITERATIONS);
  assert(res == 0);
  return NULL;
}

static void *detached_thread_func(void *arg) {
  init_loopback_res();
  do_lookup();
  int res = pthread_setspecific(key, (void *)PTHREAD_DESTRUCTOR_ITERATIONS);
  assert(res == 0);
  return NULL;
}

int main(void) {
  setenv("RES_OPTIONS", "timeout:1 attempts:1", 1);

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  assert(fd >= 0);
  memset(&closed_ns_addr, 0, sizeof(closed_ns_addr));
  closed_ns_addr.sin_family = AF_INET;
  closed_ns_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int res =
      bind(fd, (struct sockaddr *)&closed_ns_addr, sizeof(closed_ns_addr));
  assert(res == 0);
  socklen_t addrlen = sizeof(closed_ns_addr);
  res = getsockname(fd, (struct sockaddr *)&closed_ns_addr, &addrlen);
  assert(res == 0);
  close(fd);

  res = pthread_key_create(&key, key_destructor);
  assert(res == 0);
  res = pthread_key_create(&late_resolv_key, late_resolv_destructor);
  assert(res == 0);

  pthread_t joined_th;
  res = pthread_create(&joined_th, NULL, joined_thread_func, NULL);
  assert(res == 0);
  res = pthread_join(joined_th, NULL);
  assert(res == 0);

  pthread_attr_t attr;
  res = pthread_attr_init(&attr);
  assert(res == 0);
  res = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  assert(res == 0);

  pthread_t th;
  res = pthread_create(&th, &attr, detached_thread_func, NULL);
  assert(res == 0);
  pthread_attr_destroy(&attr);

  while (!atomic_load(&ready))
    sched_yield();

  return 0;
}
