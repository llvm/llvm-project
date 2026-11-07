// RUN: %clangxx_tsan -O1 %s -DBUILD_SO -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// Regression test for deadlock between glibc NPTL stack_cache_lock (held in
// allocate_stack and __deallocate_stack -> __free_stacks -> _dl_deallocate_tls
// / munmap) and TSan's SlotLocker / DoReset / signal handling across
// pthread_create, pthread_join, pthread_tryjoin_np, and pthread_detach.

#ifdef BUILD_SO

// Because this shared library is loaded at runtime via dlopen(), a
// global-dynamic TLS variable cannot be placed in static TLS space. Instead,
// glibc lazily allocates a dynamic TLS block via malloc() when a thread first
// accesses dyn_tls, and frees that block via _dl_deallocate_tls() -> free()
// while holding stack_cache_lock when the thread's stack is later evicted from
// glibc's stack_cache.
__attribute__((tls_model("global-dynamic"))) __thread char dyn_tls[4096];

extern "C" void touch_dynamic_tls() {
  // Force glibc's __tls_get_addr slow path to malloc() the dynamic TLS chunk
  // on the calling thread.
  dyn_tls[0] = 1;
  dyn_tls[sizeof(dyn_tls) - 1] = 2;
}

#else // BUILD_SO

#  include "../test.h"
#  include <dlfcn.h>
#  include <errno.h>
#  include <pthread.h>
#  include <sched.h>
#  include <signal.h>
#  include <stdio.h>
#  include <stdlib.h>
#  include <string.h>
#  include <sys/syscall.h>
#  include <sys/time.h>
#  include <unistd.h>

static long counter;
static int stop;
static int sig_count;
static void (*touch_tls_fn)();

// Handshake flag between sigprof_handler and epoch_incrementer to probe whether
// stack_cache_lock is already held by the interrupted thread.
static int query_req;

// Set to 1 on a worker thread while inside the pthread_join /
// pthread_tryjoin_np / pthread_detach stack-reclamation loop.
static __thread int in_reclaim;

// Set to 1 while inside our interposed munmap() during stack eviction.
static __thread int in_munmap;

extern "C" int __interceptor_munmap(void *addr, size_t sz);

// Interpose on munmap() to intercept glibc's __free_stacks() unmapping evicted
// thread stacks while holding stack_cache_lock.
extern "C" int munmap(void *addr, size_t sz) {
  if (in_reclaim) {
    // Called from glibc's __free_stacks while stack_cache_lock is held.
    // 1. First invoke a dummy __interceptor_munmap(nullptr, 0). During
    //    pthread_join (which uses BLOCK_REAL and sets thr->in_blocking_func = 1),
    //    ScopedInterceptor's constructor clears thr->in_blocking_func = 0.
    //    If ScopedInterceptor's destructor erroneously restores
    //    thr->in_blocking_func = 1, the thread will re-enter "blocking" mode
    //    here even though the join wait is over and stack_cache_lock is held.
    __interceptor_munmap(nullptr, 0);
    in_munmap = 1;
    // 2. Send SIGPROF via a raw kernel syscall (bypassing TSan's pthread_kill
    //    interceptor so sctx->int_signal_send is not set and TSan treats it as
    //    an asynchronous signal):
    //    - If thr->in_blocking_func == 1, TSan's SignalHandler immediately
    //      invokes sigprof_handler synchronously here.
    //    - If thr->ignore_interceptors == 0 (e.g. in pthread_tryjoin_np or
    //      pthread_detach without ScopedIgnoreInterceptors), ScopedInterceptor's
    //      destructor at the end of __interceptor_munmap(addr, sz) below calls
    //      ProcessPendingSignals() and invokes sigprof_handler while in_munmap
    //      is still 1 and stack_cache_lock is still held.
    syscall(SYS_tgkill, getpid(), syscall(SYS_gettid), SIGPROF);
    int res = __interceptor_munmap(addr, sz);
    in_munmap = 0;
    return res;
  }
  return __interceptor_munmap(addr, sz);
}

static void *noop_child(void *arg) {
  // Allocate a dynamic TLS block so _dl_deallocate_tls() calls free() during
  // future stack cache eviction.
  if (touch_tls_fn)
    touch_tls_fn();
  // Publish our kernel TID so the parent worker can wait until the kernel task
  // has completely exited before reclaiming this thread.
  if (arg)
    __atomic_store_n((pid_t *)arg, (pid_t)syscall(SYS_gettid),
                     __ATOMIC_RELEASE);
  return nullptr;
}

static void sigprof_handler(int sig) {
  (void)sig;
  __atomic_fetch_add(&sig_count, 1, __ATOMIC_RELAXED);
  if (in_munmap && !__atomic_load_n(&stop, __ATOMIC_RELAXED)) {
    // TSan delivered this signal while glibc's __free_stacks was inside munmap
    // holding stack_cache_lock. Request epoch_incrementer to call
    // pthread_create (which locks stack_cache_lock in allocate_stack) and wait
    // for it to finish, which deterministically deadlocks.
    __atomic_store_n(&query_req, 1, __ATOMIC_RELEASE);
    while (__atomic_load_n(&query_req, __ATOMIC_ACQUIRE) != 0 &&
           !__atomic_load_n(&stop, __ATOMIC_RELAXED)) {
    }
  }
}

static void *epoch_incrementer(void *arg) {
  (void)arg;
  while (!__atomic_load_n(&stop, __ATOMIC_RELAXED)) {
    // Every sequentially-consistent atomic operation increments the thread's
    // TSan epoch and locks a TidSlot. Running this in a tight loop rapidly
    // overflows TSan's 14-bit kMaxEpoch (16384), forcing DoReset() to lock
    // every TidSlot mutex in order and stressing lock ordering against
    // stack_cache_lock (e.g. in ThreadStart and OnUserFree).
    __atomic_fetch_add(&counter, 1, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&query_req, __ATOMIC_ACQUIRE) != 0) {
      pthread_t t;
      if (pthread_create(&t, nullptr, noop_child, nullptr) == 0)
        pthread_join(t, nullptr);
      __atomic_store_n(&query_req, 0, __ATOMIC_RELEASE);
    }
  }
  return nullptr;
}

enum ReclaimMode {
  kJoin = 0,
  kTryJoin = 1,
  kDetach = 2,
};

static void *worker(void *arg) {
  long id = (long)arg;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  // Use 16 MiB stacks in batches of 4 (64 MiB total per batch) so freeing a
  // batch exceeds glibc's ~40 MiB stack_cache_maxsize and triggers
  // __free_stacks (_dl_deallocate_tls + munmap) while holding stack_cache_lock.
  pthread_attr_setstacksize(&attr, 16 << 20);

  const int kBatch = 4;
  for (int i = 0; i < 15; i++) {
    pthread_t th[kBatch];
    pid_t tids[kBatch];
    int created = 0;
    for (; created < kBatch; created++) {
      tids[created] = 0;
      if (pthread_create(&th[created], &attr, noop_child, &tids[created]) != 0)
        break;
    }
    // Wait until all child kernel tasks have exited (SYS_tgkill with signal 0
    // returns ESRCH once the kernel task is gone):
    // - For kDetach, detaching *after* the child exits forces pthread_detach()
    //   itself to call __deallocate_stack -> __free_stacks on the worker
    //   thread.
    // - For kJoin, joining an already-exited thread skips the blocking futex
    //   wait and immediately enters __deallocate_stack -> __free_stacks.
    for (int j = 0; j < created; j++) {
      pid_t tid = 0;
      while ((tid = __atomic_load_n(&tids[j], __ATOMIC_ACQUIRE)) == 0)
        sched_yield();
      while (syscall(SYS_tgkill, getpid(), tid, 0) == 0)
        sched_yield();
    }

    ReclaimMode mode = static_cast<ReclaimMode>((id + i) % 3);
    in_reclaim = 1;
    for (int j = 0; j < created; j++) {
      switch (mode) {
      case kJoin:
        pthread_join(th[j], nullptr);
        break;
      case kTryJoin:
        while (pthread_tryjoin_np(th[j], nullptr) == EBUSY)
          sched_yield();
        break;
      case kDetach:
        pthread_detach(th[j]);
        break;
      }
    }
    in_reclaim = 0;
  }

  pthread_attr_destroy(&attr);
  return nullptr;
}

int main(int argc, char *argv[]) {
  (void)argc;

  // Exit early if the test deadlocks.
  alarm(15);

  char so_path[4096];
  snprintf(so_path, sizeof(so_path), "%s-so.so", argv[0]);
  void *lib = dlopen(so_path, RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  touch_tls_fn = (void (*)())dlsym(lib, "touch_dynamic_tls");
  if (!touch_tls_fn) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 1;
  }

  struct sigaction act = {};
  act.sa_handler = &sigprof_handler;
  if (sigaction(SIGPROF, &act, nullptr)) {
    perror("sigaction");
    return 1;
  }

  itimerval t;
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 50;
  t.it_interval = t.it_value;
  if (setitimer(ITIMER_PROF, &t, nullptr)) {
    perror("setitimer");
    return 1;
  }

  pthread_t inc_th;
  pthread_create(&inc_th, nullptr, epoch_incrementer, nullptr);

  const int kWorkers = 4;
  pthread_t workers[kWorkers];
  for (long i = 0; i < kWorkers; i++)
    pthread_create(&workers[i], nullptr, worker, (void *)i);
  for (long i = 0; i < kWorkers; i++)
    pthread_join(workers[i], nullptr);

  __atomic_store_n(&stop, 1, __ATOMIC_RELAXED);
  pthread_join(inc_th, nullptr);
  dlclose(lib);

  fprintf(stderr, "DONE\n");
  return 0;
}

#endif // BUILD_SO

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK-NOT: ThreadSanitizer CHECK
// CHECK: DONE
