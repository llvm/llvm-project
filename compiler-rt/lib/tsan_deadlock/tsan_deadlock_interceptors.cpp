//===-- tsan_deadlock_interceptors.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DeadlockSanitizer.
//
//===----------------------------------------------------------------------===//

#include <pthread.h>

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_glibc_version.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "tsan_deadlock_rtl.h"

using namespace __tsan_deadlock;

__attribute__((tls_model("initial-exec"))) static __thread volatile int initing;
static bool inited;

static bool InitThread() {
  if (initing)
    return false;
  if (cur_thread())
    return true;
  initing = true;
  if (!inited) {
    inited = true;
    Initialize();
  }
  ThreadInit(&thr_tls);
  initing = false;
  return true;
}

struct ThreadArg {
  void *(*fn)(void *);
  void *arg;
};

static void *ThreadTrampoline(void *arg) {
  ThreadArg *targ = static_cast<ThreadArg *>(arg);
  void *(*fn)(void *) = targ->fn;
  void *fn_arg = targ->arg;
  InternalFree(targ);
  ThreadInit(&thr_tls);
  void *retval = fn(fn_arg);
  // Also called from the pthread_exit interceptor; guard with is_inited to
  // stay idempotent.
  if (thr_tls.is_inited)
    ThreadDestroy(&thr_tls);
  return retval;
}

INTERCEPTOR(int, pthread_create, pthread_t *th, const pthread_attr_t *attr,
            void *(*fn)(void *), void *arg) {
  InitThread();
  ThreadArg *targ = static_cast<ThreadArg *>(InternalAlloc(sizeof(ThreadArg)));
  targ->fn = fn;
  targ->arg = arg;
  return REAL(pthread_create)(th, attr, ThreadTrampoline, targ);
}

INTERCEPTOR(int, pthread_join, pthread_t t, void **retval) {
  InitThread();
  return REAL(pthread_join)(t, retval);
}

INTERCEPTOR(void, pthread_exit, void *retval) {
  if (thr_tls.is_inited)
    ThreadDestroy(&thr_tls);
  REAL(pthread_exit)(retval);
}

INTERCEPTOR(int, pthread_mutex_init, pthread_mutex_t *m,
            const pthread_mutexattr_t *attr) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_mutex_init)(m, attr);
  if (res == 0) {
    bool reentrant = false;
    if (attr) {
      int type = 0;
      if (pthread_mutexattr_gettype(attr, &type) == 0)
        reentrant = (type == PTHREAD_MUTEX_RECURSIVE);
    }
    MutexInit(cur_thread(), (uptr)m, reentrant, pc);
  }
  return res;
}

INTERCEPTOR(int, pthread_mutex_destroy, pthread_mutex_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_mutex_destroy)(m);
  if (res == 0 || res == errno_EBUSY)
    MutexDestroy(cur_thread(), (uptr)m, pc);
  return res;
}

INTERCEPTOR(int, pthread_mutex_lock, pthread_mutex_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeLock(cur_thread(), (uptr)m, true, pc);
  int res = REAL(pthread_mutex_lock)(m);
  MutexAfterLock(cur_thread(), (uptr)m, true, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_mutex_trylock, pthread_mutex_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_mutex_trylock)(m);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, true, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_mutex_timedlock, pthread_mutex_t *m,
            const struct timespec *abstime) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_mutex_timedlock)(m, abstime);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, true, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_mutex_unlock, pthread_mutex_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeUnlock(cur_thread(), (uptr)m, true, pc);
  return REAL(pthread_mutex_unlock)(m);
}

INTERCEPTOR(int, pthread_spin_init, pthread_spinlock_t *m, int pshared) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_spin_init)(m, pshared);
  if (res == 0)
    MutexInit(cur_thread(), (uptr)m, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_spin_destroy, pthread_spinlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_spin_destroy)(m);
  if (res == 0)
    MutexDestroy(cur_thread(), (uptr)m, pc);
  return res;
}

INTERCEPTOR(int, pthread_spin_lock, pthread_spinlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeLock(cur_thread(), (uptr)m, true, pc);
  int res = REAL(pthread_spin_lock)(m);
  MutexAfterLock(cur_thread(), (uptr)m, true, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_spin_trylock, pthread_spinlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_spin_trylock)(m);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, true, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_spin_unlock, pthread_spinlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeUnlock(cur_thread(), (uptr)m, true, pc);
  return REAL(pthread_spin_unlock)(m);
}

INTERCEPTOR(int, pthread_rwlock_init, pthread_rwlock_t *m,
            const pthread_rwlockattr_t *attr) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_rwlock_init)(m, attr);
  if (res == 0)
    MutexInit(cur_thread(), (uptr)m, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_destroy, pthread_rwlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexDestroy(cur_thread(), (uptr)m, pc);
  return REAL(pthread_rwlock_destroy)(m);
}

INTERCEPTOR(int, pthread_rwlock_rdlock, pthread_rwlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeLock(cur_thread(), (uptr)m, false, pc);
  int res = REAL(pthread_rwlock_rdlock)(m);
  MutexAfterLock(cur_thread(), (uptr)m, false, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_tryrdlock, pthread_rwlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_rwlock_tryrdlock)(m);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, false, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_timedrdlock, pthread_rwlock_t *m,
            const timespec *abstime) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_rwlock_timedrdlock)(m, abstime);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, false, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_wrlock, pthread_rwlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeLock(cur_thread(), (uptr)m, true, pc);
  int res = REAL(pthread_rwlock_wrlock)(m);
  MutexAfterLock(cur_thread(), (uptr)m, true, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_trywrlock, pthread_rwlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_rwlock_trywrlock)(m);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, true, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_timedwrlock, pthread_rwlock_t *m,
            const timespec *abstime) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  int res = REAL(pthread_rwlock_timedwrlock)(m, abstime);
  if (res == 0)
    MutexAfterLock(cur_thread(), (uptr)m, true, true, pc);
  return res;
}

INTERCEPTOR(int, pthread_rwlock_unlock, pthread_rwlock_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeUnlock(cur_thread(), (uptr)m, false, pc);
  return REAL(pthread_rwlock_unlock)(m);
}

static pthread_cond_t *init_cond(pthread_cond_t *c, bool force = false) {
  if (!common_flags()->legacy_pthread_cond)
    return c;
  atomic_uintptr_t *p = (atomic_uintptr_t *)c;
  uptr cond = atomic_load(p, memory_order_acquire);
  if (!force && cond != 0)
    return (pthread_cond_t *)cond;
  void *newcond = InternalAlloc(sizeof(pthread_cond_t));
  internal_memset(newcond, 0, sizeof(pthread_cond_t));
  if (atomic_compare_exchange_strong(p, &cond, (uptr)newcond,
                                     memory_order_acq_rel))
    return (pthread_cond_t *)newcond;
  InternalFree(newcond);
  return (pthread_cond_t *)cond;
}

INTERCEPTOR(int, pthread_cond_init, pthread_cond_t *c,
            const pthread_condattr_t *a) {
  InitThread();
  return REAL(pthread_cond_init)(init_cond(c, true), a);
}

INTERCEPTOR(int, pthread_cond_wait, pthread_cond_t *c, pthread_mutex_t *m) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeUnlock(cur_thread(), (uptr)m, true, pc);
  int res = REAL(pthread_cond_wait)(init_cond(c), m);
  MutexBeforeLock(cur_thread(), (uptr)m, true, pc);
  MutexAfterLock(cur_thread(), (uptr)m, true, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_cond_timedwait, pthread_cond_t *c, pthread_mutex_t *m,
            const timespec *abstime) {
  uptr pc = GET_CURRENT_PC();
  InitThread();
  MutexBeforeUnlock(cur_thread(), (uptr)m, true, pc);
  int res = REAL(pthread_cond_timedwait)(init_cond(c), m, abstime);
  MutexBeforeLock(cur_thread(), (uptr)m, true, pc);
  MutexAfterLock(cur_thread(), (uptr)m, true, false, pc);
  return res;
}

INTERCEPTOR(int, pthread_cond_signal, pthread_cond_t *c) {
  InitThread();
  return REAL(pthread_cond_signal)(init_cond(c));
}

INTERCEPTOR(int, pthread_cond_broadcast, pthread_cond_t *c) {
  InitThread();
  return REAL(pthread_cond_broadcast)(init_cond(c));
}

INTERCEPTOR(int, pthread_cond_destroy, pthread_cond_t *c) {
  InitThread();
  pthread_cond_t *cond = init_cond(c);
  int res = REAL(pthread_cond_destroy)(cond);
  if (common_flags()->legacy_pthread_cond) {
    InternalFree(cond);
    atomic_store((atomic_uintptr_t *)c, 0, memory_order_relaxed);
  }
  return res;
}

INTERCEPTOR(char *, realpath, const char *path, char *resolved_path) {
  InitThread();
  return REAL(realpath)(path, resolved_path);
}

INTERCEPTOR(SSIZE_T, read, int fd, void *ptr, SIZE_T count) {
  InitThread();
  return REAL(read)(fd, ptr, count);
}

INTERCEPTOR(SSIZE_T, pread, int fd, void *ptr, SIZE_T count, OFF_T offset) {
  InitThread();
  return REAL(pread)(fd, ptr, count, offset);
}

namespace __tsan_deadlock {

void InitializeInterceptors() {
  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(pthread_join);
  INTERCEPT_FUNCTION(pthread_exit);

  INTERCEPT_FUNCTION(pthread_mutex_init);
  INTERCEPT_FUNCTION(pthread_mutex_destroy);
  INTERCEPT_FUNCTION(pthread_mutex_lock);
  INTERCEPT_FUNCTION(pthread_mutex_trylock);
  INTERCEPT_FUNCTION(pthread_mutex_timedlock);
  INTERCEPT_FUNCTION(pthread_mutex_unlock);

  INTERCEPT_FUNCTION(pthread_spin_init);
  INTERCEPT_FUNCTION(pthread_spin_destroy);
  INTERCEPT_FUNCTION(pthread_spin_lock);
  INTERCEPT_FUNCTION(pthread_spin_trylock);
  INTERCEPT_FUNCTION(pthread_spin_unlock);

  INTERCEPT_FUNCTION(pthread_rwlock_init);
  INTERCEPT_FUNCTION(pthread_rwlock_destroy);
  INTERCEPT_FUNCTION(pthread_rwlock_rdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_tryrdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_timedrdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_wrlock);
  INTERCEPT_FUNCTION(pthread_rwlock_trywrlock);
  INTERCEPT_FUNCTION(pthread_rwlock_timedwrlock);
  INTERCEPT_FUNCTION(pthread_rwlock_unlock);

  // See the comment in tsan_interceptors_posix.cpp.
#if SANITIZER_GLIBC && !__GLIBC_PREREQ(2, 36) &&                               \
    (defined(__x86_64__) || defined(__mips__) || SANITIZER_PPC64V1 ||          \
     defined(__s390x__))
  INTERCEPT_FUNCTION_VER(pthread_cond_init, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_signal, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_broadcast, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_wait, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_timedwait, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_destroy, "GLIBC_2.3.2");
#else
  INTERCEPT_FUNCTION(pthread_cond_init);
  INTERCEPT_FUNCTION(pthread_cond_signal);
  INTERCEPT_FUNCTION(pthread_cond_broadcast);
  INTERCEPT_FUNCTION(pthread_cond_wait);
  INTERCEPT_FUNCTION(pthread_cond_timedwait);
  INTERCEPT_FUNCTION(pthread_cond_destroy);
#endif

  INTERCEPT_FUNCTION(realpath);
  INTERCEPT_FUNCTION(read);
  INTERCEPT_FUNCTION(pread);
}

} // namespace __tsan_deadlock
