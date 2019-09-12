//===-- tsan_test_util_posix.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Test utils, Linux, FreeBSD, NetBSD and Darwin implementation.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_atomic.h"
#include "tsan_interface.h"
#include "tsan_posix_util.h"
#include "tsan_test_util.h"
#include "tsan_report.h"

#include "gtest/gtest.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

using namespace __tsan;

static __thread bool expect_report;
static __thread bool expect_report_reported;
static __thread ReportType expect_report_type;

static void *BeforeInitThread(void *param) {
  (void)param;
  return 0;
}

static void AtExit() {
}

void TestMutexBeforeInit() {
  // Mutexes must be usable before __tsan_init();
  pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  __interceptor_pthread_mutex_lock(&mtx);
  __interceptor_pthread_mutex_unlock(&mtx);
  __interceptor_pthread_mutex_destroy(&mtx);
  pthread_t thr;
  __interceptor_pthread_create(&thr, 0, BeforeInitThread, 0);
  __interceptor_pthread_join(thr, 0);
  atexit(AtExit);
}

namespace __tsan {
bool OnReport(const ReportDesc *rep, bool suppressed) {
  if (expect_report) {
    if (rep->typ != expect_report_type) {
      printf("Expected report of type %d, got type %d\n",
             (int)expect_report_type, (int)rep->typ);
      EXPECT_TRUE(false) << "Wrong report type";
      return false;
    }
  } else {
    EXPECT_TRUE(false) << "Unexpected report";
    return false;
  }
  expect_report_reported = true;
  return true;
}
}  // namespace __tsan

static void* allocate_addr(int size, int offset_from_aligned = 0) {
  static uintptr_t foo;
  static atomic_uintptr_t uniq = {(uintptr_t)&foo};  // Some real address.
  const int kAlign = 16;
  CHECK(offset_from_aligned < kAlign);
  size = (size + 2 * kAlign) & ~(kAlign - 1);
  uintptr_t addr = atomic_fetch_add(&uniq, size, memory_order_relaxed);
  return (void*)(addr + offset_from_aligned);
}

MemLoc::MemLoc(int offset_from_aligned)
  : loc_(allocate_addr(16, offset_from_aligned)) {
}

MemLoc::~MemLoc() {
}

Mutex::Mutex(Type type)
  : alive_()
  , type_(type) {
}

Mutex::~Mutex() {
  CHECK(!alive_);
}

void Mutex::Init() {
  CHECK(!alive_);
  alive_ = true;
  if (type_ == Normal)
    CHECK_EQ(__interceptor_pthread_mutex_init((pthread_mutex_t*)mtx_, 0), 0);
#ifndef __APPLE__
  else if (type_ == Spin)
    CHECK_EQ(pthread_spin_init((pthread_spinlock_t*)mtx_, 0), 0);
#endif
  else if (type_ == RW)
    CHECK_EQ(__interceptor_pthread_rwlock_init((pthread_rwlock_t*)mtx_, 0), 0);
  else
    CHECK(0);
}

void Mutex::StaticInit() {
  CHECK(!alive_);
  CHECK(type_ == Normal);
  alive_ = true;
  pthread_mutex_t tmp = PTHREAD_MUTEX_INITIALIZER;
  memcpy(mtx_, &tmp, sizeof(tmp));
}

void Mutex::Destroy() {
  CHECK(alive_);
  alive_ = false;
  if (type_ == Normal)
    CHECK_EQ(__interceptor_pthread_mutex_destroy((pthread_mutex_t*)mtx_), 0);
#ifndef __APPLE__
  else if (type_ == Spin)
    CHECK_EQ(pthread_spin_destroy((pthread_spinlock_t*)mtx_), 0);
#endif
  else if (type_ == RW)
    CHECK_EQ(__interceptor_pthread_rwlock_destroy((pthread_rwlock_t*)mtx_), 0);
}

void Mutex::Lock() {
  CHECK(alive_);
  if (type_ == Normal)
    CHECK_EQ(__interceptor_pthread_mutex_lock((pthread_mutex_t*)mtx_), 0);
#ifndef __APPLE__
  else if (type_ == Spin)
    CHECK_EQ(pthread_spin_lock((pthread_spinlock_t*)mtx_), 0);
#endif
  else if (type_ == RW)
    CHECK_EQ(__interceptor_pthread_rwlock_wrlock((pthread_rwlock_t*)mtx_), 0);
}

bool Mutex::TryLock() {
  CHECK(alive_);
  if (type_ == Normal)
    return __interceptor_pthread_mutex_trylock((pthread_mutex_t*)mtx_) == 0;
#ifndef __APPLE__
  else if (type_ == Spin)
    return pthread_spin_trylock((pthread_spinlock_t*)mtx_) == 0;
#endif
  else if (type_ == RW)
    return __interceptor_pthread_rwlock_trywrlock((pthread_rwlock_t*)mtx_) == 0;
  return false;
}

void Mutex::Unlock() {
  CHECK(alive_);
  if (type_ == Normal)
    CHECK_EQ(__interceptor_pthread_mutex_unlock((pthread_mutex_t*)mtx_), 0);
#ifndef __APPLE__
  else if (type_ == Spin)
    CHECK_EQ(pthread_spin_unlock((pthread_spinlock_t*)mtx_), 0);
#endif
  else if (type_ == RW)
    CHECK_EQ(__interceptor_pthread_rwlock_unlock((pthread_rwlock_t*)mtx_), 0);
}

void Mutex::ReadLock() {
  CHECK(alive_);
  CHECK(type_ == RW);
  CHECK_EQ(__interceptor_pthread_rwlock_rdlock((pthread_rwlock_t*)mtx_), 0);
}

bool Mutex::TryReadLock() {
  CHECK(alive_);
  CHECK(type_ == RW);
  return __interceptor_pthread_rwlock_tryrdlock((pthread_rwlock_t*)mtx_) ==  0;
}

void Mutex::ReadUnlock() {
  CHECK(alive_);
  CHECK(type_ == RW);
  CHECK_EQ(__interceptor_pthread_rwlock_unlock((pthread_rwlock_t*)mtx_), 0);
}

struct Event {
  enum Type {
    SHUTDOWN,
    READ,
    WRITE,
    VPTR_UPDATE,
    CALL,
    RETURN,
    MUTEX_CREATE,
    MUTEX_DESTROY,
    MUTEX_LOCK,
    MUTEX_TRYLOCK,
    MUTEX_UNLOCK,
    MUTEX_READLOCK,
    MUTEX_TRYREADLOCK,
    MUTEX_READUNLOCK,
    MEMCPY,
    MEMSET
  };
  Type type;
  void *ptr;
  uptr arg;
  uptr arg2;
  bool res;
  bool expect_report;
  ReportType report_type;

  explicit Event(Type type, const void *ptr = 0, uptr arg = 0, uptr arg2 = 0)
      : type(type),
        ptr(const_cast<void *>(ptr)),
        arg(arg),
        arg2(arg2),
        res(),
        expect_report(),
        report_type() {}

  void ExpectReport(ReportType type) {
    expect_report = true;
    report_type = type;
  }
};

struct ScopedThread::Impl {
  pthread_t thread;
  bool main;
  bool detached;
  atomic_uintptr_t event;  // Event*

  static void *ScopedThreadCallback(void *arg);
  void send(Event *ev);
  void HandleEvent(Event *ev);
};

void ScopedThread::Impl::HandleEvent(Event *ev) {
  CHECK_EQ(expect_report, false);
  expect_report = ev->expect_report;
  expect_report_reported = false;
  expect_report_type = ev->report_type;
  switch (ev->type) {
  case Event::READ:
  case Event::WRITE: {
    void (*tsan_mop)(void *addr) = 0;
    if (ev->type == Event::READ) {
      switch (ev->arg /*size*/) {
        case 1: tsan_mop = __tsan_read1; break;
        case 2: tsan_mop = __tsan_read2; break;
        case 4: tsan_mop = __tsan_read4; break;
        case 8: tsan_mop = __tsan_read8; break;
        case 16: tsan_mop = __tsan_read16; break;
      }
    } else {
      switch (ev->arg /*size*/) {
        case 1: tsan_mop = __tsan_write1; break;
        case 2: tsan_mop = __tsan_write2; break;
        case 4: tsan_mop = __tsan_write4; break;
        case 8: tsan_mop = __tsan_write8; break;
        case 16: tsan_mop = __tsan_write16; break;
      }
    }
    CHECK_NE(tsan_mop, 0);
#if defined(__FreeBSD__) || defined(__APPLE__) || defined(__NetBSD__)
    const int ErrCode = ESOCKTNOSUPPORT;
#else
    const int ErrCode = ECHRNG;
#endif
    errno = ErrCode;
    tsan_mop(ev->ptr);
    CHECK_EQ(ErrCode, errno);  // In no case must errno be changed.
    break;
  }
  case Event::VPTR_UPDATE:
    __tsan_vptr_update((void**)ev->ptr, (void*)ev->arg);
    break;
  case Event::CALL:
    __tsan_func_entry((void*)((uptr)ev->ptr));
    break;
  case Event::RETURN:
    __tsan_func_exit();
    break;
  case Event::MUTEX_CREATE:
    static_cast<Mutex*>(ev->ptr)->Init();
    break;
  case Event::MUTEX_DESTROY:
    static_cast<Mutex*>(ev->ptr)->Destroy();
    break;
  case Event::MUTEX_LOCK:
    static_cast<Mutex*>(ev->ptr)->Lock();
    break;
  case Event::MUTEX_TRYLOCK:
    ev->res = static_cast<Mutex*>(ev->ptr)->TryLock();
    break;
  case Event::MUTEX_UNLOCK:
    static_cast<Mutex*>(ev->ptr)->Unlock();
    break;
  case Event::MUTEX_READLOCK:
    static_cast<Mutex*>(ev->ptr)->ReadLock();
    break;
  case Event::MUTEX_TRYREADLOCK:
    ev->res = static_cast<Mutex*>(ev->ptr)->TryReadLock();
    break;
  case Event::MUTEX_READUNLOCK:
    static_cast<Mutex*>(ev->ptr)->ReadUnlock();
    break;
  case Event::MEMCPY:
    __interceptor_memcpy(ev->ptr, (void*)ev->arg, ev->arg2);
    break;
  case Event::MEMSET:
    __interceptor_memset(ev->ptr, ev->arg, ev->arg2);
    break;
  default: CHECK(0);
  }
  if (expect_report && !expect_report_reported) {
    printf("Missed expected report of type %d\n", (int)ev->report_type);
    EXPECT_TRUE(false) << "Missed expected race";
  }
  expect_report = false;
}

void *ScopedThread::Impl::ScopedThreadCallback(void *arg) {
  __tsan_func_entry(__builtin_return_address(0));
  Impl *impl = (Impl*)arg;
  for (;;) {
    Event* ev = (Event*)atomic_load(&impl->event, memory_order_acquire);
    if (ev == 0) {
      sched_yield();
      continue;
    }
    if (ev->type == Event::SHUTDOWN) {
      atomic_store(&impl->event, 0, memory_order_release);
      break;
    }
    impl->HandleEvent(ev);
    atomic_store(&impl->event, 0, memory_order_release);
  }
  __tsan_func_exit();
  return 0;
}

void ScopedThread::Impl::send(Event *e) {
  if (main) {
    HandleEvent(e);
  } else {
    CHECK_EQ(atomic_load(&event, memory_order_relaxed), 0);
    atomic_store(&event, (uintptr_t)e, memory_order_release);
    while (atomic_load(&event, memory_order_acquire) != 0)
      sched_yield();
  }
}

ScopedThread::ScopedThread(bool detached, bool main) {
  impl_ = new Impl;
  impl_->main = main;
  impl_->detached = detached;
  atomic_store(&impl_->event, 0, memory_order_relaxed);
  if (!main) {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(
        &attr, detached ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE);
    pthread_attr_setstacksize(&attr, 64*1024);
    __interceptor_pthread_create(&impl_->thread, &attr,
        ScopedThread::Impl::ScopedThreadCallback, impl_);
  }
}

ScopedThread::~ScopedThread() {
  if (!impl_->main) {
    Event event(Event::SHUTDOWN);
    impl_->send(&event);
    if (!impl_->detached)
      __interceptor_pthread_join(impl_->thread, 0);
  }
  delete impl_;
}

void ScopedThread::Detach() {
  CHECK(!impl_->main);
  CHECK(!impl_->detached);
  impl_->detached = true;
  __interceptor_pthread_detach(impl_->thread);
}

void ScopedThread::Access(void *addr, bool is_write,
                          int size, bool expect_race) {
  Event event(is_write ? Event::WRITE : Event::READ, addr, size);
  if (expect_race)
    event.ExpectReport(ReportTypeRace);
  impl_->send(&event);
}

void ScopedThread::VptrUpdate(const MemLoc &vptr,
                              const MemLoc &new_val,
                              bool expect_race) {
  Event event(Event::VPTR_UPDATE, vptr.loc(), (uptr)new_val.loc());
  if (expect_race)
    event.ExpectReport(ReportTypeRace);
  impl_->send(&event);
}

void ScopedThread::Call(void(*pc)()) {
  Event event(Event::CALL, (void*)((uintptr_t)pc));
  impl_->send(&event);
}

void ScopedThread::Return() {
  Event event(Event::RETURN);
  impl_->send(&event);
}

void ScopedThread::Create(const Mutex &m) {
  Event event(Event::MUTEX_CREATE, &m);
  impl_->send(&event);
}

void ScopedThread::Destroy(const Mutex &m) {
  Event event(Event::MUTEX_DESTROY, &m);
  impl_->send(&event);
}

void ScopedThread::Lock(const Mutex &m) {
  Event event(Event::MUTEX_LOCK, &m);
  impl_->send(&event);
}

bool ScopedThread::TryLock(const Mutex &m) {
  Event event(Event::MUTEX_TRYLOCK, &m);
  impl_->send(&event);
  return event.res;
}

void ScopedThread::Unlock(const Mutex &m) {
  Event event(Event::MUTEX_UNLOCK, &m);
  impl_->send(&event);
}

void ScopedThread::ReadLock(const Mutex &m) {
  Event event(Event::MUTEX_READLOCK, &m);
  impl_->send(&event);
}

bool ScopedThread::TryReadLock(const Mutex &m) {
  Event event(Event::MUTEX_TRYREADLOCK, &m);
  impl_->send(&event);
  return event.res;
}

void ScopedThread::ReadUnlock(const Mutex &m) {
  Event event(Event::MUTEX_READUNLOCK, &m);
  impl_->send(&event);
}

void ScopedThread::Memcpy(void *dst, const void *src, int size,
                          bool expect_race) {
  Event event(Event::MEMCPY, dst, (uptr)src, size);
  if (expect_race)
    event.ExpectReport(ReportTypeRace);
  impl_->send(&event);
}

void ScopedThread::Memset(void *dst, int val, int size,
                          bool expect_race) {
  Event event(Event::MEMSET, dst, val, size);
  if (expect_race)
    event.ExpectReport(ReportTypeRace);
  impl_->send(&event);
}
