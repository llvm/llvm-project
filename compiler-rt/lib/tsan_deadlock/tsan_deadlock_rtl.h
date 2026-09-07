//===-- tsan_deadlock_rtl.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DeadlockSanitizer.
//
// Main internal header.
//
//===----------------------------------------------------------------------===//

#ifndef TSAN_DEADLOCK_RTL_H
#define TSAN_DEADLOCK_RTL_H

#include "sanitizer_common/sanitizer_addrhashmap.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_deadlock_detector_interface.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __tsan_deadlock {

struct Flags : DDFlags {
  bool report_destroy_locked;
  bool report_mutex_bugs;
  bool report_bugs;
  bool halt_on_error;

  void SetDefaults() {
    second_deadlock_stack = false;
    report_destroy_locked = true;
    report_mutex_bugs = true;
    report_bugs = true;
    halt_on_error = false;
  }
};

struct UserMutex {
  DDMutex dd;
  atomic_uintptr_t owner_tid;
  u32 recursion;
  u32 creation_stk;
  u32 last_lock_stk;
  bool reentrant;
};

struct Thread {
  uptr *shadow_stack_pos;
  uptr *shadow_stack;
  uptr *shadow_stack_end;

  DDPhysicalThread *dd_pt;
  DDLogicalThread *dd_lt;

  int ignore_interceptors;
  int unique_id;
  bool is_inited;
};

struct Callback final : public DDCallback {
  Thread *thr;
  uptr pc;

  explicit Callback(Thread *thr, uptr pc);
  u32 Unwind() override;
  int UniqueTid() override;
};

typedef AddrHashMap<UserMutex, 31051> MutexHashMap;

struct Context {
  DDetector *dd;
  Mutex report_mutex;
  MutexHashMap mutex_map;
};

extern Flags tsan_deadlock_flags;
inline Flags *flags() { return &tsan_deadlock_flags; }

#if SANITIZER_APPLE
Thread *cur_thread();
void set_cur_thread(Thread *thr);
#else
__attribute__((tls_model("initial-exec"))) extern THREADLOCAL Thread thr_tls;
inline Thread *cur_thread() { return &thr_tls; }
#endif

void Initialize();
void InitializeInterceptors();

void ThreadInit(Thread *thr);
void ThreadDestroy(Thread *thr);

void MutexInit(Thread *thr, uptr m, bool reentrant, uptr pc = 0);
void MutexBeforeLock(Thread *thr, uptr m, bool writelock, uptr pc);
void MutexAfterLock(Thread *thr, uptr m, bool writelock, bool trylock, uptr pc);
void MutexBeforeUnlock(Thread *thr, uptr m, bool writelock, uptr pc);
void MutexDestroy(Thread *thr, uptr m, uptr pc);

} // namespace __tsan_deadlock

#endif // TSAN_DEADLOCK_RTL_H
