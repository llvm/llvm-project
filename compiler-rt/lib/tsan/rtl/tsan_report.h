//===-- tsan_report.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_REPORT_H
#define TSAN_REPORT_H

#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "sanitizer_common/sanitizer_thread_registry.h"
#include "sanitizer_common/sanitizer_vector.h"
#include "tsan_defs.h"
#include "tsan_mman.h"
#include "tsan_mutexset.h"
#include "tsan_shadow.h"
#include "tsan_stack_trace.h"

namespace __tsan {

enum ReportType {
  ReportTypeRace,
  ReportTypeVptrRace,
  ReportTypeUseAfterFree,
  ReportTypeVptrUseAfterFree,
  ReportTypeExternalRace,
  ReportTypeThreadLeak,
  ReportTypeMutexDestroyLocked,
  ReportTypeMutexDoubleLock,
  ReportTypeMutexInvalidAccess,
  ReportTypeMutexBadUnlock,
  ReportTypeMutexBadReadLock,
  ReportTypeMutexBadReadUnlock,
  ReportTypeSignalUnsafe,
  ReportTypeErrnoInSignal,
  ReportTypeDeadlock
};

struct ReportStack {
  StackTrace stack;
  SymbolizedStack* frames = nullptr;
  bool suppressable = false;
};

struct ReportMopMutex {
  int id = 0;
  bool write = false;
};

struct ReportMop {
  Tid tid = kInvalidTid;
  uptr addr = 0;
  int size = 0;
  bool write = false;
  bool atomic = false;
  uptr external_tag = kExternalTagNone;
  Vector<ReportMopMutex> mset;
  ReportStack stack;
};

enum ReportLocationType {
  ReportLocationInvalid,
  ReportLocationGlobal,
  ReportLocationHeap,
  ReportLocationStack,
  ReportLocationTLS,
  ReportLocationFD
};

struct ReportLocation {
  ReportLocationType type = ReportLocationInvalid;
  DataInfo global = {};
  uptr heap_chunk_start = 0;
  uptr heap_chunk_size = 0;
  uptr external_tag = kExternalTagNone;
  Tid tid = kInvalidTid;
  int fd = -1;
  bool suppressable = false;
  ReportStack stack;
};

struct ReportThread {
  Tid id = kInvalidTid;
  tid_t os_id = 0;
  bool running = false;
  ThreadType thread_type = ThreadType::Regular;
  char* name = nullptr;
  Tid parent_tid = kInvalidTid;
  ReportStack stack;
};

struct ReportMutex {
  int id = 0;
  uptr addr = 0;
  ReportStack stack;
};

class RegionAlloc {
public:
  template <typename T> T* Alloc() {
    T* obj = New<T>();
    objects_.PushBack({obj, [](void* p) {
                         T* obj = static_cast<T*>(p);
                         DestroyAndFree(obj);
                       }});
    return obj;
  }

  void* Alloc(uptr size);
  char* Strdup(const char* str);

  ~RegionAlloc();

private:
  struct Deleter {
    void* obj;
    void (*fn)(void* obj);
  };
  Vector<Deleter> objects_;
};

class ReportDesc {
 public:
   ReportType typ = ReportTypeRace;
   uptr tag = kExternalTagNone;
   Vector<ReportStack*> stacks;
   Vector<ReportMop*> mops;
   Vector<ReportLocation*> locs;
   Vector<ReportMutex*> mutexes;
   Vector<ReportThread*> threads;
   Vector<Tid> unique_tids;
   ReportStack* sleep = nullptr;
   int count = 0;

   void AddMemoryAccess(uptr addr, uptr external_tag, Shadow s, Tid tid,
                        StackTrace stack, const MutexSet* mset);
   void AddThread(const ThreadContext* tctx, bool suppressable = false);
   void AddThread(Tid unique_tid, bool suppressable = false);
   void AddStack(StackTrace stack, bool suppressable = false);
   void AddUniqueTid(Tid unique_tid);
   int AddMutex(uptr addr, StackID creation_stack_id);
   void AddLocation(uptr addr, uptr size);
   void AddSleep(StackID stack_id);
   ReportDesc() = default;

 private:
   RegionAlloc region;

   void AddHeapLocation(uptr addr, MBlock* b);

   ReportDesc(const ReportDesc&) = delete;
   void operator=(const ReportDesc&) = delete;
};

// Format and output the report.
void PrintReport(const ReportDesc *rep);
void PrintStack(const SymbolizedStack* frame);

}  // namespace __tsan

#endif  // TSAN_REPORT_H
