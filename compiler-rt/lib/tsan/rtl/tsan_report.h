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

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "sanitizer_common/sanitizer_thread_registry.h"
#include "sanitizer_common/sanitizer_vector.h"
#include "tsan_defs.h"

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
  ReportTypeDeadlock,
  ReportTypeMutexHeldWrongContext
};

struct ReportStack {
  SymbolizedStack *frames = nullptr;
  bool suppressable = false;
};

struct ReportMopMutex {
  int id;
  bool write;
};

struct ReportMop {
  int tid;
  uptr addr;
  int size;
  bool write;
  bool atomic;
  uptr external_tag;
  Vector<ReportMopMutex> mset;
  StackTrace stack_trace;
  ReportStack *stack;

  ReportMop();
};

enum ReportLocationType {
  ReportLocationGlobal,
  ReportLocationHeap,
  ReportLocationStack,
  ReportLocationTLS,
  ReportLocationFD
};

struct ReportLocation {
  ReportLocationType type = ReportLocationGlobal;
  DataInfo global = {};
  uptr heap_chunk_start = 0;
  uptr heap_chunk_size = 0;
  uptr external_tag = 0;
  Tid tid = kInvalidTid;
  int fd = 0;
  bool fd_closed = false;
  bool suppressable = false;
  StackID stack_id = 0;
  ReportStack *stack = nullptr;
};

struct ReportThread {
  Tid id;
  ThreadID os_id;
  bool running;
  ThreadType thread_type;
  char *name;
  Tid parent_tid;
  StackID stack_id;
  ReportStack *stack;
  bool suppressable;
};

struct ReportMutex {
  int id;
  uptr addr;
  StackID stack_id;
  ReportStack *stack;
};

struct AddedLocationAddr {
  uptr addr;
  usize locs_idx;
};

class ReportDesc {
 public:
  ReportType typ;
  uptr tag;
  Vector<ReportStack*> stacks;
  Vector<ReportMop*> mops;
  Vector<ReportLocation*> locs;
  Vector<AddedLocationAddr> added_location_addrs;
  Vector<ReportMutex*> mutexes;
  Vector<ReportThread*> threads;
  Vector<Tid> unique_tids;
  ReportStack *sleep;
  int count;
  int signum = 0;

  ReportDesc();
  ~ReportDesc();

 private:
  ReportDesc(const ReportDesc&);
  void operator = (const ReportDesc&);
};

// Format and output the report to the console/log. No additional logic.
void PrintReport(const ReportDesc *rep);
void PrintStack(const ReportStack *stack);

}  // namespace __tsan

#endif  // TSAN_REPORT_H
