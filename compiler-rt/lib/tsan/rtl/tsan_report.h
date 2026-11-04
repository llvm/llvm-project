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
  int id = 0;
  bool write = false;
};

struct ReportMop {
  int tid = kInvalidTid;
  uptr addr = 0;
  int size = 0;
  bool write = false;
  bool atomic = false;
  uptr external_tag = 0;
  Vector<ReportMopMutex> mset;
  StackTrace stack_trace;
  ReportStack* stack = nullptr;

  ReportMop();
  ~ReportMop();
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
  Tid id = kInvalidTid;
  ThreadID os_id = 0;
  bool running = false;
  ThreadType thread_type = ThreadType::Regular;
  char* name = nullptr;
  Tid parent_tid = kInvalidTid;
  StackID stack_id = 0;
  ReportStack* stack = nullptr;
  bool suppressable = false;
};

struct ReportMutex {
  int id = 0;
  uptr addr = 0;
  StackID stack_id = 0;
  ReportStack* stack = nullptr;
};

struct AddedStack {
  StackTrace stack_trace;
  bool suppressable = false;
};

class ReportDesc {
 public:
  ReportType typ = ReportTypeRace;
  uptr tag = kExternalTagNone;
  Vector<ReportStack*> stacks;
  Vector<AddedStack> added_stacks;
  Vector<ReportMop*> mops;
  Vector<ReportLocation*> locs;
  Vector<uptr> loc_addrs;
  Vector<ReportMutex*> mutexes;
  Vector<ReportThread*> threads;
  Vector<Tid> unique_tids;
  ReportStack* sleep = nullptr;
  StackID sleep_stack_id = 0;
  int count = 0;
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
