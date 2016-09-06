//===-- tsan_debugging.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// TSan debugging API implementation.
//===----------------------------------------------------------------------===//
#include "tsan_interface.h"
#include "tsan_report.h"
#include "tsan_rtl.h"

using namespace __tsan;

static const char *ReportTypeDescription(ReportType typ) {
  if (typ == ReportTypeRace) return "data-race";
  if (typ == ReportTypeVptrRace) return "data-race-vptr";
  if (typ == ReportTypeUseAfterFree) return "heap-use-after-free";
  if (typ == ReportTypeVptrUseAfterFree) return "heap-use-after-free-vptr";
  if (typ == ReportTypeThreadLeak) return "thread-leak";
  if (typ == ReportTypeMutexDestroyLocked) return "locked-mutex-destroy";
  if (typ == ReportTypeMutexDoubleLock) return "mutex-double-lock";
  if (typ == ReportTypeMutexInvalidAccess) return "mutex-invalid-access";
  if (typ == ReportTypeMutexBadUnlock) return "mutex-bad-unlock";
  if (typ == ReportTypeMutexBadReadLock) return "mutex-bad-read-lock";
  if (typ == ReportTypeMutexBadReadUnlock) return "mutex-bad-read-unlock";
  if (typ == ReportTypeSignalUnsafe) return "signal-unsafe-call";
  if (typ == ReportTypeErrnoInSignal) return "errno-in-signal-handler";
  if (typ == ReportTypeDeadlock) return "lock-order-inversion";
  return "";
}

static const char *ReportLocationTypeDescription(ReportLocationType typ) {
  if (typ == ReportLocationGlobal) return "global";
  if (typ == ReportLocationHeap) return "heap";
  if (typ == ReportLocationStack) return "stack";
  if (typ == ReportLocationTLS) return "tls";
  if (typ == ReportLocationFD) return "fd";
  return "";
}

static void CopyTrace(SymbolizedStack *first_frame, void **trace,
                      uptr trace_size) {
  uptr i = 0;
  for (SymbolizedStack *frame = first_frame; frame != nullptr;
       frame = frame->next) {
    trace[i++] = (void *)frame->info.address;
    if (i >= trace_size) break;
  }
}

// Meant to be called by the debugger.
SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_get_current_report() {
  return const_cast<ReportDesc*>(cur_thread()->current_report);
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_data(void *report, const char **description, int *count,
                           int *stack_count, int *mop_count, int *loc_count,
                           int *mutex_count, int *thread_count,
                           int *unique_tid_count, void **sleep_trace,
                           uptr trace_size) {
  const ReportDesc *rep = (ReportDesc *)report;
  *description = ReportTypeDescription(rep->typ);
  *count = rep->count;
  *stack_count = rep->stacks.Size();
  *mop_count = rep->mops.Size();
  *loc_count = rep->locs.Size();
  *mutex_count = rep->mutexes.Size();
  *thread_count = rep->threads.Size();
  *unique_tid_count = rep->unique_tids.Size();
  if (rep->sleep) CopyTrace(rep->sleep->frames, sleep_trace, trace_size);
  return 1;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_stack(void *report, uptr idx, void **trace,
                            uptr trace_size) {
  const ReportDesc *rep = (ReportDesc *)report;
  CHECK_LT(idx, rep->stacks.Size());
  ReportStack *stack = rep->stacks[idx];
  if (stack) CopyTrace(stack->frames, trace, trace_size);
  return stack ? 1 : 0;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_mop(void *report, uptr idx, int *tid, void **addr,
                          int *size, int *write, int *atomic, void **trace,
                          uptr trace_size) {
  const ReportDesc *rep = (ReportDesc *)report;
  CHECK_LT(idx, rep->mops.Size());
  ReportMop *mop = rep->mops[idx];
  *tid = mop->tid;
  *addr = (void *)mop->addr;
  *size = mop->size;
  *write = mop->write ? 1 : 0;
  *atomic = mop->atomic ? 1 : 0;
  if (mop->stack) CopyTrace(mop->stack->frames, trace, trace_size);
  return 1;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_loc(void *report, uptr idx, const char **type,
                          void **addr, uptr *start, uptr *size, int *tid,
                          int *fd, int *suppressable, void **trace,
                          uptr trace_size) {
  const ReportDesc *rep = (ReportDesc *)report;
  CHECK_LT(idx, rep->locs.Size());
  ReportLocation *loc = rep->locs[idx];
  *type = ReportLocationTypeDescription(loc->type);
  *addr = (void *)loc->global.start;
  *start = loc->heap_chunk_start;
  *size = loc->heap_chunk_size;
  *tid = loc->tid;
  *fd = loc->fd;
  *suppressable = loc->suppressable;
  if (loc->stack) CopyTrace(loc->stack->frames, trace, trace_size);
  return 1;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_mutex(void *report, uptr idx, uptr *mutex_id, void **addr,
                            int *destroyed, void **trace, uptr trace_size) {
  const ReportDesc *rep = (ReportDesc *)report;
  CHECK_LT(idx, rep->mutexes.Size());
  ReportMutex *mutex = rep->mutexes[idx];
  *mutex_id = mutex->id;
  *addr = (void *)mutex->addr;
  *destroyed = mutex->destroyed;
  if (mutex->stack) CopyTrace(mutex->stack->frames, trace, trace_size);
  return 1;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_thread(void *report, uptr idx, int *tid, uptr *os_id,
                             int *running, const char **name, int *parent_tid,
                             void **trace, uptr trace_size) {
  const ReportDesc *rep = (ReportDesc *)report;
  CHECK_LT(idx, rep->threads.Size());
  ReportThread *thread = rep->threads[idx];
  *tid = thread->id;
  *os_id = thread->os_id;
  *running = thread->running;
  *name = thread->name;
  *parent_tid = thread->parent_tid;
  if (thread->stack) CopyTrace(thread->stack->frames, trace, trace_size);
  return 1;
}

SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_unique_tid(void *report, uptr idx, int *tid) {
  const ReportDesc *rep = (ReportDesc *)report;
  CHECK_LT(idx, rep->unique_tids.Size());
  *tid = rep->unique_tids[idx];
  return 1;
}
