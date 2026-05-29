//===-- tsan_report.cpp ---------------------------------------------------===//
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
#include "tsan_report.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"

namespace __tsan {

class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Access()     { return Blue(); }
  const char *ThreadDescription()    { return Cyan(); }
  const char *Location()   { return Green(); }
  const char *Sleep()   { return Yellow(); }
  const char *Mutex()   { return Magenta(); }
};

ReportDesc::ReportDesc()
    : tag(kExternalTagNone)
    , stacks()
    , mops()
    , locs()
    , mutexes()
    , threads()
    , unique_tids()
    , sleep()
    , count() {
}

ReportMop::ReportMop()
    : mset() {
}

ReportDesc::~ReportDesc() {
  // FIXME(dvyukov): it must be leaking a lot of memory.
}

#if !SANITIZER_GO

const int kThreadBufSize = 32;
const char *thread_name(char *buf, Tid tid) {
  if (tid == kMainTid)
    return "main thread";
  internal_snprintf(buf, kThreadBufSize, "thread T%d", tid);
  return buf;
}

static const char *ReportTypeString(ReportType typ, uptr tag) {
  switch (typ) {
    case ReportTypeRace:
      return "data race";
    case ReportTypeVptrRace:
      return "data race on vptr (ctor/dtor vs virtual call)";
    case ReportTypeUseAfterFree:
      return "heap-use-after-free";
    case ReportTypeVptrUseAfterFree:
      return "heap-use-after-free (virtual call vs free)";
    case ReportTypeExternalRace: {
      const char *str = GetReportHeaderFromTag(tag);
      return str ? str : "race on external object";
    }
    case ReportTypeThreadLeak:
      return "thread leak";
    case ReportTypeMutexDestroyLocked:
      return "destroy of a locked mutex";
    case ReportTypeMutexDoubleLock:
      return "double lock of a mutex";
    case ReportTypeMutexInvalidAccess:
      return "use of an invalid mutex (e.g. uninitialized or destroyed)";
    case ReportTypeMutexBadUnlock:
      return "unlock of an unlocked mutex (or by a wrong thread)";
    case ReportTypeMutexBadReadLock:
      return "read lock of a write locked mutex";
    case ReportTypeMutexBadReadUnlock:
      return "read unlock of a write locked mutex";
    case ReportTypeSignalUnsafe:
      return "signal-unsafe call inside of a signal";
    case ReportTypeErrnoInSignal:
      return "signal handler spoils errno";
    case ReportTypeDeadlock:
      return "lock-order-inversion (potential deadlock)";
    case ReportTypeMutexHeldWrongContext:
      return "mutex held in the wrong context";
      // No default case so compiler warns us if we miss one
  }
  UNREACHABLE("missing case");
}

void PrintStack(const ReportStack *ent) {
  if (ent == 0 || ent->frames == 0) {
    Printf("    [failed to restore the stack]\n\n");
    return;
  }
  SymbolizedStack *frame = ent->frames;
  for (int i = 0; frame && frame->info.address; frame = frame->next, i++) {
    InternalScopedString res;
    StackTracePrinter::GetOrInit()->RenderFrame(
        &res, common_flags()->stack_trace_format, i, frame->info.address,
        &frame->info, common_flags()->symbolize_vs_style,
        common_flags()->strip_path_prefix);
    Printf("%s\n", res.data());
  }
  Printf("\n");
}

static void PrintMutexSet(Vector<ReportMopMutex> const& mset) {
  for (uptr i = 0; i < mset.Size(); i++) {
    if (i == 0)
      Printf(" (mutexes:");
    const ReportMopMutex m = mset[i];
    Printf(" %s M%u", m.write ? "write" : "read", m.id);
    Printf(i == mset.Size() - 1 ? ")" : ",");
  }
}

static const char *MopDesc(bool first, bool write, bool atomic) {
  return atomic ? (first ? (write ? "Atomic write" : "Atomic read")
                : (write ? "Previous atomic write" : "Previous atomic read"))
                : (first ? (write ? "Write" : "Read")
                : (write ? "Previous write" : "Previous read"));
}

static const char *ExternalMopDesc(bool first, bool write) {
  return first ? (write ? "Modifying" : "Read-only")
               : (write ? "Previous modifying" : "Previous read-only");
}

#  define Output(...)                                            \
    if (out) {                                                   \
      int written = internal_snprintf(out, outlen, __VA_ARGS__); \
      out += written;                                            \
      outlen -= written;                                         \
    } else {                                                     \
      Printf(__VA_ARGS__);                                       \
    }

static void PrintMop(const ReportMop* mop, bool first, char* out = NULL,
                     size_t outlen = 0) {
  Decorator d;
  char thrbuf[kThreadBufSize];

  if (!out)
    Printf("%s", d.Access());

  if (mop->external_tag == kExternalTagNone) {
    Output("  %s of size %d at %p by %s",
           MopDesc(first, mop->write, mop->atomic), mop->size, (void*)mop->addr,
           thread_name(thrbuf, mop->tid));
  } else {
    const char *object_type = GetObjectTypeFromTag(mop->external_tag);
    if (object_type == nullptr)
        object_type = "external object";
    Output("  %s access of %s at %p by %s", ExternalMopDesc(first, mop->write),
           object_type, (void*)mop->addr, thread_name(thrbuf, mop->tid));
  }

  if (!out) {
    PrintMutexSet(mop->mset);
    Printf(":\n");
    Printf("%s", d.Default());
    PrintStack(mop->stack);
  }
}

static bool PrintLocation(const ReportLocation* loc, char* out = NULL,
                          size_t outlen = 0) {
  Decorator d;
  char thrbuf[kThreadBufSize];
  bool print_stack = false;

  if (!out)
    Printf("%s", d.Location());

  if (loc->type == ReportLocationGlobal) {
    const DataInfo &global = loc->global;
    if (global.size != 0) {
      Output("  Location is global '%s' of size %zu at %p (%s+0x%zx)\n\n",
             global.name, global.size, reinterpret_cast<void*>(global.start),
             StripModuleName(global.module), global.module_offset);
    } else {
      Output("  Location is global '%s' at %p (%s+0x%zx)\n\n", global.name,
             reinterpret_cast<void*>(global.start),
             StripModuleName(global.module), global.module_offset);
    }
  } else if (loc->type == ReportLocationHeap) {
    char thrbuf[kThreadBufSize];
    const char *object_type = GetObjectTypeFromTag(loc->external_tag);
    if (!object_type) {
      Output("  Location is heap block of size %zu at %p allocated by %s",
             loc->heap_chunk_size,
             reinterpret_cast<void*>(loc->heap_chunk_start),
             thread_name(thrbuf, loc->tid));
    } else {
      Output("  Location is %s of size %zu at %p allocated by %s", object_type,
             loc->heap_chunk_size,
             reinterpret_cast<void*>(loc->heap_chunk_start),
             thread_name(thrbuf, loc->tid));
    }
    if (!out)
      Printf(":\n");
    print_stack = true;
  } else if (loc->type == ReportLocationStack) {
    Output("  Location is stack of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationTLS) {
    Output("  Location is TLS of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationFD) {
    Output("  Location is file descriptor %d %s by %s at:", loc->fd,
           loc->fd_closed ? "destroyed" : "created",
           thread_name(thrbuf, loc->tid));
    if (!out)
      Printf(":\n");
    print_stack = true;
  }

  if (!out) {
    Output("%s", d.Default());
    if (print_stack)
      PrintStack(loc->stack);
  }

  return print_stack;
}

static void PrintMutexShort(const ReportMutex *rm, const char *after) {
  Decorator d;
  Printf("%sM%d%s%s", d.Mutex(), rm->id, d.Default(), after);
}

static void PrintMutexShortWithAddress(const ReportMutex *rm,
                                       const char *after) {
  Decorator d;
  Printf("%sM%d (%p)%s%s", d.Mutex(), rm->id,
         reinterpret_cast<void *>(rm->addr), d.Default(), after);
}

static void PrintMutex(const ReportMutex* rm, char* out = NULL,
                       size_t outlen = 0) {
  Decorator d;
  if (!out)
    Printf("%s", d.Mutex());
  Output("  Mutex M%u (%p) created at", rm->id,
         reinterpret_cast<void*>(rm->addr));
  if (!out) {
    Printf(":\n");
    Printf("%s", d.Default());
    PrintStack(rm->stack);
  }
}

// Returns whether the stack is important enough to be printed
static bool PrintThread(const ReportThread* rt, char* out = NULL,
                        size_t outlen = 0) {
  Decorator d;
  if (rt->id == kMainTid)  // Little sense in describing the main thread.
    return false;
  if (!out)
    Printf("%s", d.ThreadDescription());
  Output("  Thread T%d", rt->id);
  if (rt->name && rt->name[0] != '\0') {
    Output(" '%s'", rt->name);
  }
  char thrbuf[kThreadBufSize];
  const char *thread_status = rt->running ? "running" : "finished";
  if (rt->thread_type == ThreadType::Worker) {
    Output(" (tid=%llu, %s) is a GCD worker thread", rt->os_id, thread_status);
    if (!out) {
      Printf("\n");
      Printf("\n");
      Printf("%s", d.Default());
    }
    return false;
  }
  Output(" (tid=%llu, %s) created by %s", rt->os_id, thread_status,
         thread_name(thrbuf, rt->parent_tid));
  if (!out) {
    if (rt->stack)
      Printf(" at:");
    Printf("\n");
    Printf("%s", d.Default());
    PrintStack(rt->stack);
  }
  return true;
}

static void PrintSleep(const ReportStack *s) {
  Decorator d;
  Printf("%s", d.Sleep());
  Printf("  As if synchronized via sleep:\n");
  Printf("%s", d.Default());
  PrintStack(s);
}

static ReportStack *ChooseSummaryStack(const ReportDesc *rep) {
  if (rep->mops.Size())
    return rep->mops[0]->stack;
  if (rep->stacks.Size())
    return rep->stacks[0];
  if (rep->mutexes.Size())
    return rep->mutexes[0]->stack;
  if (rep->threads.Size())
    return rep->threads[0]->stack;
  return 0;
}

static const SymbolizedStack *SkipTsanInternalFrames(SymbolizedStack *frames) {
  if (const SymbolizedStack *f = SkipInternalFrames(frames))
    return f;
  return frames;  // Fallback to the top frame.
}

void PrintReport(const ReportDesc *rep) {
  Decorator d;
  Printf("==================\n");
  const char *rep_typ_str = ReportTypeString(rep->typ, rep->tag);
  Printf("%s", d.Warning());
  Printf("WARNING: ThreadSanitizer: %s (pid=%d)\n", rep_typ_str,
         (int)internal_getpid());
  Printf("%s", d.Default());

  if (rep->typ == ReportTypeErrnoInSignal)
    Printf("  Signal %u handler invoked at:\n", rep->signum);

  if (rep->typ == ReportTypeDeadlock) {
    char thrbuf[kThreadBufSize];
    Printf("  Cycle in lock order graph: ");
    for (uptr i = 0; i < rep->mutexes.Size(); i++)
      PrintMutexShortWithAddress(rep->mutexes[i], " => ");
    PrintMutexShort(rep->mutexes[0], "\n\n");
    CHECK_GT(rep->mutexes.Size(), 0U);
    CHECK_EQ(rep->mutexes.Size() * (flags()->second_deadlock_stack ? 2 : 1),
             rep->stacks.Size());
    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      Printf("  Mutex ");
      PrintMutexShort(rep->mutexes[(i + 1) % rep->mutexes.Size()],
                      " acquired here while holding mutex ");
      PrintMutexShort(rep->mutexes[i], " in ");
      Printf("%s", d.ThreadDescription());
      Printf("%s:\n", thread_name(thrbuf, rep->unique_tids[i]));
      Printf("%s", d.Default());
      if (flags()->second_deadlock_stack) {
        PrintStack(rep->stacks[2*i]);
        Printf("  Mutex ");
        PrintMutexShort(rep->mutexes[i],
                        " previously acquired by the same thread here:\n");
        PrintStack(rep->stacks[2*i+1]);
      } else {
        PrintStack(rep->stacks[i]);
        if (i == 0)
          Printf(
              "    HINT: use TSAN_OPTIONS=second_deadlock_stack=1 "
              "to get more informative warning message\n\n");
      }
    }
  } else {
    for (uptr i = 0; i < rep->stacks.Size(); i++) {
      if (i)
        Printf("  and:\n");
      PrintStack(rep->stacks[i]);
    }
  }

  for (uptr i = 0; i < rep->mops.Size(); i++)
    PrintMop(rep->mops[i], i == 0);

  if (rep->sleep)
    PrintSleep(rep->sleep);

  for (uptr i = 0; i < rep->locs.Size(); i++)
    PrintLocation(rep->locs[i]);

  if (rep->typ != ReportTypeDeadlock) {
    for (uptr i = 0; i < rep->mutexes.Size(); i++)
      PrintMutex(rep->mutexes[i]);
  }

  for (uptr i = 0; i < rep->threads.Size(); i++)
    PrintThread(rep->threads[i]);

  if (rep->typ == ReportTypeThreadLeak && rep->count > 1)
    Printf("  And %d more similar thread leaks.\n\n", rep->count - 1);

  if (ReportStack *stack = ChooseSummaryStack(rep)) {
    if (const SymbolizedStack *frame = SkipTsanInternalFrames(stack->frames))
      ReportErrorSummary(rep_typ_str, frame->info);
  }

  if (common_flags()->print_module_map == 2)
    DumpProcessMap();

  Printf("==================\n");
#  if SANITIZER_APPLE
  if (common_flags()->crashreporter_global) {
    InternalMmapVector<llvm_sanitizer_report_payload_stack_v1> stacks;
    InternalMmapVector<char> descbuf(
        LLVM_SANITIZER_V1_STACK_DESCRIPTION_MAXLEN);
    stacks.reserve(4);
    uptr fault_address = 0, allocation_addr = 0, allocation_size = 0;

    for (uptr i = 0; i < rep->mops.Size(); i++) {
      auto mop = rep->mops[i];
      PrintMop(mop, i == 0, descbuf.data(), descbuf.size());
      GetDarwinStack(stacks, mop->tid, &mop->stack_trace,
                     LLVM_SANITIZER_V1_STACK_TYPE_OTHER, descbuf.data(),
                     /* includeThreadName */ false);
      fault_address = mop->addr;
    }

    for (uptr i = 0; i < rep->locs.Size(); i++) {
      auto loc = rep->locs[i];
      bool print_stack = PrintLocation(loc, descbuf.data(), descbuf.size());
      if (print_stack && loc->stack)
        GetDarwinStack(stacks, loc->tid, &loc->stack->raw,
                       LLVM_SANITIZER_V1_STACK_TYPE_OTHER, descbuf.data(),
                       /* includeThreadName */ false);

      if (loc->type == ReportLocationHeap) {
        allocation_addr = loc->heap_chunk_start;
        allocation_size = loc->heap_chunk_size;
      }
    }

    for (uptr i = 0; i < rep->threads.Size(); i++) {
      ReportThread* rt = rep->threads[i];
      bool print_stack = PrintThread(rt, descbuf.data(), descbuf.size());
      if (print_stack && rt->stack)
        GetDarwinStack(stacks, rt->parent_tid, &rt->stack->raw,
                       LLVM_SANITIZER_V1_STACK_TYPE_OTHER, descbuf.data(),
                       /* includeThreadName */ false);
    }

    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      ReportMutex* rm = rep->mutexes[i];
      if (!rm->stack)
        continue;
      PrintMutex(rm, descbuf.data(), descbuf.size());
      GetDarwinStack(stacks, kInvalidTid, &rm->stack->raw,
                     LLVM_SANITIZER_V1_STACK_TYPE_OTHER, descbuf.data(),
                     /* includeThreadName */ false);
    }

    SetCrashReporterGlobalForReport(rep_typ_str, fault_address, allocation_addr,
                                    allocation_size, stacks);
  }
#  endif /* SANITIZER_APPLE */
}

#else  // #if !SANITIZER_GO

const Tid kMainGoroutineId = 1;

void PrintStack(const ReportStack *ent) {
  if (ent == 0 || ent->frames == 0) {
    Printf("  [failed to restore the stack]\n");
    return;
  }
  SymbolizedStack *frame = ent->frames;
  for (int i = 0; frame; frame = frame->next, i++) {
    const AddressInfo &info = frame->info;
    Printf("  %s()\n      %s:%d +0x%zx\n", info.function,
           StripPathPrefix(info.file, common_flags()->strip_path_prefix),
           info.line, info.module_offset);
  }
}

static void PrintMop(const ReportMop *mop, bool first) {
  Printf("\n");
  Printf("%s at %p by ",
         (first ? (mop->write ? "Write" : "Read")
                : (mop->write ? "Previous write" : "Previous read")),
         reinterpret_cast<void *>(mop->addr));
  if (mop->tid == kMainGoroutineId)
    Printf("main goroutine:\n");
  else
    Printf("goroutine %d:\n", mop->tid);
  PrintStack(mop->stack);
}

static void PrintLocation(const ReportLocation *loc) {
  switch (loc->type) {
  case ReportLocationHeap: {
    Printf("\n");
    Printf("Heap block of size %zu at %p allocated by ", loc->heap_chunk_size,
           reinterpret_cast<void *>(loc->heap_chunk_start));
    if (loc->tid == kMainGoroutineId)
      Printf("main goroutine:\n");
    else
      Printf("goroutine %d:\n", loc->tid);
    PrintStack(loc->stack);
    break;
  }
  case ReportLocationGlobal: {
    Printf("\n");
    Printf("Global var %s of size %zu at %p declared at %s:%zu\n",
           loc->global.name, loc->global.size,
           reinterpret_cast<void *>(loc->global.start), loc->global.file,
           loc->global.line);
    break;
  }
  default:
    break;
  }
}

static void PrintThread(const ReportThread *rt) {
  if (rt->id == kMainGoroutineId)
    return;
  Printf("\n");
  Printf("Goroutine %d (%s) created at:\n",
    rt->id, rt->running ? "running" : "finished");
  PrintStack(rt->stack);
}

void PrintReport(const ReportDesc *rep) {
  Printf("==================\n");
  if (rep->typ == ReportTypeRace) {
    Printf("WARNING: DATA RACE");
    for (uptr i = 0; i < rep->mops.Size(); i++)
      PrintMop(rep->mops[i], i == 0);
    for (uptr i = 0; i < rep->locs.Size(); i++)
      PrintLocation(rep->locs[i]);
    for (uptr i = 0; i < rep->threads.Size(); i++)
      PrintThread(rep->threads[i]);
  } else if (rep->typ == ReportTypeDeadlock) {
    Printf("WARNING: DEADLOCK\n");
    for (uptr i = 0; i < rep->mutexes.Size(); i++) {
      Printf("Goroutine %d lock mutex %u while holding mutex %u:\n", 999,
             rep->mutexes[i]->id,
             rep->mutexes[(i + 1) % rep->mutexes.Size()]->id);
      PrintStack(rep->stacks[2*i]);
      Printf("\n");
      Printf("Mutex %u was previously locked here:\n",
             rep->mutexes[(i + 1) % rep->mutexes.Size()]->id);
      PrintStack(rep->stacks[2*i + 1]);
      Printf("\n");
    }
  }
  Printf("==================\n");
}

#endif

}  // namespace __tsan
