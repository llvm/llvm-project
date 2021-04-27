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
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "tsan_fd.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"

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

void* RegionAlloc::Alloc(uptr size) {
  void* obj = __tsan::Alloc(size);
  objects_.PushBack({obj, FreeImpl});
  return obj;
}

char* RegionAlloc::Strdup(const char* str) {
  uptr len = internal_strlen(str);
  char* s2 = static_cast<char*>(Alloc(len + 1));
  internal_memcpy(s2, str, len);
  s2[len] = 0;
  return s2;
}

RegionAlloc::~RegionAlloc() {
  for (uptr i = 0; i < objects_.Size(); i++) {
    objects_[i].fn(objects_[i].obj);
  }
}

void ReportDesc::AddMemoryAccess(uptr addr, uptr external_tag, Shadow s,
                                 Tid tid, StackTrace stack,
                                 const MutexSet* mset) {
  auto mop = *mops.PushBack(region.Alloc<ReportMop>());
  mop->tid = tid;
  mop->addr = addr + s.addr0();
  mop->size = s.size();
  mop->write = s.IsWrite();
  mop->atomic = s.IsAtomic();
  mop->stack.stack = stack;
  mop->stack.suppressable = true;
  mop->external_tag = external_tag;
  for (uptr i = 0; i < mset->Size(); i++) {
    MutexSet::Desc d = mset->Get(i);
    int id = this->AddMutex(d.addr, d.stack_id);
    ReportMopMutex mtx = {id, d.write};
    mop->mset.PushBack(mtx);
  }
}

void ReportDesc::AddThread(const ThreadContext* tctx, bool suppressable) {
  for (uptr i = 0; i < threads.Size(); i++) {
    if (threads[i]->id == tctx->tid)
      return;
  }
  auto rt = *threads.PushBack(region.Alloc<ReportThread>());
  rt->id = tctx->tid;
  rt->os_id = tctx->os_id;
  rt->running = (tctx->status == ThreadStatusRunning);
  rt->name = region.Strdup(tctx->name);
  rt->parent_tid = tctx->parent_tid;
  rt->thread_type = tctx->thread_type;
  rt->stack.stack = StackDepotGet(tctx->creation_stack_id);
  rt->stack.suppressable = suppressable;
}

#if !SANITIZER_GO
static ThreadContext* FindThreadByTidLocked(Tid tid) {
  return static_cast<ThreadContext*>(ctx->thread_registry.GetThreadLocked(tid));
}
#endif

void ReportDesc::AddThread(Tid tid, bool suppressable) {
#if !SANITIZER_GO
  if (const ThreadContext* tctx = FindThreadByTidLocked(tid))
    AddThread(tctx, suppressable);
#endif
}

int ReportDesc::AddMutex(uptr addr, StackID creation_stack_id) {
  for (uptr i = 0; i < mutexes.Size(); i++) {
    auto rm = mutexes[i];
    if (rm->addr == addr)
      return rm->id;
  }
  auto rm = *mutexes.PushBack(region.Alloc<ReportMutex>());
  rm->id = mutexes.Size() - 1;
  rm->addr = addr;
  rm->stack.stack = StackDepotGet(creation_stack_id);
  return rm->id;
}

void ReportDesc::AddLocation(uptr addr, uptr size) {
  if (addr == 0)
    return;
#if !SANITIZER_GO
  int fd = -1;
  Tid create_tid = kInvalidTid;
  StackID create_stack = kInvalidStackID;
  if (FdLocation(addr, &fd, &create_tid, &create_stack)) {
    auto loc = *locs.PushBack(region.Alloc<ReportLocation>());
    loc->type = ReportLocationFD;
    loc->fd = fd;
    loc->tid = create_tid;
    loc->stack.stack = StackDepotGet(create_stack);
    AddThread(create_tid);
    return;
  }
  //!!! check if this is Java heap and find block start
  Allocator* a = allocator();
  if (a->PointerIsMine((void*)addr)) {
    void* block_begin = a->GetBlockBegin((void*)addr);
    if (block_begin) {
      if (MBlock* b = ctx->metamap.GetBlock((uptr)block_begin))
        AddHeapLocation((uptr)block_begin, b);
    }
    return;
  }
  uptr block_begin = 0;
  if (MBlock* b = JavaHeapBlock(addr, &block_begin)) {
    AddHeapLocation(block_begin, b);
    return;
  }
  bool is_stack = false;
  if (ThreadContext* tctx = IsThreadStackOrTls(addr, &is_stack)) {
    auto loc = *locs.PushBack(region.Alloc<ReportLocation>());
    loc->type = is_stack ? ReportLocationStack : ReportLocationTLS;
    loc->tid = tctx->tid;
    AddThread(tctx);
    return;
  }
#endif
  // We will try to symbolize this as a global later.
  auto loc = *locs.PushBack(region.Alloc<ReportLocation>());
  loc->type = ReportLocationInvalid;
  loc->suppressable = true;
  loc->global.start = addr;
}

#if !SANITIZER_GO
void ReportDesc::AddHeapLocation(uptr addr, MBlock* b) {
  auto loc = *locs.PushBack(region.Alloc<ReportLocation>());
  loc->type = ReportLocationHeap;
  loc->heap_chunk_start = addr;
  loc->heap_chunk_size = b->siz;
  loc->external_tag = b->tag;
  loc->tid = b->tid;
  loc->stack.stack = StackDepotGet(b->stk);
  if (ThreadContext* tctx = FindThreadByTidLocked(b->tid))
    AddThread(tctx);
}
#endif

void ReportDesc::AddStack(StackTrace stack, bool suppressable) {
  auto rs = *stacks.PushBack(region.Alloc<ReportStack>());
  rs->stack = stack;
  rs->suppressable = suppressable;
}

void ReportDesc::AddUniqueTid(Tid unique_tid) {
  unique_tids.PushBack(unique_tid);
}

#if !SANITIZER_GO
void ReportDesc::AddSleep(StackID stack_id) {
  sleep = region.Alloc<ReportStack>();
  sleep->stack = StackDepotGet(stack_id);
}
#endif

#if !SANITIZER_GO

const int kThreadBufSize = 32;
const char* thread_name(char* buf, Tid tid) {
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
    // No default case so compiler warns us if we miss one
  }
  UNREACHABLE("missing case");
}

void PrintStack(const SymbolizedStack* frame) {
  if (frame == nullptr) {
    Printf("    [failed to restore the stack]\n\n");
    return;
  }
  const char* const kInterposedFunctionPrefix =
      SANITIZER_MAC ? "wrap_" : "__interceptor_";
  for (int i = 0; frame && frame->info.address; frame = frame->next, i++) {
    InternalScopedString res;
    RenderFrame(&res, common_flags()->stack_trace_format, i,
                frame->info.address, &frame->info,
                common_flags()->symbolize_vs_style,
                common_flags()->strip_path_prefix, kInterposedFunctionPrefix);
    Printf("%s\n", res.data());
  }
  Printf("\n");
}

static void PrintMutexSet(Vector<ReportMopMutex> const& mset) {
  for (uptr i = 0; i < mset.Size(); i++) {
    if (i == 0)
      Printf(" (mutexes:");
    const ReportMopMutex m = mset[i];
    Printf(" %s M%d", m.write ? "write" : "read", m.id);
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

static void PrintMop(const ReportMop *mop, bool first) {
  Decorator d;
  char thrbuf[kThreadBufSize];
  Printf("%s", d.Access());
  if (mop->external_tag == kExternalTagNone) {
    Printf("  %s of size %d at %p by %s",
           MopDesc(first, mop->write, mop->atomic), mop->size,
           (void *)mop->addr, thread_name(thrbuf, mop->tid));
  } else {
    const char *object_type = GetObjectTypeFromTag(mop->external_tag);
    if (object_type == nullptr)
        object_type = "external object";
    Printf("  %s access of %s at %p by %s",
           ExternalMopDesc(first, mop->write), object_type,
           (void *)mop->addr, thread_name(thrbuf, mop->tid));
  }
  PrintMutexSet(mop->mset);
  Printf(":\n");
  Printf("%s", d.Default());
  PrintStack(mop->stack.frames);
}

static void PrintLocation(const ReportLocation *loc) {
  Decorator d;
  char thrbuf[kThreadBufSize];
  bool print_stack = false;
  Printf("%s", d.Location());
  if (loc->type == ReportLocationGlobal) {
    const DataInfo &global = loc->global;
    if (global.size != 0)
      Printf("  Location is global '%s' of size %zu at %p (%s+%p)\n\n",
             global.name, global.size, global.start,
             StripModuleName(global.module), global.module_offset);
    else
      Printf("  Location is global '%s' at %p (%s+%p)\n\n", global.name,
             global.start, StripModuleName(global.module),
             global.module_offset);
  } else if (loc->type == ReportLocationHeap) {
    char thrbuf[kThreadBufSize];
    const char *object_type = GetObjectTypeFromTag(loc->external_tag);
    if (!object_type) {
      Printf("  Location is heap block of size %zu at %p allocated by %s:\n",
             loc->heap_chunk_size, loc->heap_chunk_start,
             thread_name(thrbuf, loc->tid));
    } else {
      Printf("  Location is %s of size %zu at %p allocated by %s:\n",
             object_type, loc->heap_chunk_size, loc->heap_chunk_start,
             thread_name(thrbuf, loc->tid));
    }
    print_stack = true;
  } else if (loc->type == ReportLocationStack) {
    Printf("  Location is stack of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationTLS) {
    Printf("  Location is TLS of %s.\n\n", thread_name(thrbuf, loc->tid));
  } else if (loc->type == ReportLocationFD) {
    Printf("  Location is file descriptor %d created by %s at:\n",
        loc->fd, thread_name(thrbuf, loc->tid));
    print_stack = true;
  }
  Printf("%s", d.Default());
  if (print_stack)
    PrintStack(loc->stack.frames);
}

static void PrintMutexShort(const ReportMutex *rm, const char *after) {
  Decorator d;
  Printf("%sM%d%s%s", d.Mutex(), rm->id, d.Default(), after);
}

static void PrintMutexShortWithAddress(const ReportMutex *rm,
                                       const char *after) {
  Decorator d;
  Printf("%sM%d (%p)%s%s", d.Mutex(), rm->id, rm->addr, d.Default(), after);
}

static void PrintMutex(const ReportMutex *rm) {
  Decorator d;
  Printf("%s", d.Mutex());
  Printf("  Mutex M%d (%p) created at:\n", rm->id, rm->addr);
  Printf("%s", d.Default());
  PrintStack(rm->stack.frames);
}

static void PrintThread(const ReportThread *rt) {
  Decorator d;
  if (rt->id == kMainTid)  // Little sense in describing the main thread.
    return;
  Printf("%s", d.ThreadDescription());
  Printf("  Thread T%d", rt->id);
  if (rt->name && rt->name[0] != '\0')
    Printf(" '%s'", rt->name);
  char thrbuf[kThreadBufSize];
  const char *thread_status = rt->running ? "running" : "finished";
  if (rt->thread_type == ThreadType::Worker) {
    Printf(" (tid=%zu, %s) is a GCD worker thread\n", rt->os_id, thread_status);
    Printf("\n");
    Printf("%s", d.Default());
    return;
  }
  Printf(" (tid=%zu, %s) created by %s", rt->os_id, thread_status,
         thread_name(thrbuf, rt->parent_tid));
  if (rt->stack.frames)
    Printf(" at:");
  Printf("\n");
  Printf("%s", d.Default());
  if (rt->stack.frames)
    PrintStack(rt->stack.frames);
}

static void PrintSleep(const ReportStack *s) {
  Decorator d;
  Printf("%s", d.Sleep());
  Printf("  As if synchronized via sleep:\n");
  Printf("%s", d.Default());
  PrintStack(s->frames);
}

static ReportStack *ChooseSummaryStack(const ReportDesc *rep) {
  if (rep->mops.Size())
    return &rep->mops[0]->stack;
  if (rep->stacks.Size())
    return rep->stacks[0];
  if (rep->mutexes.Size())
    return &rep->mutexes[0]->stack;
  if (rep->threads.Size())
    return &rep->threads[0]->stack;
  return nullptr;
}

static bool FrameIsInternal(const SymbolizedStack *frame) {
  if (frame == 0)
    return false;
  const char *file = frame->info.file;
  const char *module = frame->info.module;
  if (file != 0 &&
      (internal_strstr(file, "tsan_interceptors_posix.cpp") ||
       internal_strstr(file, "sanitizer_common_interceptors.inc") ||
       internal_strstr(file, "tsan_interface_")))
    return true;
  if (module != 0 && (internal_strstr(module, "libclang_rt.tsan_")))
    return true;
  return false;
}

static SymbolizedStack *SkipTsanInternalFrames(SymbolizedStack *frames) {
  while (FrameIsInternal(frames) && frames->next)
    frames = frames->next;
  return frames;
}

void PrintReport(const ReportDesc *rep) {
  Decorator d;
  Printf("==================\n");
  const char *rep_typ_str = ReportTypeString(rep->typ, rep->tag);
  Printf("%s", d.Warning());
  Printf("WARNING: ThreadSanitizer: %s (pid=%d)\n", rep_typ_str,
         (int)internal_getpid());
  Printf("%s", d.Default());

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
        PrintStack(rep->stacks[2 * i]->frames);
        Printf("  Mutex ");
        PrintMutexShort(rep->mutexes[i],
                        " previously acquired by the same thread here:\n");
        PrintStack(rep->stacks[2 * i + 1]->frames);
      } else {
        PrintStack(rep->stacks[i]->frames);
        if (i == 0)
          Printf("    Hint: use TSAN_OPTIONS=second_deadlock_stack=1 "
                 "to get more informative warning message\n\n");
      }
    }
  } else {
    for (uptr i = 0; i < rep->stacks.Size(); i++) {
      if (i)
        Printf("  and:\n");
      PrintStack(rep->stacks[i]->frames);
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
    if (SymbolizedStack* frame = SkipTsanInternalFrames(stack->frames))
      ReportErrorSummary(rep_typ_str, frame->info);
  }

  if (common_flags()->print_module_map == 2)
    DumpProcessMap();

  Printf("==================\n");
}

#else  // #if !SANITIZER_GO

const Tid kMainGoroutineId = static_cast<Tid>(1);

void PrintStack(const SymbolizedStack* frame) {
  if (frame == nullptr) {
    Printf("  [failed to restore the stack]\n");
    return;
  }
  for (int i = 0; frame; frame = frame->next, i++) {
    const AddressInfo &info = frame->info;
    Printf("  %s()\n      %s:%d +0x%zx\n", info.function,
        StripPathPrefix(info.file, common_flags()->strip_path_prefix),
        info.line, (void *)info.module_offset);
  }
}

static void PrintMop(const ReportMop *mop, bool first) {
  Printf("\n");
  Printf("%s at %p by ",
      (first ? (mop->write ? "Write" : "Read")
             : (mop->write ? "Previous write" : "Previous read")), mop->addr);
  if (mop->tid == kMainGoroutineId)
    Printf("main goroutine:\n");
  else
    Printf("goroutine %d:\n", mop->tid);
  PrintStack(mop->stack.frames);
}

static void PrintLocation(const ReportLocation *loc) {
  switch (loc->type) {
  case ReportLocationHeap: {
    Printf("\n");
    Printf("Heap block of size %zu at %p allocated by ",
        loc->heap_chunk_size, loc->heap_chunk_start);
    if (loc->tid == kMainGoroutineId)
      Printf("main goroutine:\n");
    else
      Printf("goroutine %d:\n", loc->tid);
    PrintStack(loc->stack.frames);
    break;
  }
  case ReportLocationGlobal: {
    Printf("\n");
    Printf("Global var %s of size %zu at %p declared at %s:%zu\n",
        loc->global.name, loc->global.size, loc->global.start,
        loc->global.file, loc->global.line);
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
  PrintStack(rt->stack.frames);
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
      uptr next = (i + 1) % rep->mutexes.Size();
      Printf("Goroutine %d lock mutex %d while holding mutex %d:\n", 999, i,
             next);
      PrintStack(rep->stacks[2 * i]->frames);
      Printf("\n");
      Printf("Mutex %d was previously locked here:\n", next);
      PrintStack(rep->stacks[2 * i + 1]->frames);
      Printf("\n");
    }
  }
  Printf("==================\n");
}

#endif

}  // namespace __tsan
