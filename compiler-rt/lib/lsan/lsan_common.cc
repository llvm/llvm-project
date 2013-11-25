//=-- lsan_common.cc ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Implementation of common leak checking functionality.
//
//===----------------------------------------------------------------------===//

#include "lsan_common.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stoptheworld.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_report_decorator.h"

#if CAN_SANITIZE_LEAKS
namespace __lsan {

// This mutex is used to prevent races between DoLeakCheck and IgnoreObject.
BlockingMutex global_mutex(LINKER_INITIALIZED);

THREADLOCAL int disable_counter;
bool DisabledInThisThread() { return disable_counter > 0; }

Flags lsan_flags;

static void InitializeFlags() {
  Flags *f = flags();
  // Default values.
  f->report_objects = false;
  f->resolution = 0;
  f->max_leaks = 0;
  f->exitcode = 23;
  f->suppressions="";
  f->use_registers = true;
  f->use_globals = true;
  f->use_stacks = true;
  f->use_tls = true;
  f->use_unaligned = false;
  f->verbosity = 0;
  f->log_pointers = false;
  f->log_threads = false;

  const char *options = GetEnv("LSAN_OPTIONS");
  if (options) {
    ParseFlag(options, &f->use_registers, "use_registers");
    ParseFlag(options, &f->use_globals, "use_globals");
    ParseFlag(options, &f->use_stacks, "use_stacks");
    ParseFlag(options, &f->use_tls, "use_tls");
    ParseFlag(options, &f->use_unaligned, "use_unaligned");
    ParseFlag(options, &f->report_objects, "report_objects");
    ParseFlag(options, &f->resolution, "resolution");
    CHECK_GE(&f->resolution, 0);
    ParseFlag(options, &f->max_leaks, "max_leaks");
    CHECK_GE(&f->max_leaks, 0);
    ParseFlag(options, &f->verbosity, "verbosity");
    ParseFlag(options, &f->log_pointers, "log_pointers");
    ParseFlag(options, &f->log_threads, "log_threads");
    ParseFlag(options, &f->exitcode, "exitcode");
    ParseFlag(options, &f->suppressions, "suppressions");
  }
}

SuppressionContext *suppression_ctx;

void InitializeSuppressions() {
  CHECK(!suppression_ctx);
  ALIGNED(64) static char placeholder_[sizeof(SuppressionContext)];
  suppression_ctx = new(placeholder_) SuppressionContext;
  char *suppressions_from_file;
  uptr buffer_size;
  if (ReadFileToBuffer(flags()->suppressions, &suppressions_from_file,
                       &buffer_size, 1 << 26 /* max_len */))
    suppression_ctx->Parse(suppressions_from_file);
  if (flags()->suppressions[0] && !buffer_size) {
    Printf("LeakSanitizer: failed to read suppressions file '%s'\n",
           flags()->suppressions);
    Die();
  }
  if (&__lsan_default_suppressions)
    suppression_ctx->Parse(__lsan_default_suppressions());
}

void InitCommonLsan() {
  InitializeFlags();
  if (common_flags()->detect_leaks) {
    // Initialization which can fail or print warnings should only be done if
    // LSan is actually enabled.
    InitializeSuppressions();
    InitializePlatformSpecificModules();
  }
}

class Decorator: private __sanitizer::AnsiColorDecorator {
 public:
  Decorator() : __sanitizer::AnsiColorDecorator(PrintsToTtyCached()) { }
  const char *Error() { return Red(); }
  const char *Leak() { return Blue(); }
  const char *End() { return Default(); }
};

static inline bool CanBeAHeapPointer(uptr p) {
  // Since our heap is located in mmap-ed memory, we can assume a sensible lower
  // bound on heap addresses.
  const uptr kMinAddress = 4 * 4096;
  if (p < kMinAddress) return false;
#ifdef __x86_64__
  // Accept only canonical form user-space addresses.
  return ((p >> 47) == 0);
#else
  return true;
#endif
}

// Scans the memory range, looking for byte patterns that point into allocator
// chunks. Marks those chunks with |tag| and adds them to |frontier|.
// There are two usage modes for this function: finding reachable or ignored
// chunks (|tag| = kReachable or kIgnored) and finding indirectly leaked chunks
// (|tag| = kIndirectlyLeaked). In the second case, there's no flood fill,
// so |frontier| = 0.
void ScanRangeForPointers(uptr begin, uptr end,
                          Frontier *frontier,
                          const char *region_type, ChunkTag tag) {
  const uptr alignment = flags()->pointer_alignment();
  if (flags()->log_pointers)
    Report("Scanning %s range %p-%p.\n", region_type, begin, end);
  uptr pp = begin;
  if (pp % alignment)
    pp = pp + alignment - pp % alignment;
  for (; pp + sizeof(void *) <= end; pp += alignment) {  // NOLINT
    void *p = *reinterpret_cast<void **>(pp);
    if (!CanBeAHeapPointer(reinterpret_cast<uptr>(p))) continue;
    uptr chunk = PointsIntoChunk(p);
    if (!chunk) continue;
    // Pointers to self don't count. This matters when tag == kIndirectlyLeaked.
    if (chunk == begin) continue;
    LsanMetadata m(chunk);
    // Reachable beats ignored beats leaked.
    if (m.tag() == kReachable) continue;
    if (m.tag() == kIgnored && tag != kReachable) continue;
    m.set_tag(tag);
    if (flags()->log_pointers)
      Report("%p: found %p pointing into chunk %p-%p of size %zu.\n", pp, p,
             chunk, chunk + m.requested_size(), m.requested_size());
    if (frontier)
      frontier->push_back(chunk);
  }
}

void ForEachExtraStackRangeCb(uptr begin, uptr end, void* arg) {
  Frontier *frontier = reinterpret_cast<Frontier *>(arg);
  ScanRangeForPointers(begin, end, frontier, "FAKE STACK", kReachable);
}

// Scans thread data (stacks and TLS) for heap pointers.
static void ProcessThreads(SuspendedThreadsList const &suspended_threads,
                           Frontier *frontier) {
  InternalScopedBuffer<uptr> registers(SuspendedThreadsList::RegisterCount());
  uptr registers_begin = reinterpret_cast<uptr>(registers.data());
  uptr registers_end = registers_begin + registers.size();
  for (uptr i = 0; i < suspended_threads.thread_count(); i++) {
    uptr os_id = static_cast<uptr>(suspended_threads.GetThreadID(i));
    if (flags()->log_threads) Report("Processing thread %d.\n", os_id);
    uptr stack_begin, stack_end, tls_begin, tls_end, cache_begin, cache_end;
    bool thread_found = GetThreadRangesLocked(os_id, &stack_begin, &stack_end,
                                              &tls_begin, &tls_end,
                                              &cache_begin, &cache_end);
    if (!thread_found) {
      // If a thread can't be found in the thread registry, it's probably in the
      // process of destruction. Log this event and move on.
      if (flags()->log_threads)
        Report("Thread %d not found in registry.\n", os_id);
      continue;
    }
    uptr sp;
    bool have_registers =
        (suspended_threads.GetRegistersAndSP(i, registers.data(), &sp) == 0);
    if (!have_registers) {
      Report("Unable to get registers from thread %d.\n");
      // If unable to get SP, consider the entire stack to be reachable.
      sp = stack_begin;
    }

    if (flags()->use_registers && have_registers)
      ScanRangeForPointers(registers_begin, registers_end, frontier,
                           "REGISTERS", kReachable);

    if (flags()->use_stacks) {
      if (flags()->log_threads)
        Report("Stack at %p-%p, SP = %p.\n", stack_begin, stack_end, sp);
      if (sp < stack_begin || sp >= stack_end) {
        // SP is outside the recorded stack range (e.g. the thread is running a
        // signal handler on alternate stack). Again, consider the entire stack
        // range to be reachable.
        if (flags()->log_threads)
          Report("WARNING: stack pointer not in stack range.\n");
      } else {
        // Shrink the stack range to ignore out-of-scope values.
        stack_begin = sp;
      }
      ScanRangeForPointers(stack_begin, stack_end, frontier, "STACK",
                           kReachable);
      ForEachExtraStackRange(os_id, ForEachExtraStackRangeCb, frontier);
    }

    if (flags()->use_tls) {
      if (flags()->log_threads) Report("TLS at %p-%p.\n", tls_begin, tls_end);
      if (cache_begin == cache_end) {
        ScanRangeForPointers(tls_begin, tls_end, frontier, "TLS", kReachable);
      } else {
        // Because LSan should not be loaded with dlopen(), we can assume
        // that allocator cache will be part of static TLS image.
        CHECK_LE(tls_begin, cache_begin);
        CHECK_GE(tls_end, cache_end);
        if (tls_begin < cache_begin)
          ScanRangeForPointers(tls_begin, cache_begin, frontier, "TLS",
                               kReachable);
        if (tls_end > cache_end)
          ScanRangeForPointers(cache_end, tls_end, frontier, "TLS", kReachable);
      }
    }
  }
}

static void FloodFillTag(Frontier *frontier, ChunkTag tag) {
  while (frontier->size()) {
    uptr next_chunk = frontier->back();
    frontier->pop_back();
    LsanMetadata m(next_chunk);
    ScanRangeForPointers(next_chunk, next_chunk + m.requested_size(), frontier,
                         "HEAP", tag);
  }
}

// ForEachChunk callback. If the chunk is marked as leaked, marks all chunks
// which are reachable from it as indirectly leaked.
static void MarkIndirectlyLeakedCb(uptr chunk, void *arg) {
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (m.allocated() && m.tag() != kReachable) {
    ScanRangeForPointers(chunk, chunk + m.requested_size(),
                         /* frontier */ 0, "HEAP", kIndirectlyLeaked);
  }
}

// ForEachChunk callback. If chunk is marked as ignored, adds its address to
// frontier.
static void CollectIgnoredCb(uptr chunk, void *arg) {
  CHECK(arg);
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (m.allocated() && m.tag() == kIgnored)
    reinterpret_cast<Frontier *>(arg)->push_back(chunk);
}

// Sets the appropriate tag on each chunk.
static void ClassifyAllChunks(SuspendedThreadsList const &suspended_threads) {
  // Holds the flood fill frontier.
  Frontier frontier(GetPageSizeCached());

  if (flags()->use_globals)
    ProcessGlobalRegions(&frontier);
  ProcessThreads(suspended_threads, &frontier);
  FloodFillTag(&frontier, kReachable);
  // The check here is relatively expensive, so we do this in a separate flood
  // fill. That way we can skip the check for chunks that are reachable
  // otherwise.
  if (flags()->log_pointers)
    Report("Processing platform-specific allocations.\n");
  ProcessPlatformSpecificAllocations(&frontier);
  FloodFillTag(&frontier, kReachable);

  if (flags()->log_pointers)
    Report("Scanning ignored chunks.\n");
  CHECK_EQ(0, frontier.size());
  ForEachChunk(CollectIgnoredCb, &frontier);
  FloodFillTag(&frontier, kIgnored);

  // Iterate over leaked chunks and mark those that are reachable from other
  // leaked chunks.
  if (flags()->log_pointers)
    Report("Scanning leaked chunks.\n");
  ForEachChunk(MarkIndirectlyLeakedCb, 0 /* arg */);
}

static void PrintStackTraceById(u32 stack_trace_id) {
  CHECK(stack_trace_id);
  uptr size = 0;
  const uptr *trace = StackDepotGet(stack_trace_id, &size);
  StackTrace::PrintStack(trace, size);
}

// ForEachChunk callback. Aggregates unreachable chunks into a LeakReport.
static void CollectLeaksCb(uptr chunk, void *arg) {
  CHECK(arg);
  LeakReport *leak_report = reinterpret_cast<LeakReport *>(arg);
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (!m.allocated()) return;
  if (m.tag() == kDirectlyLeaked || m.tag() == kIndirectlyLeaked) {
    uptr resolution = flags()->resolution;
    if (resolution > 0) {
      uptr size = 0;
      const uptr *trace = StackDepotGet(m.stack_trace_id(), &size);
      size = Min(size, resolution);
      leak_report->Add(StackDepotPut(trace, size), m.requested_size(), m.tag());
    } else {
      leak_report->Add(m.stack_trace_id(), m.requested_size(), m.tag());
    }
  }
}

// ForEachChunkCallback. Prints addresses of unreachable chunks.
static void PrintLeakedCb(uptr chunk, void *arg) {
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (!m.allocated()) return;
  if (m.tag() == kDirectlyLeaked || m.tag() == kIndirectlyLeaked) {
    Printf("%s leaked %zu byte object at %p.\n",
           m.tag() == kDirectlyLeaked ? "Directly" : "Indirectly",
           m.requested_size(), chunk);
  }
}

static void PrintMatchedSuppressions() {
  InternalMmapVector<Suppression *> matched(1);
  suppression_ctx->GetMatched(&matched);
  if (!matched.size())
    return;
  const char *line = "-----------------------------------------------------";
  Printf("%s\n", line);
  Printf("Suppressions used:\n");
  Printf("  count      bytes template\n");
  for (uptr i = 0; i < matched.size(); i++)
    Printf("%7zu %10zu %s\n", static_cast<uptr>(matched[i]->hit_count),
           matched[i]->weight, matched[i]->templ);
  Printf("%s\n\n", line);
}

static void PrintLeaked() {
  Printf("\n");
  Printf("Reporting individual objects:\n");
  ForEachChunk(PrintLeakedCb, 0 /* arg */);
}

struct DoLeakCheckParam {
  bool success;
  LeakReport leak_report;
};

static void DoLeakCheckCallback(const SuspendedThreadsList &suspended_threads,
                                void *arg) {
  DoLeakCheckParam *param = reinterpret_cast<DoLeakCheckParam *>(arg);
  CHECK(param);
  CHECK(!param->success);
  CHECK(param->leak_report.IsEmpty());
  ClassifyAllChunks(suspended_threads);
  ForEachChunk(CollectLeaksCb, &param->leak_report);
  if (!param->leak_report.IsEmpty() && flags()->report_objects)
    PrintLeaked();
  param->success = true;
}

void DoLeakCheck() {
  EnsureMainThreadIDIsCorrect();
  BlockingMutexLock l(&global_mutex);
  static bool already_done;
  if (already_done) return;
  already_done = true;
  if (&__lsan_is_turned_off && __lsan_is_turned_off())
    return;

  DoLeakCheckParam param;
  param.success = false;
  LockThreadRegistry();
  LockAllocator();
  StopTheWorld(DoLeakCheckCallback, &param);
  UnlockAllocator();
  UnlockThreadRegistry();

  if (!param.success) {
    Report("LeakSanitizer has encountered a fatal error.\n");
    Die();
  }
  uptr have_unsuppressed = param.leak_report.ApplySuppressions();
  if (have_unsuppressed) {
    Decorator d;
    Printf("\n"
           "================================================================="
           "\n");
    Printf("%s", d.Error());
    Report("ERROR: LeakSanitizer: detected memory leaks\n");
    Printf("%s", d.End());
    param.leak_report.PrintLargest(flags()->max_leaks);
  }
  if (have_unsuppressed || (flags()->verbosity >= 1)) {
    PrintMatchedSuppressions();
    param.leak_report.PrintSummary();
  }
  if (have_unsuppressed && flags()->exitcode)
    internal__exit(flags()->exitcode);
}

static Suppression *GetSuppressionForAddr(uptr addr) {
  static const uptr kMaxAddrFrames = 16;
  InternalScopedBuffer<AddressInfo> addr_frames(kMaxAddrFrames);
  for (uptr i = 0; i < kMaxAddrFrames; i++) new (&addr_frames[i]) AddressInfo();
  uptr addr_frames_num = Symbolizer::Get()->SymbolizeCode(
      addr, addr_frames.data(), kMaxAddrFrames);
  for (uptr i = 0; i < addr_frames_num; i++) {
    Suppression* s;
    if (suppression_ctx->Match(addr_frames[i].function, SuppressionLeak, &s) ||
        suppression_ctx->Match(addr_frames[i].file, SuppressionLeak, &s) ||
        suppression_ctx->Match(addr_frames[i].module, SuppressionLeak, &s))
      return s;
  }
  return 0;
}

static Suppression *GetSuppressionForStack(u32 stack_trace_id) {
  uptr size = 0;
  const uptr *trace = StackDepotGet(stack_trace_id, &size);
  for (uptr i = 0; i < size; i++) {
    Suppression *s =
        GetSuppressionForAddr(StackTrace::GetPreviousInstructionPc(trace[i]));
    if (s) return s;
  }
  return 0;
}

///// LeakReport implementation. /////

// A hard limit on the number of distinct leaks, to avoid quadratic complexity
// in LeakReport::Add(). We don't expect to ever see this many leaks in
// real-world applications.
// FIXME: Get rid of this limit by changing the implementation of LeakReport to
// use a hash table.
const uptr kMaxLeaksConsidered = 5000;

void LeakReport::Add(u32 stack_trace_id, uptr leaked_size, ChunkTag tag) {
  CHECK(tag == kDirectlyLeaked || tag == kIndirectlyLeaked);
  bool is_directly_leaked = (tag == kDirectlyLeaked);
  for (uptr i = 0; i < leaks_.size(); i++)
    if (leaks_[i].stack_trace_id == stack_trace_id &&
        leaks_[i].is_directly_leaked == is_directly_leaked) {
      leaks_[i].hit_count++;
      leaks_[i].total_size += leaked_size;
      return;
    }
  if (leaks_.size() == kMaxLeaksConsidered) return;
  Leak leak = { /* hit_count */ 1, leaked_size, stack_trace_id,
                is_directly_leaked, /* is_suppressed */ false };
  leaks_.push_back(leak);
}

static bool LeakComparator(const Leak &leak1, const Leak &leak2) {
  if (leak1.is_directly_leaked == leak2.is_directly_leaked)
    return leak1.total_size > leak2.total_size;
  else
    return leak1.is_directly_leaked;
}

void LeakReport::PrintLargest(uptr num_leaks_to_print) {
  CHECK(leaks_.size() <= kMaxLeaksConsidered);
  Printf("\n");
  if (leaks_.size() == kMaxLeaksConsidered)
    Printf("Too many leaks! Only the first %zu leaks encountered will be "
           "reported.\n",
           kMaxLeaksConsidered);

  uptr unsuppressed_count = 0;
  for (uptr i = 0; i < leaks_.size(); i++)
    if (!leaks_[i].is_suppressed) unsuppressed_count++;
  if (num_leaks_to_print > 0 && num_leaks_to_print < unsuppressed_count)
    Printf("The %zu largest leak(s):\n", num_leaks_to_print);
  InternalSort(&leaks_, leaks_.size(), LeakComparator);
  uptr leaks_printed = 0;
  Decorator d;
  for (uptr i = 0; i < leaks_.size(); i++) {
    if (leaks_[i].is_suppressed) continue;
    Printf("%s", d.Leak());
    Printf("%s leak of %zu byte(s) in %zu object(s) allocated from:\n",
           leaks_[i].is_directly_leaked ? "Direct" : "Indirect",
           leaks_[i].total_size, leaks_[i].hit_count);
    Printf("%s", d.End());
    PrintStackTraceById(leaks_[i].stack_trace_id);
    leaks_printed++;
    if (leaks_printed == num_leaks_to_print) break;
  }
  if (leaks_printed < unsuppressed_count) {
    uptr remaining = unsuppressed_count - leaks_printed;
    Printf("Omitting %zu more leak(s).\n", remaining);
  }
}

void LeakReport::PrintSummary() {
  CHECK(leaks_.size() <= kMaxLeaksConsidered);
  uptr bytes = 0, allocations = 0;
  for (uptr i = 0; i < leaks_.size(); i++) {
      if (leaks_[i].is_suppressed) continue;
      bytes += leaks_[i].total_size;
      allocations += leaks_[i].hit_count;
  }
  InternalScopedBuffer<char> summary(kMaxSummaryLength);
  internal_snprintf(summary.data(), summary.size(),
                    "%zu byte(s) leaked in %zu allocation(s).", bytes,
                    allocations);
  ReportErrorSummary(summary.data());
}

uptr LeakReport::ApplySuppressions() {
  uptr unsuppressed_count = 0;
  for (uptr i = 0; i < leaks_.size(); i++) {
    Suppression *s = GetSuppressionForStack(leaks_[i].stack_trace_id);
    if (s) {
      s->weight += leaks_[i].total_size;
      s->hit_count += leaks_[i].hit_count;
      leaks_[i].is_suppressed = true;
    } else {
    unsuppressed_count++;
    }
  }
  return unsuppressed_count;
}
}  // namespace __lsan
#endif  // CAN_SANITIZE_LEAKS

using namespace __lsan;  // NOLINT

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_ignore_object(const void *p) {
#if CAN_SANITIZE_LEAKS
  if (!common_flags()->detect_leaks)
    return;
  // Cannot use PointsIntoChunk or LsanMetadata here, since the allocator is not
  // locked.
  BlockingMutexLock l(&global_mutex);
  IgnoreObjectResult res = IgnoreObjectLocked(p);
  if (res == kIgnoreObjectInvalid && flags()->verbosity >= 2)
    Report("__lsan_ignore_object(): no heap object found at %p", p);
  if (res == kIgnoreObjectAlreadyIgnored && flags()->verbosity >= 2)
    Report("__lsan_ignore_object(): "
           "heap object at %p is already being ignored\n", p);
  if (res == kIgnoreObjectSuccess && flags()->verbosity >= 3)
    Report("__lsan_ignore_object(): ignoring heap object at %p\n", p);
#endif  // CAN_SANITIZE_LEAKS
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_disable() {
#if CAN_SANITIZE_LEAKS
  __lsan::disable_counter++;
#endif
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_enable() {
#if CAN_SANITIZE_LEAKS
  if (!__lsan::disable_counter && common_flags()->detect_leaks) {
    Report("Unmatched call to __lsan_enable().\n");
    Die();
  }
  __lsan::disable_counter--;
#endif
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_do_leak_check() {
#if CAN_SANITIZE_LEAKS
  if (common_flags()->detect_leaks)
    __lsan::DoLeakCheck();
#endif  // CAN_SANITIZE_LEAKS
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
int __lsan_is_turned_off() {
  return 0;
}
#endif
}  // extern "C"
