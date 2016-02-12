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
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_suppressions.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

#if CAN_SANITIZE_LEAKS
namespace __lsan {

// This mutex is used to prevent races between DoLeakCheck and IgnoreObject, and
// also to protect the global list of root regions.
BlockingMutex global_mutex(LINKER_INITIALIZED);

THREADLOCAL int disable_counter;
bool DisabledInThisThread() { return disable_counter > 0; }
void DisableInThisThread() { disable_counter++; }
void EnableInThisThread() {
  if (!disable_counter && common_flags()->detect_leaks) {
    Report("Unmatched call to __lsan_enable().\n");
    Die();
  }
  disable_counter--;
}

Flags lsan_flags;

void Flags::SetDefaults() {
#define LSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "lsan_flags.inc"
#undef LSAN_FLAG
}

void RegisterLsanFlags(FlagParser *parser, Flags *f) {
#define LSAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "lsan_flags.inc"
#undef LSAN_FLAG
}

#define LOG_POINTERS(...)                           \
  do {                                              \
    if (flags()->log_pointers) Report(__VA_ARGS__); \
  } while (0);

#define LOG_THREADS(...)                           \
  do {                                             \
    if (flags()->log_threads) Report(__VA_ARGS__); \
  } while (0);

ALIGNED(64) static char suppression_placeholder[sizeof(SuppressionContext)];
static SuppressionContext *suppression_ctx = nullptr;
static const char kSuppressionLeak[] = "leak";
static const char *kSuppressionTypes[] = { kSuppressionLeak };

void InitializeSuppressions() {
  CHECK_EQ(nullptr, suppression_ctx);
  suppression_ctx = new (suppression_placeholder) // NOLINT
      SuppressionContext(kSuppressionTypes, ARRAY_SIZE(kSuppressionTypes));
  suppression_ctx->ParseFromFile(flags()->suppressions);
  if (&__lsan_default_suppressions)
    suppression_ctx->Parse(__lsan_default_suppressions());
}

static SuppressionContext *GetSuppressionContext() {
  CHECK(suppression_ctx);
  return suppression_ctx;
}

struct RootRegion {
  const void *begin;
  uptr size;
};

InternalMmapVector<RootRegion> *root_regions;

void InitializeRootRegions() {
  CHECK(!root_regions);
  ALIGNED(64) static char placeholder[sizeof(InternalMmapVector<RootRegion>)];
  root_regions = new(placeholder) InternalMmapVector<RootRegion>(1);
}

void InitCommonLsan() {
  InitializeRootRegions();
  if (common_flags()->detect_leaks) {
    // Initialization which can fail or print warnings should only be done if
    // LSan is actually enabled.
    InitializeSuppressions();
    InitializePlatformSpecificModules();
  }
}

class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Error() { return Red(); }
  const char *Leak() { return Blue(); }
  const char *End() { return Default(); }
};

static inline bool CanBeAHeapPointer(uptr p) {
  // Since our heap is located in mmap-ed memory, we can assume a sensible lower
  // bound on heap addresses.
  const uptr kMinAddress = 4 * 4096;
  if (p < kMinAddress) return false;
#if defined(__x86_64__)
  // Accept only canonical form user-space addresses.
  return ((p >> 47) == 0);
#elif defined(__mips64)
  return ((p >> 40) == 0);
#elif defined(__aarch64__)
  unsigned runtimeVMA =
    (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1);
  return ((p >> runtimeVMA) == 0);
#else
  return true;
#endif
}

// Scans the memory range, looking for byte patterns that point into allocator
// chunks. Marks those chunks with |tag| and adds them to |frontier|.
// There are two usage modes for this function: finding reachable chunks
// (|tag| = kReachable) and finding indirectly leaked chunks
// (|tag| = kIndirectlyLeaked). In the second case, there's no flood fill,
// so |frontier| = 0.
void ScanRangeForPointers(uptr begin, uptr end,
                          Frontier *frontier,
                          const char *region_type, ChunkTag tag) {
  CHECK(tag == kReachable || tag == kIndirectlyLeaked);
  const uptr alignment = flags()->pointer_alignment();
  LOG_POINTERS("Scanning %s range %p-%p.\n", region_type, begin, end);
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
    if (m.tag() == kReachable || m.tag() == kIgnored) continue;

    // Do this check relatively late so we can log only the interesting cases.
    if (!flags()->use_poisoned && WordIsPoisoned(pp)) {
      LOG_POINTERS(
          "%p is poisoned: ignoring %p pointing into chunk %p-%p of size "
          "%zu.\n",
          pp, p, chunk, chunk + m.requested_size(), m.requested_size());
      continue;
    }

    m.set_tag(tag);
    LOG_POINTERS("%p: found %p pointing into chunk %p-%p of size %zu.\n", pp, p,
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
    LOG_THREADS("Processing thread %d.\n", os_id);
    uptr stack_begin, stack_end, tls_begin, tls_end, cache_begin, cache_end;
    DTLS *dtls;
    bool thread_found = GetThreadRangesLocked(os_id, &stack_begin, &stack_end,
                                              &tls_begin, &tls_end,
                                              &cache_begin, &cache_end, &dtls);
    if (!thread_found) {
      // If a thread can't be found in the thread registry, it's probably in the
      // process of destruction. Log this event and move on.
      LOG_THREADS("Thread %d not found in registry.\n", os_id);
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
      LOG_THREADS("Stack at %p-%p (SP = %p).\n", stack_begin, stack_end, sp);
      if (sp < stack_begin || sp >= stack_end) {
        // SP is outside the recorded stack range (e.g. the thread is running a
        // signal handler on alternate stack, or swapcontext was used).
        // Again, consider the entire stack range to be reachable.
        LOG_THREADS("WARNING: stack pointer not in stack range.\n");
        uptr page_size = GetPageSizeCached();
        int skipped = 0;
        while (stack_begin < stack_end &&
               !IsAccessibleMemoryRange(stack_begin, 1)) {
          skipped++;
          stack_begin += page_size;
        }
        LOG_THREADS("Skipped %d guard page(s) to obtain stack %p-%p.\n",
                    skipped, stack_begin, stack_end);
      } else {
        // Shrink the stack range to ignore out-of-scope values.
        stack_begin = sp;
      }
      ScanRangeForPointers(stack_begin, stack_end, frontier, "STACK",
                           kReachable);
      ForEachExtraStackRange(os_id, ForEachExtraStackRangeCb, frontier);
    }

    if (flags()->use_tls) {
      LOG_THREADS("TLS at %p-%p.\n", tls_begin, tls_end);
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
      if (dtls) {
        for (uptr j = 0; j < dtls->dtv_size; ++j) {
          uptr dtls_beg = dtls->dtv[j].beg;
          uptr dtls_end = dtls_beg + dtls->dtv[j].size;
          if (dtls_beg < dtls_end) {
            LOG_THREADS("DTLS %zu at %p-%p.\n", j, dtls_beg, dtls_end);
            ScanRangeForPointers(dtls_beg, dtls_end, frontier, "DTLS",
                                 kReachable);
          }
        }
      }
    }
  }
}

static void ProcessRootRegion(Frontier *frontier, uptr root_begin,
                              uptr root_end) {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr begin, end, prot;
  while (proc_maps.Next(&begin, &end,
                        /*offset*/ nullptr, /*filename*/ nullptr,
                        /*filename_size*/ 0, &prot)) {
    uptr intersection_begin = Max(root_begin, begin);
    uptr intersection_end = Min(end, root_end);
    if (intersection_begin >= intersection_end) continue;
    bool is_readable = prot & MemoryMappingLayout::kProtectionRead;
    LOG_POINTERS("Root region %p-%p intersects with mapped region %p-%p (%s)\n",
                 root_begin, root_end, begin, end,
                 is_readable ? "readable" : "unreadable");
    if (is_readable)
      ScanRangeForPointers(intersection_begin, intersection_end, frontier,
                           "ROOT", kReachable);
  }
}

// Scans root regions for heap pointers.
static void ProcessRootRegions(Frontier *frontier) {
  if (!flags()->use_root_regions) return;
  CHECK(root_regions);
  for (uptr i = 0; i < root_regions->size(); i++) {
    RootRegion region = (*root_regions)[i];
    uptr begin_addr = reinterpret_cast<uptr>(region.begin);
    ProcessRootRegion(frontier, begin_addr, begin_addr + region.size);
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
                         /* frontier */ nullptr, "HEAP", kIndirectlyLeaked);
  }
}

// ForEachChunk callback. If chunk is marked as ignored, adds its address to
// frontier.
static void CollectIgnoredCb(uptr chunk, void *arg) {
  CHECK(arg);
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (m.allocated() && m.tag() == kIgnored) {
    LOG_POINTERS("Ignored: chunk %p-%p of size %zu.\n",
                 chunk, chunk + m.requested_size(), m.requested_size());
    reinterpret_cast<Frontier *>(arg)->push_back(chunk);
  }
}

// Sets the appropriate tag on each chunk.
static void ClassifyAllChunks(SuspendedThreadsList const &suspended_threads) {
  // Holds the flood fill frontier.
  Frontier frontier(1);

  ForEachChunk(CollectIgnoredCb, &frontier);
  ProcessGlobalRegions(&frontier);
  ProcessThreads(suspended_threads, &frontier);
  ProcessRootRegions(&frontier);
  FloodFillTag(&frontier, kReachable);

  // The check here is relatively expensive, so we do this in a separate flood
  // fill. That way we can skip the check for chunks that are reachable
  // otherwise.
  LOG_POINTERS("Processing platform-specific allocations.\n");
  CHECK_EQ(0, frontier.size());
  ProcessPlatformSpecificAllocations(&frontier);
  FloodFillTag(&frontier, kReachable);

  // Iterate over leaked chunks and mark those that are reachable from other
  // leaked chunks.
  LOG_POINTERS("Scanning leaked chunks.\n");
  ForEachChunk(MarkIndirectlyLeakedCb, nullptr);
}

// ForEachChunk callback. Resets the tags to pre-leak-check state.
static void ResetTagsCb(uptr chunk, void *arg) {
  (void)arg;
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (m.allocated() && m.tag() != kIgnored)
    m.set_tag(kDirectlyLeaked);
}

static void PrintStackTraceById(u32 stack_trace_id) {
  CHECK(stack_trace_id);
  StackDepotGet(stack_trace_id).Print();
}

// ForEachChunk callback. Aggregates information about unreachable chunks into
// a LeakReport.
static void CollectLeaksCb(uptr chunk, void *arg) {
  CHECK(arg);
  LeakReport *leak_report = reinterpret_cast<LeakReport *>(arg);
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (!m.allocated()) return;
  if (m.tag() == kDirectlyLeaked || m.tag() == kIndirectlyLeaked) {
    u32 resolution = flags()->resolution;
    u32 stack_trace_id = 0;
    if (resolution > 0) {
      StackTrace stack = StackDepotGet(m.stack_trace_id());
      stack.size = Min(stack.size, resolution);
      stack_trace_id = StackDepotPut(stack);
    } else {
      stack_trace_id = m.stack_trace_id();
    }
    leak_report->AddLeakedChunk(chunk, stack_trace_id, m.requested_size(),
                                m.tag());
  }
}

static void PrintMatchedSuppressions() {
  InternalMmapVector<Suppression *> matched(1);
  GetSuppressionContext()->GetMatched(&matched);
  if (!matched.size())
    return;
  const char *line = "-----------------------------------------------------";
  Printf("%s\n", line);
  Printf("Suppressions used:\n");
  Printf("  count      bytes template\n");
  for (uptr i = 0; i < matched.size(); i++)
    Printf("%7zu %10zu %s\n", static_cast<uptr>(atomic_load_relaxed(
        &matched[i]->hit_count)), matched[i]->weight, matched[i]->templ);
  Printf("%s\n\n", line);
}

struct CheckForLeaksParam {
  bool success;
  LeakReport leak_report;
};

static void CheckForLeaksCallback(const SuspendedThreadsList &suspended_threads,
                                  void *arg) {
  CheckForLeaksParam *param = reinterpret_cast<CheckForLeaksParam *>(arg);
  CHECK(param);
  CHECK(!param->success);
  ClassifyAllChunks(suspended_threads);
  ForEachChunk(CollectLeaksCb, &param->leak_report);
  // Clean up for subsequent leak checks. This assumes we did not overwrite any
  // kIgnored tags.
  ForEachChunk(ResetTagsCb, nullptr);
  param->success = true;
}

static bool CheckForLeaks() {
  if (&__lsan_is_turned_off && __lsan_is_turned_off())
      return false;
  EnsureMainThreadIDIsCorrect();
  CheckForLeaksParam param;
  param.success = false;
  LockThreadRegistry();
  LockAllocator();
  DoStopTheWorld(CheckForLeaksCallback, &param);
  UnlockAllocator();
  UnlockThreadRegistry();

  if (!param.success) {
    Report("LeakSanitizer has encountered a fatal error.\n");
    Report(
        "HINT: For debugging, try setting environment variable "
        "LSAN_OPTIONS=verbosity=1:log_threads=1\n");
    Die();
  }
  param.leak_report.ApplySuppressions();
  uptr unsuppressed_count = param.leak_report.UnsuppressedLeakCount();
  if (unsuppressed_count > 0) {
    Decorator d;
    Printf("\n"
           "================================================================="
           "\n");
    Printf("%s", d.Error());
    Report("ERROR: LeakSanitizer: detected memory leaks\n");
    Printf("%s", d.End());
    param.leak_report.ReportTopLeaks(flags()->max_leaks);
  }
  if (common_flags()->print_suppressions)
    PrintMatchedSuppressions();
  if (unsuppressed_count > 0) {
    param.leak_report.PrintSummary();
    return true;
  }
  return false;
}

void DoLeakCheck() {
  BlockingMutexLock l(&global_mutex);
  static bool already_done;
  if (already_done) return;
  already_done = true;
  bool have_leaks = CheckForLeaks();
  if (!have_leaks) {
    return;
  }
  if (common_flags()->exitcode) {
    Die();
  }
}

static int DoRecoverableLeakCheck() {
  BlockingMutexLock l(&global_mutex);
  bool have_leaks = CheckForLeaks();
  return have_leaks ? 1 : 0;
}

static Suppression *GetSuppressionForAddr(uptr addr) {
  Suppression *s = nullptr;

  // Suppress by module name.
  SuppressionContext *suppressions = GetSuppressionContext();
  if (const char *module_name =
          Symbolizer::GetOrInit()->GetModuleNameForPc(addr))
    if (suppressions->Match(module_name, kSuppressionLeak, &s))
      return s;

  // Suppress by file or function name.
  SymbolizedStack *frames = Symbolizer::GetOrInit()->SymbolizePC(addr);
  for (SymbolizedStack *cur = frames; cur; cur = cur->next) {
    if (suppressions->Match(cur->info.function, kSuppressionLeak, &s) ||
        suppressions->Match(cur->info.file, kSuppressionLeak, &s)) {
      break;
    }
  }
  frames->ClearAll();
  return s;
}

static Suppression *GetSuppressionForStack(u32 stack_trace_id) {
  StackTrace stack = StackDepotGet(stack_trace_id);
  for (uptr i = 0; i < stack.size; i++) {
    Suppression *s = GetSuppressionForAddr(
        StackTrace::GetPreviousInstructionPc(stack.trace[i]));
    if (s) return s;
  }
  return nullptr;
}

///// LeakReport implementation. /////

// A hard limit on the number of distinct leaks, to avoid quadratic complexity
// in LeakReport::AddLeakedChunk(). We don't expect to ever see this many leaks
// in real-world applications.
// FIXME: Get rid of this limit by changing the implementation of LeakReport to
// use a hash table.
const uptr kMaxLeaksConsidered = 5000;

void LeakReport::AddLeakedChunk(uptr chunk, u32 stack_trace_id,
                                uptr leaked_size, ChunkTag tag) {
  CHECK(tag == kDirectlyLeaked || tag == kIndirectlyLeaked);
  bool is_directly_leaked = (tag == kDirectlyLeaked);
  uptr i;
  for (i = 0; i < leaks_.size(); i++) {
    if (leaks_[i].stack_trace_id == stack_trace_id &&
        leaks_[i].is_directly_leaked == is_directly_leaked) {
      leaks_[i].hit_count++;
      leaks_[i].total_size += leaked_size;
      break;
    }
  }
  if (i == leaks_.size()) {
    if (leaks_.size() == kMaxLeaksConsidered) return;
    Leak leak = { next_id_++, /* hit_count */ 1, leaked_size, stack_trace_id,
                  is_directly_leaked, /* is_suppressed */ false };
    leaks_.push_back(leak);
  }
  if (flags()->report_objects) {
    LeakedObject obj = {leaks_[i].id, chunk, leaked_size};
    leaked_objects_.push_back(obj);
  }
}

static bool LeakComparator(const Leak &leak1, const Leak &leak2) {
  if (leak1.is_directly_leaked == leak2.is_directly_leaked)
    return leak1.total_size > leak2.total_size;
  else
    return leak1.is_directly_leaked;
}

void LeakReport::ReportTopLeaks(uptr num_leaks_to_report) {
  CHECK(leaks_.size() <= kMaxLeaksConsidered);
  Printf("\n");
  if (leaks_.size() == kMaxLeaksConsidered)
    Printf("Too many leaks! Only the first %zu leaks encountered will be "
           "reported.\n",
           kMaxLeaksConsidered);

  uptr unsuppressed_count = UnsuppressedLeakCount();
  if (num_leaks_to_report > 0 && num_leaks_to_report < unsuppressed_count)
    Printf("The %zu top leak(s):\n", num_leaks_to_report);
  InternalSort(&leaks_, leaks_.size(), LeakComparator);
  uptr leaks_reported = 0;
  for (uptr i = 0; i < leaks_.size(); i++) {
    if (leaks_[i].is_suppressed) continue;
    PrintReportForLeak(i);
    leaks_reported++;
    if (leaks_reported == num_leaks_to_report) break;
  }
  if (leaks_reported < unsuppressed_count) {
    uptr remaining = unsuppressed_count - leaks_reported;
    Printf("Omitting %zu more leak(s).\n", remaining);
  }
}

void LeakReport::PrintReportForLeak(uptr index) {
  Decorator d;
  Printf("%s", d.Leak());
  Printf("%s leak of %zu byte(s) in %zu object(s) allocated from:\n",
         leaks_[index].is_directly_leaked ? "Direct" : "Indirect",
         leaks_[index].total_size, leaks_[index].hit_count);
  Printf("%s", d.End());

  PrintStackTraceById(leaks_[index].stack_trace_id);

  if (flags()->report_objects) {
    Printf("Objects leaked above:\n");
    PrintLeakedObjectsForLeak(index);
    Printf("\n");
  }
}

void LeakReport::PrintLeakedObjectsForLeak(uptr index) {
  u32 leak_id = leaks_[index].id;
  for (uptr j = 0; j < leaked_objects_.size(); j++) {
    if (leaked_objects_[j].leak_id == leak_id)
      Printf("%p (%zu bytes)\n", leaked_objects_[j].addr,
             leaked_objects_[j].size);
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
  InternalScopedString summary(kMaxSummaryLength);
  summary.append("%zu byte(s) leaked in %zu allocation(s).", bytes,
                 allocations);
  ReportErrorSummary(summary.data());
}

void LeakReport::ApplySuppressions() {
  for (uptr i = 0; i < leaks_.size(); i++) {
    Suppression *s = GetSuppressionForStack(leaks_[i].stack_trace_id);
    if (s) {
      s->weight += leaks_[i].total_size;
      atomic_store_relaxed(&s->hit_count, atomic_load_relaxed(&s->hit_count) +
          leaks_[i].hit_count);
      leaks_[i].is_suppressed = true;
    }
  }
}

uptr LeakReport::UnsuppressedLeakCount() {
  uptr result = 0;
  for (uptr i = 0; i < leaks_.size(); i++)
    if (!leaks_[i].is_suppressed) result++;
  return result;
}

} // namespace __lsan
#endif // CAN_SANITIZE_LEAKS

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
  if (res == kIgnoreObjectInvalid)
    VReport(1, "__lsan_ignore_object(): no heap object found at %p", p);
  if (res == kIgnoreObjectAlreadyIgnored)
    VReport(1, "__lsan_ignore_object(): "
           "heap object at %p is already being ignored\n", p);
  if (res == kIgnoreObjectSuccess)
    VReport(1, "__lsan_ignore_object(): ignoring heap object at %p\n", p);
#endif // CAN_SANITIZE_LEAKS
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_register_root_region(const void *begin, uptr size) {
#if CAN_SANITIZE_LEAKS
  BlockingMutexLock l(&global_mutex);
  CHECK(root_regions);
  RootRegion region = {begin, size};
  root_regions->push_back(region);
  VReport(1, "Registered root region at %p of size %llu\n", begin, size);
#endif // CAN_SANITIZE_LEAKS
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_unregister_root_region(const void *begin, uptr size) {
#if CAN_SANITIZE_LEAKS
  BlockingMutexLock l(&global_mutex);
  CHECK(root_regions);
  bool removed = false;
  for (uptr i = 0; i < root_regions->size(); i++) {
    RootRegion region = (*root_regions)[i];
    if (region.begin == begin && region.size == size) {
      removed = true;
      uptr last_index = root_regions->size() - 1;
      (*root_regions)[i] = (*root_regions)[last_index];
      root_regions->pop_back();
      VReport(1, "Unregistered root region at %p of size %llu\n", begin, size);
      break;
    }
  }
  if (!removed) {
    Report(
        "__lsan_unregister_root_region(): region at %p of size %llu has not "
        "been registered.\n",
        begin, size);
    Die();
  }
#endif // CAN_SANITIZE_LEAKS
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_disable() {
#if CAN_SANITIZE_LEAKS
  __lsan::DisableInThisThread();
#endif
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_enable() {
#if CAN_SANITIZE_LEAKS
  __lsan::EnableInThisThread();
#endif
}

SANITIZER_INTERFACE_ATTRIBUTE
void __lsan_do_leak_check() {
#if CAN_SANITIZE_LEAKS
  if (common_flags()->detect_leaks)
    __lsan::DoLeakCheck();
#endif // CAN_SANITIZE_LEAKS
}

SANITIZER_INTERFACE_ATTRIBUTE
int __lsan_do_recoverable_leak_check() {
#if CAN_SANITIZE_LEAKS
  if (common_flags()->detect_leaks)
    return __lsan::DoRecoverableLeakCheck();
#endif // CAN_SANITIZE_LEAKS
  return 0;
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
int __lsan_is_turned_off() {
  return 0;
}
#endif
} // extern "C"
