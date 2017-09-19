//===-- sanitizer_common_libcdep.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"

#include "sanitizer_allocator_interface.h"
#include "sanitizer_file.h"
#include "sanitizer_flags.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_report_decorator.h"
#include "sanitizer_stackdepot.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

#if SANITIZER_POSIX
#include "sanitizer_posix.h"
#endif

namespace __sanitizer {

#if !SANITIZER_FUCHSIA

bool ReportFile::SupportsColors() {
  SpinMutexLock l(mu);
  ReopenIfNecessary();
  return SupportsColoredOutput(fd);
}

static INLINE bool ReportSupportsColors() {
  return report_file.SupportsColors();
}

#else  // SANITIZER_FUCHSIA

// Fuchsia's logs always go through post-processing that handles colorization.
static INLINE bool ReportSupportsColors() { return true; }

#endif  // !SANITIZER_FUCHSIA

bool ColorizeReports() {
  // FIXME: Add proper Windows support to AnsiColorDecorator and re-enable color
  // printing on Windows.
  if (SANITIZER_WINDOWS)
    return false;

  const char *flag = common_flags()->color;
  return internal_strcmp(flag, "always") == 0 ||
         (internal_strcmp(flag, "auto") == 0 && ReportSupportsColors());
}

static void (*sandboxing_callback)();
void SetSandboxingCallback(void (*f)()) {
  sandboxing_callback = f;
}

void ReportErrorSummary(const char *error_type, const StackTrace *stack,
                        const char *alt_tool_name) {
#if !SANITIZER_GO
  if (!common_flags()->print_summary)
    return;
  if (stack->size == 0) {
    ReportErrorSummary(error_type);
    return;
  }
  // Currently, we include the first stack frame into the report summary.
  // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
  uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
  SymbolizedStack *frame = Symbolizer::GetOrInit()->SymbolizePC(pc);
  ReportErrorSummary(error_type, frame->info, alt_tool_name);
  frame->ClearAll();
#endif
}

static void (*SoftRssLimitExceededCallback)(bool exceeded);
void SetSoftRssLimitExceededCallback(void (*Callback)(bool exceeded)) {
  CHECK_EQ(SoftRssLimitExceededCallback, nullptr);
  SoftRssLimitExceededCallback = Callback;
}

#if SANITIZER_LINUX && !SANITIZER_GO
void BackgroundThread(void *arg) {
  uptr hard_rss_limit_mb = common_flags()->hard_rss_limit_mb;
  uptr soft_rss_limit_mb = common_flags()->soft_rss_limit_mb;
  bool heap_profile = common_flags()->heap_profile;
  uptr prev_reported_rss = 0;
  uptr prev_reported_stack_depot_size = 0;
  bool reached_soft_rss_limit = false;
  uptr rss_during_last_reported_profile = 0;
  while (true) {
    SleepForMillis(100);
    uptr current_rss_mb = GetRSS() >> 20;
    if (Verbosity()) {
      // If RSS has grown 10% since last time, print some information.
      if (prev_reported_rss * 11 / 10 < current_rss_mb) {
        Printf("%s: RSS: %zdMb\n", SanitizerToolName, current_rss_mb);
        prev_reported_rss = current_rss_mb;
      }
      // If stack depot has grown 10% since last time, print it too.
      StackDepotStats *stack_depot_stats = StackDepotGetStats();
      if (prev_reported_stack_depot_size * 11 / 10 <
          stack_depot_stats->allocated) {
        Printf("%s: StackDepot: %zd ids; %zdM allocated\n",
               SanitizerToolName,
               stack_depot_stats->n_uniq_ids,
               stack_depot_stats->allocated >> 20);
        prev_reported_stack_depot_size = stack_depot_stats->allocated;
      }
    }
    // Check RSS against the limit.
    if (hard_rss_limit_mb && hard_rss_limit_mb < current_rss_mb) {
      Report("%s: hard rss limit exhausted (%zdMb vs %zdMb)\n",
             SanitizerToolName, hard_rss_limit_mb, current_rss_mb);
      DumpProcessMap();
      Die();
    }
    if (soft_rss_limit_mb) {
      if (soft_rss_limit_mb < current_rss_mb && !reached_soft_rss_limit) {
        reached_soft_rss_limit = true;
        Report("%s: soft rss limit exhausted (%zdMb vs %zdMb)\n",
               SanitizerToolName, soft_rss_limit_mb, current_rss_mb);
        if (SoftRssLimitExceededCallback)
          SoftRssLimitExceededCallback(true);
      } else if (soft_rss_limit_mb >= current_rss_mb &&
                 reached_soft_rss_limit) {
        reached_soft_rss_limit = false;
        if (SoftRssLimitExceededCallback)
          SoftRssLimitExceededCallback(false);
      }
    }
    if (heap_profile &&
        current_rss_mb > rss_during_last_reported_profile * 1.1) {
      Printf("\n\nHEAP PROFILE at RSS %zdMb\n", current_rss_mb);
      __sanitizer_print_memory_profile(90, 20);
      rss_during_last_reported_profile = current_rss_mb;
    }
  }
}
#endif

#if !SANITIZER_FUCHSIA && !SANITIZER_GO
void StartReportDeadlySignal() {
  // Write the first message using fd=2, just in case.
  // It may actually fail to write in case stderr is closed.
  CatastrophicErrorWrite(SanitizerToolName, internal_strlen(SanitizerToolName));
  static const char kDeadlySignal[] = ":DEADLYSIGNAL\n";
  CatastrophicErrorWrite(kDeadlySignal, sizeof(kDeadlySignal) - 1);
}

static void MaybeReportNonExecRegion(uptr pc) {
#if SANITIZER_FREEBSD || SANITIZER_LINUX || SANITIZER_NETBSD
  MemoryMappingLayout proc_maps(/*cache_enabled*/ true);
  MemoryMappedSegment segment;
  while (proc_maps.Next(&segment)) {
    if (pc >= segment.start && pc < segment.end && !segment.IsExecutable())
      Report("Hint: PC is at a non-executable region. Maybe a wild jump?\n");
  }
#endif
}

static void PrintMemoryByte(InternalScopedString *str, const char *before,
                            u8 byte) {
  SanitizerCommonDecorator d;
  str->append("%s%s%x%x%s ", before, d.MemoryByte(), byte >> 4, byte & 15,
              d.Default());
}

static void MaybeDumpInstructionBytes(uptr pc) {
  if (!common_flags()->dump_instruction_bytes || (pc < GetPageSizeCached()))
    return;
  InternalScopedString str(1024);
  str.append("First 16 instruction bytes at pc: ");
  if (IsAccessibleMemoryRange(pc, 16)) {
    for (int i = 0; i < 16; ++i) {
      PrintMemoryByte(&str, "", ((u8 *)pc)[i]);
    }
    str.append("\n");
  } else {
    str.append("unaccessible\n");
  }
  Report("%s", str.data());
}

static void MaybeDumpRegisters(void *context) {
  if (!common_flags()->dump_registers) return;
  SignalContext::DumpAllRegisters(context);
}

static void ReportStackOverflowImpl(const SignalContext &sig, u32 tid,
                                    UnwindSignalStackCallbackType unwind,
                                    const void *unwind_context) {
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  static const char kDescription[] = "stack-overflow";
  Report("ERROR: %s: %s on address %p (pc %p bp %p sp %p T%d)\n",
         SanitizerToolName, kDescription, (void *)sig.addr, (void *)sig.pc,
         (void *)sig.bp, (void *)sig.sp, tid);
  Printf("%s", d.Default());
  InternalScopedBuffer<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  unwind(sig, unwind_context, stack);
  stack->Print();
  ReportErrorSummary(kDescription, stack);
}

static void ReportDeadlySignalImpl(const SignalContext &sig, u32 tid,
                                   UnwindSignalStackCallbackType unwind,
                                   const void *unwind_context) {
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  const char *description = sig.Describe();
  Report("ERROR: %s: %s on unknown address %p (pc %p bp %p sp %p T%d)\n",
         SanitizerToolName, description, (void *)sig.addr, (void *)sig.pc,
         (void *)sig.bp, (void *)sig.sp, tid);
  Printf("%s", d.Default());
  if (sig.pc < GetPageSizeCached())
    Report("Hint: pc points to the zero page.\n");
  if (sig.is_memory_access) {
    const char *access_type =
        sig.write_flag == SignalContext::WRITE
            ? "WRITE"
            : (sig.write_flag == SignalContext::READ ? "READ" : "UNKNOWN");
    Report("The signal is caused by a %s memory access.\n", access_type);
    if (sig.addr < GetPageSizeCached())
      Report("Hint: address points to the zero page.\n");
  }
  MaybeReportNonExecRegion(sig.pc);
  InternalScopedBuffer<BufferedStackTrace> stack_buffer(1);
  BufferedStackTrace *stack = stack_buffer.data();
  stack->Reset();
  unwind(sig, unwind_context, stack);
  stack->Print();
  MaybeDumpInstructionBytes(sig.pc);
  MaybeDumpRegisters(sig.context);
  Printf("%s can not provide additional info.\n", SanitizerToolName);
  ReportErrorSummary(description, stack);
}

void ReportDeadlySignal(const SignalContext &sig, u32 tid,
                        UnwindSignalStackCallbackType unwind,
                        const void *unwind_context) {
  if (sig.IsStackOverflow())
    ReportStackOverflowImpl(sig, tid, unwind, unwind_context);
  else
    ReportDeadlySignalImpl(sig, tid, unwind, unwind_context);
}
#endif  // !SANITIZER_FUCHSIA && !SANITIZER_GO

void WriteToSyslog(const char *msg) {
  InternalScopedString msg_copy(kErrorMessageBufferSize);
  msg_copy.append("%s", msg);
  char *p = msg_copy.data();
  char *q;

  // Print one line at a time.
  // syslog, at least on Android, has an implicit message length limit.
  do {
    q = internal_strchr(p, '\n');
    if (q)
      *q = '\0';
    WriteOneLineToSyslog(p);
    if (q)
      p = q + 1;
  } while (q);
}

void MaybeStartBackgroudThread() {
#if SANITIZER_LINUX && \
    !SANITIZER_GO  // Need to implement/test on other platforms.
  // Start the background thread if one of the rss limits is given.
  if (!common_flags()->hard_rss_limit_mb &&
      !common_flags()->soft_rss_limit_mb &&
      !common_flags()->heap_profile) return;
  if (!&real_pthread_create) return;  // Can't spawn the thread anyway.
  internal_start_thread(BackgroundThread, nullptr);
#endif
}

}  // namespace __sanitizer

SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_sandbox_on_notify,
                             __sanitizer_sandbox_arguments *args) {
  __sanitizer::PrepareForSandboxing(args);
  if (__sanitizer::sandboxing_callback)
    __sanitizer::sandboxing_callback();
}
