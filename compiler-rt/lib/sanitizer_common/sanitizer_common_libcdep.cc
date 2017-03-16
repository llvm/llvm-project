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
#include "sanitizer_flags.h"
#include "sanitizer_stackdepot.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

#if SANITIZER_POSIX
#include "sanitizer_posix.h"
#endif

namespace __sanitizer {

bool ReportFile::SupportsColors() {
  SpinMutexLock l(mu);
  ReopenIfNecessary();
  return SupportsColoredOutput(fd);
}

bool ColorizeReports() {
  // FIXME: Add proper Windows support to AnsiColorDecorator and re-enable color
  // printing on Windows.
  if (SANITIZER_WINDOWS)
    return false;

  const char *flag = common_flags()->color;
  return internal_strcmp(flag, "always") == 0 ||
         (internal_strcmp(flag, "auto") == 0 && report_file.SupportsColors());
}

static void (*sandboxing_callback)();
void SetSandboxingCallback(void (*f)()) {
  sandboxing_callback = f;
}

void ReportErrorSummary(const char *error_type, const StackTrace *stack) {
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
  ReportErrorSummary(error_type, frame->info);
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
