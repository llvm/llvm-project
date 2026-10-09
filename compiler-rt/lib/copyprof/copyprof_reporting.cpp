//===-- copyprof_reporting.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the reporting mechanisms for CopyProf, generating
/// reports with stack traces when unnecessary object copies are destroyed.
///
//===----------------------------------------------------------------------===//

#include "copyprof_reporting.h"

#include <pthread.h>

#include "copyprof_flags.h"
#include "copyprof_report.h"
#include "copyprof_report_sink.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __copyprof {
namespace {

// TODO: Make configurable via flags.
constexpr int kMaxNumStackFrames = 30;

// Per-thread state for logging reports.
// Allocated using THREADLOCAL and pthread API to avoid C++ runtime
// dependencies that thread_local would introduce.
struct PerThreadReportState {
  ReportBuffer* buffer;
  usize obj_size_threshold;
  bool must_allocate;
  bool initialized;
};

THREADLOCAL PerThreadReportState tls_report_state;

// pthread key to register a callback that returns the thread's report buffer to
// the sink when the thread exits.
static pthread_once_t tls_key_once = PTHREAD_ONCE_INIT;
static pthread_key_t tls_cleanup_key;

void TlsCleanup(void*) {
  if (tls_report_state.initialized) {
    ReturnReportBuffer(tls_report_state.buffer);
    tls_report_state.buffer = nullptr;
    tls_report_state.initialized = false;
  }
}

void CreateTlsKey() { pthread_key_create(&tls_cleanup_key, TlsCleanup); }

void EnsureInitialized(PerThreadReportState& state) {
  if (LIKELY(state.initialized))
    return;
  state.buffer = AcquireReportBuffer();
  state.obj_size_threshold = flags()->obj_size_threshold;
  state.must_allocate = flags()->must_allocate;
  state.initialized = true;

  // Register the cleanup destructor, then set a non-null value on this thread
  // so the destructor actually fires when the thread exits.
  pthread_once(&tls_key_once, CreateTlsKey);
  pthread_setspecific(tls_cleanup_key, (void*)1);
}

bool SkipReport(const PerThreadReportState& state, usize obj_size,
                bool did_allocate) {
  return (obj_size <= state.obj_size_threshold) ||
         (state.must_allocate && !did_allocate);
}

void* AllocateReportMem(PerThreadReportState& state) {
  if (void* ptr = state.buffer->TryAllocate(sizeof(CopyProfReport));
      ptr != nullptr) {
    return ptr;
  }
  ReturnReportBuffer(state.buffer);
  state.buffer = AcquireReportBuffer();
  return state.buffer->TryAllocate(sizeof(CopyProfReport));
}

}  // namespace

void LogCopyProfReport(uptr pc, uptr bp, uptr obj_size, bool did_allocate) {
  EnsureInitialized(tls_report_state);
  if (SkipReport(tls_report_state, obj_size, did_allocate))
    return;
  UNINITIALIZED BufferedStackTrace stack_trace;
  stack_trace.Unwind(pc, bp, /*context=*/nullptr,
                     common_flags()->fast_unwind_on_fatal, kMaxNumStackFrames);
  new (AllocateReportMem(tls_report_state)) CopyProfReport{
      .stack_trace_id = StackDepotPut(stack_trace), .copy_num_bytes = obj_size};
}
void FlushAndReturnCurrentThreadBuffer() {
  if (!tls_report_state.initialized)
    return;
  ReturnReportBuffer(tls_report_state.buffer);
  tls_report_state.buffer = nullptr;
  tls_report_state.initialized = false;
}

}  // namespace __copyprof
