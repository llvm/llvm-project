//===-- sanitizer_report_receiver.h ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Incremental reporting API for sanitizers. Sanitizer runtimes construct a
// ScopedSanitizerReport at the start of an error report and push structured
// events (title, addresses, backtraces) as they become known. Each registered
// ReportReceiver observes the events in order and can translate them into
// its own stable format (e.g. the Darwin crash-reporter payload).
//
// The ScopedSanitizerReport is threaded by reference (or pointer, where the
// caller may not be inside a report) through the sanitizer's Print methods.
// AddTitleF is the one method that also drives stderr output — it formats
// the "ERROR: <sanitizer>: ..." header line, prints it with decoration, and
// notifies OnTitle. AddAddress and AddStack are receiver-only; the callers
// keep their own printing right next to the AddX call so a single site drives
// both.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_REPORT_RECEIVER_H
#define SANITIZER_REPORT_RECEIVER_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

enum ReportStackKind {
  kReportStackOther = 0,
  kReportStackAllocation = 1,
  kReportStackDeallocation = 2,
  // The stack of the offending operation itself (the memory access that
  // faulted, the second-free call, the misbehaving alloc-parameter call, etc).
  kReportStackFault = 3,
};

// Optional role hint for OnAddress. Receivers may use this to tell a fault
// address apart from an associated heap allocation.
enum ReportAddressKind {
  kReportAddressOther = 0,
  kReportAddressFault = 1,
  kReportAddressAllocation = 2,
};

// Observer of a sanitizer report as it is emitted. Callbacks are invoked in
// order between OnStart and OnFinish. All string/frame pointers passed in are
// valid only for the duration of the call.
class ReportReceiver {
 public:
  virtual ~ReportReceiver() {}
  virtual void OnStart(const char *sanitizer, const char *type) {}
  // Fully-formatted "ERROR: <sanitizer>: ..." header line for the report.
  virtual void OnTitle(const char *title, uptr title_len) {}
  // A memory address involved in the report. `size` may be 0.
  virtual void OnAddress(uptr addr, uptr size, ReportAddressKind kind,
                         const char *desc, uptr desc_len) {}
  // A backtrace involved in the report. `frames` is top-of-stack-first.
  virtual void OnStack(const uptr *frames, uptr num_frames,
                       ReportStackKind kind, const char *desc,
                       uptr desc_len) {}
  virtual void OnFinish() {}
};

// Register a receiver. The pointer must remain valid for the process lifetime.
// Not thread-safe; call during platform init before any report can fire.
void RegisterReportReceiver(ReportReceiver *receiver);

// Piped to all registered receivers. Construct at the start of a report,
// destroy at the end. Passed by reference (or pointer where the caller may
// be outside a report) through the sanitizer's Print methods.
class ScopedSanitizerReport {
 public:
  ScopedSanitizerReport(const char *sanitizer, const char *type);
  ~ScopedSanitizerReport();

  // Format printf-style, print the resulting line with error decoration + pid
  // prefix, and notify OnTitle. Caller writes the full title text (including
  // the "ERROR: AddressSanitizer:" or "WARNING: ThreadSanitizer:" prefix); no
  // newline needed — one is appended for printing.
  void AddTitleF(const char *fmt, ...) FORMAT(2, 3);

  // Notify OnAddress. Does not print. addr == 0 is treated as a no-op.
  void AddAddress(uptr addr, uptr size, ReportAddressKind kind,
                  const char *desc, uptr desc_len);

  // Notify OnStack. Does not print. Empty stack is treated as a no-op.
  void AddStack(const uptr *frames, uptr num_frames, ReportStackKind kind,
                const char *desc, uptr desc_len);

  const char *sanitizer() const { return sanitizer_; }

 private:
  const char *sanitizer_;

  ScopedSanitizerReport(const ScopedSanitizerReport &) = delete;
  void operator=(const ScopedSanitizerReport &) = delete;
};

}  // namespace __sanitizer

#endif  // SANITIZER_REPORT_RECEIVER_H
