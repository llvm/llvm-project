//===-- sanitizer_report_receiver.h ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Incremental reporting API: sanitizers push structured events (title,
// addresses, stacks) to a registered ReportReceiver as a report is emitted.
// NotifyX methods only notify the receiver; the caller prints. PrintTitleF also
// prints the header.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_REPORT_RECEIVER_H
#define SANITIZER_REPORT_RECEIVER_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// Role of a stack within a report.
enum ReportStackKind {
  // Anything that is not an allocation or deallocation, including the stack of
  // the offending operation itself (distinguished by its description).
  kReportStackOther = 0,
  kReportStackAllocation = 1,
  kReportStackDeallocation = 2,
};

// Pass as the `tid` of NotifyStack when the thread that produced the stack is
// not known.
const u64 kReportUnknownThread = ~0ull;

// Optional role hint for OnAddress. Receivers may use this to tell a fault
// address apart from an associated heap allocation.
enum ReportAddressKind {
  kReportAddressOther = 0,
  kReportAddressFault = 1,
  kReportAddressAllocation = 2,
};

// Observer of a report. Callbacks run in order between OnStart and OnFinish;
// pointers are valid only during the call. Strings are NUL-terminated and
// `desc` may be null.
class ReportReceiver {
 public:
  virtual ~ReportReceiver() {}
  // `type` is the short error name, e.g. "heap-use-after-free".
  virtual void OnStart(const char* sanitizer, const char* type) {}
  // Fully-formatted "ERROR: <sanitizer>: ..." header line for the report.
  virtual void OnTitle(const char* title) {}
  // A memory address involved in the report. `size` may be 0.
  virtual void OnAddress(uptr addr, uptr size, ReportAddressKind kind,
                         const char* desc) {}
  // A backtrace involved in the report. `frames` is top-of-stack-first. `tid`
  // is the sanitizer's own id for the thread that produced the stack (or
  // kReportUnknownThread); mapping it to an OS thread is left to the receiver.
  virtual void OnStack(const uptr* frames, uptr num_frames,
                       ReportStackKind kind, u64 tid, const char* desc) {}
  virtual void OnFinish() {}
};

// Register the (single) receiver; it must outlive the process. Registering a
// second one is fatal. Not thread-safe; call during init.
void RegisterReportReceiver(ReportReceiver* receiver);

// Forwards events to the registered receiver, if any. Lives for one report.
class ScopedSanitizerReport {
 public:
  // `title_with_pid`: PrintTitleF prefixes "==pid==" (as Report() does).
  ScopedSanitizerReport(const char* sanitizer, const char* type,
                        bool title_with_pid = true);
  ~ScopedSanitizerReport();

  // Format printf-style, print the line wrapped in `color` (e.g.
  // Decorator::Error()), and notify OnTitle. The caller supplies the full text
  // including the "ERROR: ...:" prefix, without a newline. Titles are
  // formatted into a fixed stack buffer; longer ones are truncated (DCHECK in
  // debug builds).
  void PrintTitleF(const char* color, const char* fmt, ...) FORMAT(3, 4);

  // Notify OnAddress. Does not print. addr == 0 is treated as a no-op.
  void NotifyAddress(uptr addr, uptr size, ReportAddressKind kind,
                     const char* desc);

  // Notify OnStack. Does not print. Empty stack is treated as a no-op.
  void NotifyStack(const uptr* frames, uptr num_frames, ReportStackKind kind,
                   const char* desc, u64 tid);

  const char* sanitizer() const { return sanitizer_; }

 private:
  const char* sanitizer_;
  bool title_with_pid_;

  ScopedSanitizerReport(const ScopedSanitizerReport&) = delete;
  void operator=(const ScopedSanitizerReport&) = delete;
};

}  // namespace __sanitizer

#endif  // SANITIZER_REPORT_RECEIVER_H
