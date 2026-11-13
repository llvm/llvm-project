//===-- sanitizer_report_receiver.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_report_receiver.h"

#include <stdarg.h>

#include "sanitizer_common.h"
#include "sanitizer_report_decorator.h"

namespace __sanitizer {

// Single receiver for now; registration happens once at init.
static ReportReceiver* g_receiver = nullptr;

void RegisterReportReceiver(ReportReceiver* r) {
  if (!r)
    return;
  CHECK(!g_receiver && "only one ReportReceiver is supported");
  g_receiver = r;
}

ScopedSanitizerReport::ScopedSanitizerReport(const char* sanitizer,
                                             const char* type,
                                             bool title_with_pid)
    : sanitizer_(sanitizer ? sanitizer : ""), title_with_pid_(title_with_pid) {
  if (g_receiver)
    g_receiver->OnStart(sanitizer, type);
}

ScopedSanitizerReport::~ScopedSanitizerReport() {
  if (g_receiver)
    g_receiver->OnFinish();
}

static void NotifyTitle(const char* title) {
  if (g_receiver)
    g_receiver->OnTitle(title);
}

// Stack buffer, like Report(): this can run when the allocator has failed.
static const uptr kTitleBufSize = 320;

void ScopedSanitizerReport::PrintTitleF(const char* color, const char* fmt,
                                        ...) {
  char buf[kTitleBufSize];
  va_list ap;
  va_start(ap, fmt);
  int len = VSNPrintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (len < 0)
    return;
  // A truncated title is a bug at the call site.
  DCHECK_LT((uptr)len, sizeof(buf));

  SanitizerCommonDecorator d;
  Printf("%s", color);
  if (title_with_pid_)
    Report("%s\n", buf);
  else
    Printf("%s\n", buf);
  Printf("%s", d.Default());
  NotifyTitle(buf);
}

void ScopedSanitizerReport::NotifyAddress(uptr addr, uptr size,
                                          ReportAddressKind kind,
                                          const char* desc) {
  if (!addr)
    return;
  if (g_receiver)
    g_receiver->OnAddress(addr, size, kind, desc);
}

void ScopedSanitizerReport::NotifyStack(const uptr* frames, uptr num_frames,
                                        ReportStackKind kind, const char* desc,
                                        u64 tid) {
  if (!frames || !num_frames)
    return;
  if (g_receiver)
    g_receiver->OnStack(frames, num_frames, kind, tid, desc);
}

}  // namespace __sanitizer
