//===-- sanitizer_report_receiver.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_report_receiver.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_report_decorator.h"

#include <stdarg.h>

namespace __sanitizer {

// Internal in sanitizer_printf.cpp — not in a public header but linked in the
// same library. Used to format AddTitleF's varargs without re-implementing
// the sanitizer printf state machine.
int VSNPrintf(char *buff, int buff_length, const char *format, va_list args);

// Fixed-size list; registration happens once at init.
static const int kMaxReceivers = 4;
static ReportReceiver *g_receivers[kMaxReceivers];
static int g_num_receivers = 0;

void RegisterReportReceiver(ReportReceiver *r) {
  if (!r)
    return;
  if (g_num_receivers >= kMaxReceivers)
    return;
  g_receivers[g_num_receivers++] = r;
}

ScopedSanitizerReport::ScopedSanitizerReport(const char *sanitizer,
                                             const char *type)
    : sanitizer_(sanitizer ? sanitizer : "") {
  for (int i = 0; i < g_num_receivers; i++)
    g_receivers[i]->OnStart(sanitizer, type);
}

ScopedSanitizerReport::~ScopedSanitizerReport() {
  for (int i = 0; i < g_num_receivers; i++)
    g_receivers[i]->OnFinish();
}

void ScopedSanitizerReport::AddTitleF(const char *fmt, ...) {
  // Format into a heap-owned buffer so we don't blow the -Wframe-larger-than
  // limit with a stack-resident 1 KiB title buffer.
  InternalMmapVector<char> buf(1024);
  va_list ap;
  va_start(ap, fmt);
  int len = VSNPrintf(buf.data(), buf.size(), fmt, ap);
  va_end(ap);
  if (len < 0)
    return;
  if ((uptr)len >= buf.size())
    len = buf.size() - 1;

  // Print with error decoration + pid prefix. Matches the shape of ASan's
  // Report("ERROR: AddressSanitizer: ...\n") sandwich.
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  Report("%s\n", buf.data());
  Printf("%s", d.Default());

  for (int i = 0; i < g_num_receivers; i++)
    g_receivers[i]->OnTitle(buf.data(), (uptr)len);
}

void ScopedSanitizerReport::AddAddress(uptr addr, uptr size,
                                       ReportAddressKind kind, const char *desc,
                                       uptr desc_len) {
  if (!addr)
    return;
  for (int i = 0; i < g_num_receivers; i++)
    g_receivers[i]->OnAddress(addr, size, kind, desc, desc_len);
}

void ScopedSanitizerReport::AddStack(const uptr *frames, uptr num_frames,
                                     ReportStackKind kind, const char *desc,
                                     uptr desc_len) {
  if (!frames || !num_frames)
    return;
  for (int i = 0; i < g_num_receivers; i++)
    g_receivers[i]->OnStack(frames, num_frames, kind, desc, desc_len);
}

}  // namespace __sanitizer
