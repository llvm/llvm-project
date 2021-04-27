//===-- tsan_symbolize.cpp ------------------------------------------------===//
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

#include "tsan_symbolize.h"

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "tsan_flags.h"
#include "tsan_report.h"
#include "tsan_rtl.h"

namespace __tsan {

// Legacy API.
// May be overriden by JIT/JAVA/etc,
// whatever produces PCs marked with kExternalPCBit.
SANITIZER_WEAK_DEFAULT_IMPL
bool __tsan_symbolize_external(uptr pc, char *func_buf, uptr func_siz,
                               char *file_buf, uptr file_siz, int *line,
                               int *col) {
  return false;
}

// New API: call __tsan_symbolize_external_ex only when it exists.
// Once old clients are gone, provide dummy implementation.
SANITIZER_WEAK_DEFAULT_IMPL
void __tsan_symbolize_external_ex(uptr pc,
                                  void (*add_frame)(void *, const char *,
                                                    const char *, int, int),
                                  void *ctx) {}

struct SymbolizedStackBuilder {
  SymbolizedStack *head;
  SymbolizedStack *tail;
  uptr addr;
};

static void AddFrame(void *ctx, const char *function_name, const char *file,
                     int line, int column) {
  SymbolizedStackBuilder *ssb = (struct SymbolizedStackBuilder *)ctx;
  if (ssb->tail) {
    ssb->tail->next = SymbolizedStack::New(ssb->addr);
    ssb->tail = ssb->tail->next;
  } else {
    ssb->head = ssb->tail = SymbolizedStack::New(ssb->addr);
  }
  AddressInfo *info = &ssb->tail->info;
  if (function_name) {
    info->function = internal_strdup(function_name);
  }
  if (file) {
    info->file = internal_strdup(file);
  }
  info->line = line;
  info->column = column;
}

// Symbolizer makes lots of intercepted calls. If we try to process them,
// at best it will cause deadlocks on internal mutexes.
struct SymbolizerScope : ScopedIgnoreInterceptors {
#if !SANITIZER_GO
  SymbolizerScope() {
    cur_thread()->in_symbolizer++;
  }
  ~SymbolizerScope() {
    cur_thread()->in_symbolizer--;
  }
#endif
};

SymbolizedStack *SymbolizeCode(uptr addr) {
  SymbolizerScope scope;
  // Check if PC comes from non-native land.
  if (!(addr & kExternalPCBit))
    return Symbolizer::GetOrInit()->SymbolizePC(addr);
  SymbolizedStackBuilder ssb = {nullptr, nullptr, addr};
  __tsan_symbolize_external_ex(addr, AddFrame, &ssb);
  if (ssb.head)
    return ssb.head;
  // Legacy code: remove along with the declaration above
  // once all clients using this API are gone.
  // Declare static to not consume too much stack space.
  // We symbolize reports in a single thread, so this is fine.
  static char func_buf[1024];
  static char file_buf[1024];
  int line, col;
  SymbolizedStack* frame = SymbolizedStack::New(addr);
  if (__tsan_symbolize_external(addr, func_buf, sizeof(func_buf), file_buf,
                                sizeof(file_buf), &line, &col)) {
    frame->info.function = internal_strdup(func_buf);
    frame->info.file = internal_strdup(file_buf);
    frame->info.line = line;
    frame->info.column = col;
  }
  return frame;
}

bool SymbolizeData(uptr addr, ReportLocation* loc) {
  SymbolizerScope scope;
  if (!Symbolizer::GetOrInit()->SymbolizeData(addr, &loc->global))
    return false;
  loc->type = ReportLocationGlobal;
  return true;
}

void SymbolizerFlush() {
  SymbolizerScope scope;
  Symbolizer::GetOrInit()->Flush();
}

}  // namespace __tsan
