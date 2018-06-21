//===-- tsan_symbolize.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

void EnterSymbolizer() {
  ThreadState *thr = cur_thread();
  CHECK(!thr->in_symbolizer);
  thr->in_symbolizer = true;
  thr->ignore_interceptors++;
}

void ExitSymbolizer() {
  ThreadState *thr = cur_thread();
  CHECK(thr->in_symbolizer);
  thr->in_symbolizer = false;
  thr->ignore_interceptors--;
}

// May be overriden by JIT/JAVA/etc,
// whatever produces PCs marked with kExternalPCBit.
SANITIZER_WEAK_DEFAULT_IMPL
bool __tsan_symbolize_external(uptr pc, char *func_buf, uptr func_siz,
                               char *file_buf, uptr file_siz, int *line,
                               int *col) {
  return false;
}

SymbolizedStack *SymbolizeCode(uptr addr) {
  // Check if PC comes from non-native land.
  if (addr & kExternalPCBit) {
    // Declare static to not consume too much stack space.
    // We symbolize reports in a single thread, so this is fine.
    static char func_buf[1024];
    static char file_buf[1024];
    int line, col;
    SymbolizedStack *frame = SymbolizedStack::New(addr);
    if (__tsan_symbolize_external(addr, func_buf, sizeof(func_buf), file_buf,
                                  sizeof(file_buf), &line, &col)) {
      frame->info.function = internal_strdup(func_buf);
      frame->info.file = internal_strdup(file_buf);
      frame->info.line = line;
      frame->info.column = col;
    }
    return frame;
  }
  return Symbolizer::GetOrInit()->SymbolizePC(addr);
}

ReportLocation *SymbolizeData(uptr addr) {
  DataInfo info;
  if (!Symbolizer::GetOrInit()->SymbolizeData(addr, &info))
    return 0;
  ReportLocation *ent = ReportLocation::New(ReportLocationGlobal);
  internal_memcpy(&ent->global, &info, sizeof(info));
  return ent;
}

void SymbolizeFlush() {
  Symbolizer::GetOrInit()->Flush();
}

}  // namespace __tsan
