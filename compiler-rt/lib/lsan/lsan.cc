//=-- lsan.cc -------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Standalone LSan RTL.
//
//===----------------------------------------------------------------------===//

#include "lsan.h"

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "lsan_allocator.h"
#include "lsan_common.h"
#include "lsan_thread.h"

bool lsan_inited;
bool lsan_init_is_running;

namespace __lsan {

static void InitializeCommonFlags() {
  CommonFlags *cf = common_flags();
  SetCommonFlagDefaults();
  cf->external_symbolizer_path = GetEnv("LSAN_SYMBOLIZER_PATH");
  cf->malloc_context_size = 30;
  cf->detect_leaks = true;

  ParseCommonFlagsFromString(GetEnv("LSAN_OPTIONS"));
}

}  // namespace __lsan

using namespace __lsan;  // NOLINT

extern "C" void __lsan_init() {
  CHECK(!lsan_init_is_running);
  if (lsan_inited)
    return;
  lsan_init_is_running = true;
  SanitizerToolName = "LeakSanitizer";
  InitializeCommonFlags();
  InitializeAllocator();
  InitTlsSize();
  InitializeInterceptors();
  InitializeThreadRegistry();
  u32 tid = ThreadCreate(0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(tid, GetTid());
  SetCurrentThread(tid);

  // Start symbolizer process if necessary.
  if (common_flags()->symbolize) {
    Symbolizer::Init(common_flags()->external_symbolizer_path);
  } else {
    Symbolizer::Disable();
  }

  InitCommonLsan();
  if (common_flags()->detect_leaks && common_flags()->leak_check_at_exit)
    Atexit(DoLeakCheck);
  lsan_inited = true;
  lsan_init_is_running = false;
}

