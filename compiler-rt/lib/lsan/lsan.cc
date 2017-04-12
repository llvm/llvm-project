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
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "lsan_allocator.h"
#include "lsan_common.h"
#include "lsan_thread.h"

bool lsan_inited;
bool lsan_init_is_running;

namespace __lsan {

///// Interface to the common LSan module. /////
bool WordIsPoisoned(uptr addr) {
  return false;
}

}  // namespace __lsan

using namespace __lsan;  // NOLINT

static void InitializeFlags() {
  // Set all the default values.
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.external_symbolizer_path = GetEnv("LSAN_SYMBOLIZER_PATH");
    cf.malloc_context_size = 30;
    cf.intercept_tls_get_addr = true;
    cf.detect_leaks = true;
    cf.exitcode = 23;
    OverrideCommonFlags(cf);
  }

  Flags *f = flags();
  f->SetDefaults();

  FlagParser parser;
  RegisterLsanFlags(&parser, f);
  RegisterCommonFlags(&parser);

  parser.ParseString(GetEnv("LSAN_OPTIONS"));

  SetVerbosity(common_flags()->verbosity);

  if (Verbosity()) ReportUnrecognizedFlags();

  if (common_flags()->help) parser.PrintFlagDescriptions();
}

extern "C" void __lsan_init() {
  CHECK(!lsan_init_is_running);
  if (lsan_inited)
    return;
  lsan_init_is_running = true;
  SanitizerToolName = "LeakSanitizer";
  CacheBinaryName();
  AvoidCVE_2016_2143();
  InitializeFlags();
  InitCommonLsan();
  InitializeAllocator();
  ReplaceSystemMalloc();
  InitTlsSize();
  InitializeInterceptors();
  InitializeThreadRegistry();
  u32 tid = ThreadCreate(0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(tid, GetTid());
  SetCurrentThread(tid);

  if (common_flags()->detect_leaks && common_flags()->leak_check_at_exit)
    Atexit(DoLeakCheck);

  InitializeCoverage(common_flags()->coverage, common_flags()->coverage_dir);

  lsan_inited = true;
  lsan_init_is_running = false;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_print_stack_trace() {
  GET_STACK_TRACE_FATAL;
  stack.Print();
}
