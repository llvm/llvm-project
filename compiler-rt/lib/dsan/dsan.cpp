//=-- dsan.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Standalone DSan RTL.
//
//===----------------------------------------------------------------------===//

#include "dsan.h"

#include "dsan_allocator.h"
#include "dsan_common.h"
#include "dsan_thread.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_interface_internal.h"

bool dsan_inited;
bool dsan_init_is_running;

namespace __dsan {

///// Interface to the common DSan module. /////
bool WordIsPoisoned(uptr addr) { return false; }

}  // namespace __dsan

void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void* context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  using namespace __dsan;
  uptr stack_top = 0, stack_bottom = 0;
  if (ThreadContextDsanBase* t = GetCurrentThread()) {
    stack_top = t->stack_end();
    stack_bottom = t->stack_begin();
  }
  if (SANITIZER_MIPS && !IsValidFrame(bp, stack_top, stack_bottom))
    return;
  bool fast = StackTrace::WillUseFastUnwind(request_fast);
  Unwind(max_depth, pc, bp, context, stack_top, stack_bottom, fast);
}

using namespace __dsan;

static void InitializeFlags() {
  // Set all the default values.
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.external_symbolizer_path = GetEnv("DSAN_SYMBOLIZER_PATH");
    cf.malloc_context_size = 30;
    cf.intercept_tls_get_addr = true;
    cf.detect_leaks = false;
    cf.exitcode = 77;
    OverrideCommonFlags(cf);
  }

  FlagParser parser;
  RegisterCommonFlags(&parser);

  // Override from user-specified string.
  const char* dsan_default_options = __dsan_default_options();
  parser.ParseString(dsan_default_options);
  parser.ParseStringFromEnv("DSAN_OPTIONS");

  InitializeCommonFlags();

  if (Verbosity())
    ReportUnrecognizedFlags();

  if (common_flags()->help)
    parser.PrintFlagDescriptions();

  __sanitizer_set_report_path(common_flags()->log_path);
}

extern "C" void __dsan_init() {
  CHECK(!dsan_init_is_running);
  if (dsan_inited)
    return;
  dsan_init_is_running = true;
  SanitizerToolName = "DoubleFreeSanitizer";
  CacheBinaryName();
  AvoidCVE_2016_2143();
  InitializeFlags();
  InitializePlatformEarly();
  InitCommonDsan();
  InitializeAllocator();
  ReplaceSystemMalloc();
  InitializeInterceptors();
  InitializeThreads();
  InstallDeadlySignalHandlers(DsanOnDeadlySignal);
  InitializeMainThread();
  InstallAtForkHandler();

  InitializeCoverage(common_flags()->coverage, common_flags()->coverage_dir);

  dsan_inited = true;
  dsan_init_is_running = false;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_print_stack_trace() {
  GET_STACK_TRACE_FATAL;
  stack.Print();
}
