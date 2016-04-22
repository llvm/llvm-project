//===-- esan.cpp ----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Main file (entry points) for the Esan run-time.
//===----------------------------------------------------------------------===//

#include "esan.h"
#include "esan_interface_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"

// See comment below.
extern "C" {
extern void __cxa_atexit(void (*function)(void));
}

namespace __esan {

bool EsanIsInitialized;
ToolType WhichTool;

static const char EsanOptsEnv[] = "ESAN_OPTIONS";

// We are combining multiple performance tuning tools under the umbrella of
// one EfficiencySanitizer super-tool.  Most of our tools have very similar
// memory access instrumentation, shadow memory mapping, libc interception,
// etc., and there is typically more shared code than distinct code.
//
// We are not willing to dispatch on tool dynamically in our fastpath
// instrumentation: thus, which tool to use is a static option selected
// at compile time and passed to __esan_init().
//
// We are willing to pay the overhead of tool dispatch in the slowpath to more
// easily share code.  We expect to only come here rarely.
// If this becomes a performance hit, we can add separate interface
// routines for each subtool (e.g., __esan_cache_frag_aligned_load_4).
// But for libc interceptors, we'll have to do one of the following:
// A) Add multiple-include support to sanitizer_common_interceptors.inc,
//    instantiate it separately for each tool, and call the selected
//    tool's intercept setup code.
// B) Build separate static runtime libraries, one for each tool.
// C) Completely split the tools into separate sanitizers.

void processRangeAccess(uptr PC, uptr Addr, int Size, bool IsWrite) {
  VPrintf(3, "in esan::%s %p: %c %p %d\n", __FUNCTION__, PC,
          IsWrite ? 'w' : 'r', Addr, Size);
  if (WhichTool == ESAN_CacheFrag) {
    // TODO(bruening): add shadow mapping and update shadow bits here.
    // We'll move this to cache_frag.cpp once we have something.
  }
}

static void initializeFlags() {
  // Once we add our own flags we'll parse them here.
  // For now the common ones are sufficient.
  FlagParser Parser;
  SetCommonFlagsDefaults();
  RegisterCommonFlags(&Parser);
  Parser.ParseString(GetEnv(EsanOptsEnv));
  InitializeCommonFlags();
  if (Verbosity())
    ReportUnrecognizedFlags();
  if (common_flags()->help)
    Parser.PrintFlagDescriptions();
  __sanitizer_set_report_path(common_flags()->log_path);
}

void initializeLibrary(ToolType Tool) {
  // We assume there is only one thread during init.
  if (EsanIsInitialized) {
    CHECK(Tool == WhichTool);
    return;
  }
  WhichTool = Tool;
  SanitizerToolName = "EfficiencySanitizer";
  initializeFlags();

  // Intercepting libc _exit or exit via COMMON_INTERCEPTOR_ON_EXIT only
  // finalizes on an explicit exit call by the app.  To handle a normal
  // exit we register an atexit handler.
  ::__cxa_atexit((void (*)())finalizeLibrary);

  VPrintf(1, "in esan::%s\n", __FUNCTION__);
  if (WhichTool != ESAN_CacheFrag) {
    Printf("ERROR: unknown tool %d requested\n", WhichTool);
    Die();
  }

  EsanIsInitialized = true;
}

int finalizeLibrary() {
  VPrintf(1, "in esan::%s\n", __FUNCTION__);
  if (WhichTool == ESAN_CacheFrag) {
    // FIXME NYI: we need to add sampling + callstack gathering and have a
    // strategy for how to generate a final report.
    // We'll move this to cache_frag.cpp once we have something.
    Report("%s is not finished: nothing yet to report\n", SanitizerToolName);
  }
  return 0;
}

} // namespace __esan
