//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Initialization and runtime flags for the device address sanitizer.
///
//===----------------------------------------------------------------------===//

#include "dasan.h"

#include <dlfcn.h>

#include "dasan_flags.h"
#include "dasan_report.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_offload.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __sanitizer;

namespace __dasan {

Mutex DasanMutex;
Flags DasanFlags;

void Flags::SetDefaults() {
#define DASAN_FLAG(Type, Name, Default, Description) Name = Default;
#include "dasan_flags.inc"
#undef DASAN_FLAG
}

static void RegisterDasanFlags(FlagParser* Parser, Flags* F) {
#define DASAN_FLAG(Type, Name, Default, Description) \
  RegisterFlag(Parser, #Name, Description, &F->Name);
#include "dasan_flags.inc"
#undef DASAN_FLAG
}

void InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags Cf;
    Cf.CopyFrom(*common_flags());
    Cf.external_symbolizer_path = GetEnv("DASAN_SYMBOLIZER_PATH");
    Cf.malloc_context_size = 30;
    OverrideCommonFlags(Cf);
  }

  Flags* F = flags();
  F->SetDefaults();

  FlagParser Parser;
  RegisterDasanFlags(&Parser, F);
  RegisterCommonFlags(&Parser);

  Parser.ParseString(__dasan_default_options());
  Parser.ParseStringFromEnv("DASAN_OPTIONS");

  InitializeCommonFlags();

  if (Verbosity())
    ReportUnrecognizedFlags();

  if (common_flags()->help)
    Parser.PrintFlagDescriptions();
}

static StaticSpinMutex InitMutex;
static atomic_uint8_t Initialized;

enum {
  DASAN_RT_VERSION_UNDEFINED = 0,
  DASAN_RT_VERSION_DYNAMIC,
  DASAN_RT_VERSION_STATIC,
};

// One symbol for every copy of the interceptors.
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE int __dasan_rt_version;
}

// This copy lives in libclang_rt.dasan.so. The static copy lives in the
// executable, so its address does not.
static bool ThisCopyIsDynamic() {
  Dl_info Info = {};
  if (!dladdr(reinterpret_cast<void*>(&ThisCopyIsDynamic), &Info) ||
      !Info.dli_fname)
    return false;
  return internal_strstr(Info.dli_fname, "libclang_rt.dasan");
}

static void ReportIncompatibleRuntime() {
  Report(
      "ERROR: %s: incompatible runtimes. Link with -shared-libsan so the "
      "executable uses the same runtime as the sanitized library.\n",
      SanitizerToolName);
  Die();
}

// DASan rejects a static runtime in the executable once the shared runtime is
// loaded. The interceptor would bind to the other copy and call itself.
static void CheckIncompatibleRuntime() {
  if (ThisCopyIsDynamic()) {
    if (__dasan_rt_version == DASAN_RT_VERSION_UNDEFINED)
      __dasan_rt_version = DASAN_RT_VERSION_DYNAMIC;
    else if (__dasan_rt_version != DASAN_RT_VERSION_DYNAMIC)
      ReportIncompatibleRuntime();
    return;
  }

  if (__dasan_rt_version == DASAN_RT_VERSION_UNDEFINED) {
    MemoryMappingLayout Layout(/*cache_enabled=*/true);
    char Name[kMaxPathLength];
    MemoryMappedSegment Segment(Name, sizeof(Name));
    while (Layout.Next(&Segment)) {
      if (internal_strstr(Name, "libclang_rt.dasan"))
        ReportIncompatibleRuntime();
    }
    __dasan_rt_version = DASAN_RT_VERSION_STATIC;
    return;
  }
  if (__dasan_rt_version != DASAN_RT_VERSION_STATIC)
    ReportIncompatibleRuntime();
}

void Initialize() {
  if (LIKELY(atomic_load(&Initialized, memory_order_acquire)))
    return;

  SpinMutexLock L(&InitMutex);
  if (atomic_load(&Initialized, memory_order_relaxed))
    return;

  SanitizerToolName = "DeviceAddressSanitizer";
  InitializeFlags();
  CheckIncompatibleRuntime();

  Offload::Get().RegisterHandler(HandleDeviceReport);
  Atexit([] { Offload::Get().UntrackImages(); });
  AddDieCallback([] { Offload::Get().UntrackImages(); });

  VReport(1, "%s: initialized\n", SanitizerToolName);
  atomic_store(&Initialized, 1, memory_order_release);
}

}  // namespace __dasan

namespace {

// pthread_getattr_np reads /proc. The bounds last for the life of the thread.
struct StackBounds {
  uptr Top;
  uptr Bottom;
  bool Ready;
};
THREADLOCAL StackBounds CachedBounds;
}  // namespace

void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void* context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  if (!CachedBounds.Ready) {
    GetThreadStackTopAndBottom(false, &CachedBounds.Top, &CachedBounds.Bottom);
    CachedBounds.Ready = true;
  }
  bool Fast = StackTrace::WillUseFastUnwind(request_fast);
  Unwind(max_depth, pc, bp, context, CachedBounds.Top, CachedBounds.Bottom,
         Fast);
}

SANITIZER_INTERFACE_WEAK_DEF(const char*, __dasan_default_options, void) {
  return "";
}

extern "C" void __dasan_init() { __dasan::Initialize(); }
