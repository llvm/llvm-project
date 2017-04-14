//===-- ubsan_init.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Initialization of UBSan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_platform.h"
#if CAN_SANITIZE_UB
#include "ubsan_diag.h"
#include "ubsan_init.h"
#include "ubsan_flags.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __ubsan;

const char *__ubsan::GetSanititizerToolName() {
  return "UndefinedBehaviorSanitizer";
}

static enum {
  UBSAN_MODE_UNKNOWN = 0,
  UBSAN_MODE_STANDALONE,
  UBSAN_MODE_PLUGIN
} ubsan_mode;
static StaticSpinMutex ubsan_init_mu;

static void CommonInit() {
  InitializeSuppressions();
}

static void CommonStandaloneInit() {
  SanitizerToolName = GetSanititizerToolName();
  InitializeFlags();
  CacheBinaryName();
  __sanitizer_set_report_path(common_flags()->log_path);
  AndroidLogInit();
  InitializeCoverage(common_flags()->coverage, common_flags()->coverage_dir);
  CommonInit();
  ubsan_mode = UBSAN_MODE_STANDALONE;
}

void __ubsan::InitAsStandalone() {
  if (SANITIZER_CAN_USE_PREINIT_ARRAY) {
    CHECK_EQ(UBSAN_MODE_UNKNOWN, ubsan_mode);
    CommonStandaloneInit();
    return;
  }
  SpinMutexLock l(&ubsan_init_mu);
  CHECK_NE(UBSAN_MODE_PLUGIN, ubsan_mode);
  if (ubsan_mode == UBSAN_MODE_UNKNOWN)
    CommonStandaloneInit();
}

void __ubsan::InitAsStandaloneIfNecessary() {
  if (SANITIZER_CAN_USE_PREINIT_ARRAY) {
    CHECK_NE(UBSAN_MODE_UNKNOWN, ubsan_mode);
    return;
  }
  SpinMutexLock l(&ubsan_init_mu);
  if (ubsan_mode == UBSAN_MODE_UNKNOWN)
    CommonStandaloneInit();
}

void __ubsan::InitAsPlugin() {
#if !SANITIZER_CAN_USE_PREINIT_ARRAY
  SpinMutexLock l(&ubsan_init_mu);
#endif
  CHECK_EQ(UBSAN_MODE_UNKNOWN, ubsan_mode);
  CommonInit();
  ubsan_mode = UBSAN_MODE_PLUGIN;
}

#endif  // CAN_SANITIZE_UB
