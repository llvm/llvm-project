//===-- asan_flags.cc -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan flag parsing logic.
//===----------------------------------------------------------------------===//

#include "asan_activation.h"
#include "asan_flags.h"
#include "asan_interface_internal.h"
#include "asan_stack.h"
#include "lsan/lsan_common.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "ubsan/ubsan_flags.h"
#include "ubsan/ubsan_platform.h"

namespace __asan {

Flags asan_flags_dont_use_directly;  // use via flags().

static const char *MaybeCallAsanDefaultOptions() {
  return (&__asan_default_options) ? __asan_default_options() : "";
}

static const char *MaybeUseAsanDefaultOptionsCompileDefinition() {
#ifdef ASAN_DEFAULT_OPTIONS
// Stringize the macro value.
# define ASAN_STRINGIZE(x) #x
# define ASAN_STRINGIZE_OPTIONS(options) ASAN_STRINGIZE(options)
  return ASAN_STRINGIZE_OPTIONS(ASAN_DEFAULT_OPTIONS);
#else
  return "";
#endif
}

void Flags::SetDefaults() {
#define ASAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "asan_flags.inc"
#undef ASAN_FLAG
}

static void RegisterAsanFlags(FlagParser *parser, Flags *f) {
#define ASAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "asan_flags.inc"
#undef ASAN_FLAG
}

void InitializeFlags() {
  // Set the default values and prepare for parsing ASan and common flags.
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.detect_leaks = CAN_SANITIZE_LEAKS;
    cf.external_symbolizer_path = GetEnv("ASAN_SYMBOLIZER_PATH");
    cf.malloc_context_size = kDefaultMallocContextSize;
    cf.intercept_tls_get_addr = true;
    cf.exitcode = 1;
    OverrideCommonFlags(cf);
  }
  Flags *f = flags();
  f->SetDefaults();

  FlagParser asan_parser;
  RegisterAsanFlags(&asan_parser, f);
  RegisterCommonFlags(&asan_parser);

  // Set the default values and prepare for parsing LSan and UBSan flags
  // (which can also overwrite common flags).
#if CAN_SANITIZE_LEAKS
  __lsan::Flags *lf = __lsan::flags();
  lf->SetDefaults();

  FlagParser lsan_parser;
  __lsan::RegisterLsanFlags(&lsan_parser, lf);
  RegisterCommonFlags(&lsan_parser);
#endif

#if CAN_SANITIZE_UB
  __ubsan::Flags *uf = __ubsan::flags();
  uf->SetDefaults();

  FlagParser ubsan_parser;
  __ubsan::RegisterUbsanFlags(&ubsan_parser, uf);
  RegisterCommonFlags(&ubsan_parser);
#endif

  // Override from ASan compile definition.
  const char *asan_compile_def = MaybeUseAsanDefaultOptionsCompileDefinition();
  asan_parser.ParseString(asan_compile_def);

  // Override from user-specified string.
  const char *asan_default_options = MaybeCallAsanDefaultOptions();
  asan_parser.ParseString(asan_default_options);
#if CAN_SANITIZE_UB
  const char *ubsan_default_options = __ubsan::MaybeCallUbsanDefaultOptions();
  ubsan_parser.ParseString(ubsan_default_options);
#endif

  // Override from command line.
  asan_parser.ParseString(GetEnv("ASAN_OPTIONS"));
#if CAN_SANITIZE_LEAKS
  lsan_parser.ParseString(GetEnv("LSAN_OPTIONS"));
#endif
#if CAN_SANITIZE_UB
  ubsan_parser.ParseString(GetEnv("UBSAN_OPTIONS"));
#endif

  InitializeCommonFlags();

  // TODO(eugenis): dump all flags at verbosity>=2?
  if (Verbosity()) ReportUnrecognizedFlags();

  if (common_flags()->help) {
    // TODO(samsonov): print all of the flags (ASan, LSan, common).
    asan_parser.PrintFlagDescriptions();
  }

  // Flag validation:
  if (!CAN_SANITIZE_LEAKS && common_flags()->detect_leaks) {
    Report("%s: detect_leaks is not supported on this platform.\n",
           SanitizerToolName);
    Die();
  }
  // Make "strict_init_order" imply "check_initialization_order".
  // TODO(samsonov): Use a single runtime flag for an init-order checker.
  if (f->strict_init_order) {
    f->check_initialization_order = true;
  }
  CHECK_LE((uptr)common_flags()->malloc_context_size, kStackTraceMax);
  CHECK_LE(f->min_uar_stack_size_log, f->max_uar_stack_size_log);
  CHECK_GE(f->redzone, 16);
  CHECK_GE(f->max_redzone, f->redzone);
  CHECK_LE(f->max_redzone, 2048);
  CHECK(IsPowerOfTwo(f->redzone));
  CHECK(IsPowerOfTwo(f->max_redzone));

  // quarantine_size is deprecated but we still honor it.
  // quarantine_size can not be used together with quarantine_size_mb.
  if (f->quarantine_size >= 0 && f->quarantine_size_mb >= 0) {
    Report("%s: please use either 'quarantine_size' (deprecated) or "
           "quarantine_size_mb, but not both\n", SanitizerToolName);
    Die();
  }
  if (f->quarantine_size >= 0)
    f->quarantine_size_mb = f->quarantine_size >> 20;
  if (f->quarantine_size_mb < 0) {
    const int kDefaultQuarantineSizeMb =
        (ASAN_LOW_MEMORY) ? 1UL << 6 : 1UL << 8;
    f->quarantine_size_mb = kDefaultQuarantineSizeMb;
  }
  if (!f->replace_str && common_flags()->intercept_strlen) {
    Report("WARNING: strlen interceptor is enabled even though replace_str=0. "
           "Use intercept_strlen=0 to disable it.");
  }
  if (!f->replace_str && common_flags()->intercept_strchr) {
    Report("WARNING: strchr* interceptors are enabled even though "
           "replace_str=0. Use intercept_strchr=0 to disable them.");
  }
}

}  // namespace __asan

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char* __asan_default_options() { return ""; }
}  // extern "C"
#endif
