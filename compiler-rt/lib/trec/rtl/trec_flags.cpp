//===-- trec_flags.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

#include "trec_flags.h"

#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "trec_rtl.h"
#include "ubsan/ubsan_flags.h"

namespace __trec {

// Can be overriden in frontend.
#ifdef TREC_EXTERNAL_HOOKS
extern "C" const char *__trec_default_options();
#else
SANITIZER_WEAK_DEFAULT_IMPL
const char *__trec_default_options() { return ""; }
#endif

void Flags::SetDefaults() {
#define TREC_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "trec_flags.inc"
#undef TREC_FLAG
}

void RegisterTrecFlags(FlagParser *parser, Flags *f) {
#define TREC_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "trec_flags.inc"
#undef TREC_FLAG
}

void InitializeFlags(Flags *f, const char *env, const char *env_option_name) {
  SetCommonFlagsDefaults();
  {
    // Override some common flags defaults.
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    // gyq: disable deadlock detection
    cf.detect_deadlocks = false;
    if (SANITIZER_GO) {
      // Does not work as expected for Go: runtime handles SIGABRT and crashes.
      cf.abort_on_error = false;
    }
    cf.print_suppressions = false;
    cf.stack_trace_format = "    #%n %f %S %M";
    cf.exitcode = 0;
    cf.intercept_tls_get_addr = true;
    OverrideCommonFlags(cf);
  }

  f->SetDefaults();

  FlagParser parser;
  RegisterTrecFlags(&parser, f);
  RegisterCommonFlags(&parser);

  // Override from command line.
  parser.ParseString(env, env_option_name);

  InitializeCommonFlags();

  if (Verbosity())
    ReportUnrecognizedFlags();

  if (common_flags()->help)
    parser.PrintFlagDescriptions();
}

}  // namespace __trec
