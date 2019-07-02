//===-- options_parser.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/optional/options_parser.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gwp_asan/options.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"

namespace gwp_asan {
namespace options {
namespace {
void registerGwpAsanFlags(__sanitizer::FlagParser *parser, Options *o) {
#define GWP_ASAN_OPTION(Type, Name, DefaultValue, Description)                 \
  RegisterFlag(parser, #Name, Description, &o->Name);
#include "gwp_asan/options.inc"
#undef GWP_ASAN_OPTION
}

const char *getCompileDefinitionGwpAsanDefaultOptions() {
#ifdef GWP_ASAN_DEFAULT_OPTIONS
  return SANITIZER_STRINGIFY(GWP_ASAN_DEFAULT_OPTIONS);
#else
  return "";
#endif
}

const char *getGwpAsanDefaultOptions() {
  return (__gwp_asan_default_options) ? __gwp_asan_default_options() : "";
}

Options *getOptionsInternal() {
  static Options GwpAsanFlags;
  return &GwpAsanFlags;
}
} // anonymous namespace

void initOptions() {
  __sanitizer::SetCommonFlagsDefaults();

  Options *o = getOptionsInternal();
  o->setDefaults();

  __sanitizer::FlagParser Parser;
  registerGwpAsanFlags(&Parser, o);

  // Override from compile definition.
  Parser.ParseString(getCompileDefinitionGwpAsanDefaultOptions());

  // Override from user-specified string.
  Parser.ParseString(getGwpAsanDefaultOptions());

  // Override from environment.
  Parser.ParseString(__sanitizer::GetEnv("GWP_ASAN_OPTIONS"));

  __sanitizer::InitializeCommonFlags();
  if (__sanitizer::Verbosity())
    __sanitizer::ReportUnrecognizedFlags();

  if (!o->Enabled)
    return;

  // Sanity checks for the parameters.
  if (o->MaxSimultaneousAllocations <= 0) {
    __sanitizer::Printf("GWP-ASan ERROR: MaxSimultaneousAllocations must be > "
                        "0 when GWP-ASan is enabled.\n");
    exit(EXIT_FAILURE);
  }

  if (o->SampleRate < 1) {
    __sanitizer::Printf(
        "GWP-ASan ERROR: SampleRate must be > 0 when GWP-ASan is enabled.\n");
    exit(EXIT_FAILURE);
  }

  o->Printf = __sanitizer::Printf;
}

Options &getOptions() { return *getOptionsInternal(); }

} // namespace options
} // namespace gwp_asan
