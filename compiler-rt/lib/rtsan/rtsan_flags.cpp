//===--- rtsan_flags.cpp - Realtime Sanitizer -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of RealtimeSanitizer.
//
//===----------------------------------------------------------------------===//

#include "rtsan/rtsan_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"

using namespace __sanitizer;
using namespace __rtsan;

Flags __rtsan::flags_data;

SANITIZER_INTERFACE_WEAK_DEF(const char *, __rtsan_default_options, void) {
  return "";
}

static void RegisterRtsanFlags(FlagParser *parser, Flags *f) {
#define RTSAN_FLAG(Type, Name, DefaultValue, Description)                      \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "rtsan_flags.inc"
#undef RTSAN_FLAG
}

void __rtsan::InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.exitcode = 43;
    cf.external_symbolizer_path = GetEnv("RTSAN_SYMBOLIZER_PATH");
    OverrideCommonFlags(cf);
  }

  FlagParser parser;
  RegisterRtsanFlags(&parser, &flags());
  RegisterCommonFlags(&parser);

  // Override from user-specified string.
  parser.ParseString(__rtsan_default_options());

  parser.ParseStringFromEnv("RTSAN_OPTIONS");

  InitializeCommonFlags();

  if (Verbosity())
    ReportUnrecognizedFlags();

  if (common_flags()->help)
    parser.PrintFlagDescriptions();
}
