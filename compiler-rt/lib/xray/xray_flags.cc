//===-- xray_flags.cc -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// XRay flag parsing logic.
//===----------------------------------------------------------------------===//

#include "xray_flags.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "xray_defs.h"

using namespace __sanitizer;

namespace __xray {

Flags xray_flags_dont_use_directly; // use via flags().

void Flags::setDefaults() XRAY_NEVER_INSTRUMENT {
#define XRAY_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "xray_flags.inc"
#undef XRAY_FLAG
}

static void registerXRayFlags(FlagParser *P, Flags *F) XRAY_NEVER_INSTRUMENT {
#define XRAY_FLAG(Type, Name, DefaultValue, Description)                       \
  RegisterFlag(P, #Name, Description, &F->Name);
#include "xray_flags.inc"
#undef XRAY_FLAG
}

void initializeFlags() XRAY_NEVER_INSTRUMENT {
  SetCommonFlagsDefaults();
  auto *F = flags();
  F->setDefaults();

  FlagParser XRayParser;
  registerXRayFlags(&XRayParser, F);
  RegisterCommonFlags(&XRayParser);

  // Override from command line.
  XRayParser.ParseString(GetEnv("XRAY_OPTIONS"));

  InitializeCommonFlags();

  if (Verbosity())
    ReportUnrecognizedFlags();

  if (common_flags()->help) {
    XRayParser.PrintFlagDescriptions();
  }
}

} // namespace __xray
