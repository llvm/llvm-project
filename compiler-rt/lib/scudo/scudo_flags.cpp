//===-- scudo_flags.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Hardened Allocator flag parsing logic.
///
//===----------------------------------------------------------------------===//

#include "scudo_flags.h"
#include "scudo_utils.h"

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_flag_parser.h"

namespace __scudo {

Flags scudo_flags_dont_use_directly;  // use via flags().

void Flags::setDefaults() {
#define SCUDO_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "scudo_flags.inc"
#undef SCUDO_FLAG
}

static void RegisterScudoFlags(FlagParser *parser, Flags *f) {
#define SCUDO_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "scudo_flags.inc"
#undef SCUDO_FLAG
}

void initFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.exitcode = 1;
    OverrideCommonFlags(cf);
  }
  Flags *f = getFlags();
  f->setDefaults();

  FlagParser scudo_parser;
  RegisterScudoFlags(&scudo_parser, f);
  RegisterCommonFlags(&scudo_parser);

  scudo_parser.ParseString(GetEnv("SCUDO_OPTIONS"));

  InitializeCommonFlags();

  // Sanity checks and default settings for the Quarantine parameters.

  if (f->QuarantineSizeMb < 0) {
    const int DefaultQuarantineSizeMb = 64;
    f->QuarantineSizeMb = DefaultQuarantineSizeMb;
  }
  // We enforce an upper limit for the quarantine size of 4Gb.
  if (f->QuarantineSizeMb > (4 * 1024)) {
    dieWithMessage("ERROR: the quarantine size is too large\n");
  }
  if (f->ThreadLocalQuarantineSizeKb < 0) {
    const int DefaultThreadLocalQuarantineSizeKb = 1024;
    f->ThreadLocalQuarantineSizeKb = DefaultThreadLocalQuarantineSizeKb;
  }
  // And an upper limit of 128Mb for the thread quarantine cache.
  if (f->ThreadLocalQuarantineSizeKb > (128 * 1024)) {
    dieWithMessage("ERROR: the per thread quarantine cache size is too "
                   "large\n");
  }
}

Flags *getFlags() {
  return &scudo_flags_dont_use_directly;
}

}
