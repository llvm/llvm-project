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

extern "C" SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
const char* __scudo_default_options();

namespace __scudo {

Flags ScudoFlags;  // Use via getFlags().

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

static const char *callGetScudoDefaultOptions() {
  return (&__scudo_default_options) ? __scudo_default_options() : "";
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

  FlagParser ScudoParser;
  RegisterScudoFlags(&ScudoParser, f);
  RegisterCommonFlags(&ScudoParser);

  // Override from user-specified string.
  const char *ScudoDefaultOptions = callGetScudoDefaultOptions();
  ScudoParser.ParseString(ScudoDefaultOptions);

  // Override from environment.
  ScudoParser.ParseString(GetEnv("SCUDO_OPTIONS"));

  InitializeCommonFlags();

  // Sanity checks and default settings for the Quarantine parameters.

  if (f->QuarantineSizeMb < 0) {
    const int DefaultQuarantineSizeMb = FIRST_32_SECOND_64(16, 64);
    f->QuarantineSizeMb = DefaultQuarantineSizeMb;
  }
  // We enforce an upper limit for the quarantine size of 4Gb.
  if (f->QuarantineSizeMb > (4 * 1024)) {
    dieWithMessage("ERROR: the quarantine size is too large\n");
  }
  if (f->ThreadLocalQuarantineSizeKb < 0) {
    const int DefaultThreadLocalQuarantineSizeKb =
        FIRST_32_SECOND_64(256, 1024);
    f->ThreadLocalQuarantineSizeKb = DefaultThreadLocalQuarantineSizeKb;
  }
  // And an upper limit of 128Mb for the thread quarantine cache.
  if (f->ThreadLocalQuarantineSizeKb > (128 * 1024)) {
    dieWithMessage("ERROR: the per thread quarantine cache size is too "
                   "large\n");
  }
  if (f->ThreadLocalQuarantineSizeKb == 0 && f->QuarantineSizeMb > 0) {
    dieWithMessage("ERROR: ThreadLocalQuarantineSizeKb can be set to 0 only "
                   "when QuarantineSizeMb is set to 0\n");
  }
}

Flags *getFlags() {
  return &ScudoFlags;
}

}  // namespace __scudo
