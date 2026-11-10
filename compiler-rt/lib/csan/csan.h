//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Internal declarations for the host ConcurrencySanitizer runtime.
///
//===----------------------------------------------------------------------===//

#ifndef CSAN_H
#define CSAN_H

#include "csan_defs.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __csan {

using __sanitizer::s32;
using __sanitizer::u32;
using __sanitizer::u64;
using __sanitizer::uptr;

struct Flags {
#define CSAN_FLAG(Type, Name, DefaultValue, Description) Type Name;
#include "csan_flags.inc"
#undef CSAN_FLAG
  void SetDefaults();
};

extern Flags flags_data;
inline Flags *flags() { return &flags_data; }

void InitializeFlags();

enum ValueChange { kValueChangeMaybe, kValueChangeFalse, kValueChangeTrue };

static constexpr u32 kMaxStackFrames = 64;

struct AccessInfo {
  const volatile void *ptr;
  uptr size;
  int access_type;
  u32 tid;
  uptr pc;
  uptr bp;
};

void RecordDataRace();
void ReportKnownOrigin(const AccessInfo &AI, ValueChange VC, uptr PeerPC,
                       int PeerAccess, uptr PeerSize, u64 Old, u64 New);
void ReportUnknownOrigin(const AccessInfo &AI, u64 Old, u64 New);

} // namespace __csan

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE const char *
__csan_default_options();
}

#endif // CSAN_H
