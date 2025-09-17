//===-- tracing.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TRACING_H_
#define SCUDO_TRACING_H_

#if defined(SCUDO_ENABLE_TRACING)

// This file must include definitions for all of the functions below.
#include "custom_scudo_tracing.h"

#else

// Should start a trace in the given scope, and end the trace when going out of
// scope.
#define SCUDO_SCOPED_TRACE(Name)

// Create a trace name for the call to releaseToOS.
static inline const char *GetReleaseToOSTraceName(scudo::ReleaseToOS) {
  return nullptr;
}

// Create a trace name for the call to releaseToOSMaybe in the primary.
static inline const char *
GetPrimaryReleaseToOSMaybeTraceName(scudo::ReleaseToOS) {
  return nullptr;
}

static inline const char *GetPrimaryReleaseToOSTraceName(scudo::ReleaseToOS) {
  return nullptr;
}

// Create a trace name for the call to releaseToOS in the secondary.
static inline const char *GetSecondaryReleaseToOSTraceName(scudo::ReleaseToOS) {
  return nullptr;
}

// Create a trace name for the call to releaseOlderThan in the secondary.
static inline const char *GetSecondaryReleaseOlderThanTraceName() {
  return nullptr;
}

#endif

#endif // SCUDO_TRACING_H_
