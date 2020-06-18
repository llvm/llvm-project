/*===- InstrProfilingInternal.c - Support library for PGO instrumentation -===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#if !defined(__Fuchsia__)

#include "InstrProfilingInternal.h"

static unsigned ProfileDumped = 0;

COMPILER_RT_VISIBILITY unsigned lprofProfileDumped() {
  return ProfileDumped;
}

COMPILER_RT_VISIBILITY void lprofSetProfileDumped(unsigned Value) {
  ProfileDumped = Value;
}

static unsigned RuntimeCounterRelocation = 0;

COMPILER_RT_VISIBILITY unsigned lprofRuntimeCounterRelocation(void) {
  return RuntimeCounterRelocation;
}

COMPILER_RT_VISIBILITY void lprofSetRuntimeCounterRelocation(unsigned Value) {
  RuntimeCounterRelocation = Value;
}

#endif
