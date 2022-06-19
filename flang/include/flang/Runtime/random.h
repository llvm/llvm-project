//===-- include/flang/Runtime/random.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Intrinsic subroutines RANDOM_INIT, RANDOM_NUMBER, and RANDOM_SEED.

#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {
class Descriptor;
extern "C" {

void RTNAME(RandomInit)(bool repeatable, bool image_distinct);

void RTNAME(RandomNumber)(
    const Descriptor &harvest, const char *source, int line);

// Subroutine RANDOM_SEED can be called with at most one of its optional
// arguments; they each (plus the default case) map to these entry points.
void RTNAME(RandomSeedSize)(const Descriptor &, const char *source, int line);
void RTNAME(RandomSeedPut)(const Descriptor &, const char *source, int line);
void RTNAME(RandomSeedGet)(const Descriptor &, const char *source, int line);
void RTNAME(RandomSeedDefaultPut)();
} // extern "C"
} // namespace Fortran::runtime
