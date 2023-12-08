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

// RANDOM_SEED may be called with a value for at most one of its three
// optional arguments.  Most calls map to an entry point for that value,
// or the entry point for no values.  If argument presence cannot be
// determined at compile time, function RandomSeed can be called to make
// the selection at run time.
void RTNAME(RandomSeedSize)(
    const Descriptor *size, const char *source, int line);
void RTNAME(RandomSeedPut)(const Descriptor *put, const char *source, int line);
void RTNAME(RandomSeedGet)(const Descriptor *get, const char *source, int line);
void RTNAME(RandomSeedDefaultPut)();
void RTNAME(RandomSeed)(const Descriptor *size, const Descriptor *put,
    const Descriptor *get, const char *source, int line);

} // extern "C"
} // namespace Fortran::runtime
