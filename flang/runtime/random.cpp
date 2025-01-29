//===-- runtime/random.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the intrinsic subroutines RANDOM_INIT, RANDOM_NUMBER, and
// RANDOM_SEED.

#include "flang/Runtime/random.h"
#include "lock.h"
#include "random-templates.h"
#include "terminator.h"
#include "flang/Common/float128.h"
#include "flang/Common/leading-zero-bit-count.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <time.h>

namespace Fortran::runtime::random {

Lock lock;
Generator generator;
Fortran::common::optional<GeneratedWord> nextValue;

extern "C" {

void RTNAME(RandomInit)(bool repeatable, bool /*image_distinct*/) {
  // TODO: multiple images and image_distinct: add image number
  {
    CriticalSection critical{lock};
    if (repeatable) {
      generator.seed(0);
    } else {
#ifdef CLOCK_REALTIME
      timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      generator.seed(ts.tv_sec ^ ts.tv_nsec);
#else
      generator.seed(time(nullptr));
#endif
    }
  }
}

void RTNAME(RandomNumber)(
    const Descriptor &harvest, const char *source, int line) {
  Terminator terminator{source, line};
  auto typeCode{harvest.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      typeCode &&
          (typeCode->first == TypeCategory::Real ||
              typeCode->first == TypeCategory::Unsigned));
  int kind{typeCode->second};
  if (typeCode->first == TypeCategory::Real) {
    switch (kind) {
    // TODO: REAL (2 & 3)
    case 4:
      GenerateReal<CppTypeFor<TypeCategory::Real, 4>, 24>(harvest);
      return;
    case 8:
      GenerateReal<CppTypeFor<TypeCategory::Real, 8>, 53>(harvest);
      return;
    case 10:
      if constexpr (HasCppTypeFor<TypeCategory::Real, 10>) {
#if HAS_FLOAT80
        GenerateReal<CppTypeFor<TypeCategory::Real, 10>, 64>(harvest);
        return;
#endif
      }
      break;
    }
    terminator.Crash(
        "not yet implemented: intrinsic: REAL(KIND=%d) in RANDOM_NUMBER", kind);
  } else if (typeCode->first == TypeCategory::Unsigned) {
    switch (kind) {
    case 1:
      GenerateUnsigned<CppTypeFor<TypeCategory::Unsigned, 1>>(harvest);
      return;
    case 2:
      GenerateUnsigned<CppTypeFor<TypeCategory::Unsigned, 2>>(harvest);
      return;
    case 4:
      GenerateUnsigned<CppTypeFor<TypeCategory::Unsigned, 4>>(harvest);
      return;
    case 8:
      GenerateUnsigned<CppTypeFor<TypeCategory::Unsigned, 8>>(harvest);
      return;
#ifdef __SIZEOF_INT128__
    case 16:
      if constexpr (HasCppTypeFor<TypeCategory::Unsigned, 16>) {
        GenerateUnsigned<CppTypeFor<TypeCategory::Unsigned, 16>>(harvest);
        return;
      }
      break;
#endif
    }
    terminator.Crash(
        "not yet implemented: intrinsic: UNSIGNED(KIND=%d) in RANDOM_NUMBER",
        kind);
  }
}

void RTNAME(RandomSeedSize)(
    const Descriptor *size, const char *source, int line) {
  if (!size || !size->raw().base_addr) {
    RTNAME(RandomSeedDefaultPut)();
    return;
  }
  Terminator terminator{source, line};
  auto typeCode{size->type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      size->rank() == 0 && typeCode &&
          typeCode->first == TypeCategory::Integer);
  int sizeArg{typeCode->second};
  switch (sizeArg) {
  case 4:
    *size->OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>() = 1;
    break;
  case 8:
    *size->OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>() = 1;
    break;
  default:
    terminator.Crash(
        "not yet implemented: intrinsic: RANDOM_SEED(SIZE=): size %d\n",
        sizeArg);
  }
}

void RTNAME(RandomSeedPut)(
    const Descriptor *put, const char *source, int line) {
  if (!put || !put->raw().base_addr) {
    RTNAME(RandomSeedDefaultPut)();
    return;
  }
  Terminator terminator{source, line};
  auto typeCode{put->type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      put->rank() == 1 && typeCode &&
          typeCode->first == TypeCategory::Integer &&
          put->GetDimension(0).Extent() >= 1);
  int putArg{typeCode->second};
  GeneratedWord seed;
  switch (putArg) {
  case 4:
    seed = *put->OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>();
    break;
  case 8:
    seed = *put->OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>();
    break;
  default:
    terminator.Crash(
        "not yet implemented: intrinsic: RANDOM_SEED(PUT=): put %d\n", putArg);
  }
  {
    CriticalSection critical{lock};
    generator.seed(seed);
    nextValue = seed;
  }
}

void RTNAME(RandomSeedDefaultPut)() {
  // TODO: should this be time &/or image dependent?
  {
    CriticalSection critical{lock};
    generator.seed(0);
  }
}

void RTNAME(RandomSeedGet)(
    const Descriptor *get, const char *source, int line) {
  if (!get || !get->raw().base_addr) {
    RTNAME(RandomSeedDefaultPut)();
    return;
  }
  Terminator terminator{source, line};
  auto typeCode{get->type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      get->rank() == 1 && typeCode &&
          typeCode->first == TypeCategory::Integer &&
          get->GetDimension(0).Extent() >= 1);
  int getArg{typeCode->second};
  GeneratedWord seed;
  {
    CriticalSection critical{lock};
    seed = GetNextValue();
    nextValue = seed;
  }
  switch (getArg) {
  case 4:
    *get->OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>() = seed;
    break;
  case 8:
    *get->OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>() = seed;
    break;
  default:
    terminator.Crash(
        "not yet implemented: intrinsic: RANDOM_SEED(GET=): get %d\n", getArg);
  }
}

void RTNAME(RandomSeed)(const Descriptor *size, const Descriptor *put,
    const Descriptor *get, const char *source, int line) {
  bool sizePresent = size && size->raw().base_addr;
  bool putPresent = put && put->raw().base_addr;
  bool getPresent = get && get->raw().base_addr;
  if (sizePresent + putPresent + getPresent > 1)
    Terminator{source, line}.Crash(
        "RANDOM_SEED must have either 1 or no arguments");
  if (sizePresent)
    RTNAME(RandomSeedSize)(size, source, line);
  else if (putPresent)
    RTNAME(RandomSeedPut)(put, source, line);
  else if (getPresent)
    RTNAME(RandomSeedGet)(get, source, line);
  else
    RTNAME(RandomSeedDefaultPut)();
}

} // extern "C"
} // namespace Fortran::runtime::random
