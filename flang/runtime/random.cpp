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
#include "terminator.h"
#include "flang/Common/leading-zero-bit-count.h"
#include "flang/Common/uint128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/float128.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <limits>
#include <memory>
#include <random>

namespace Fortran::runtime {

// Newer "Minimum standard", recommended by Park, Miller, and Stockmeyer in
// 1993. Same as C++17 std::minstd_rand, but explicitly instantiated for
// permanence.
using Generator =
    std::linear_congruential_engine<std::uint_fast32_t, 48271, 0, 2147483647>;

using GeneratedWord = typename Generator::result_type;
static constexpr std::uint64_t range{
    static_cast<std::uint64_t>(Generator::max() - Generator::min() + 1)};
static constexpr bool rangeIsPowerOfTwo{(range & (range - 1)) == 0};
static constexpr int rangeBits{
    64 - common::LeadingZeroBitCount(range) - !rangeIsPowerOfTwo};

static Lock lock;
static Generator generator;
static std::optional<GeneratedWord> nextValue;

// Call only with lock held
static GeneratedWord GetNextValue() {
  GeneratedWord result;
  if (nextValue.has_value()) {
    result = *nextValue;
    nextValue.reset();
  } else {
    result = generator();
  }
  return result;
}

template <typename REAL, int PREC>
inline void Generate(const Descriptor &harvest) {
  static constexpr std::size_t minBits{
      std::max<std::size_t>(PREC, 8 * sizeof(GeneratedWord))};
  using Int = common::HostUnsignedIntType<minBits>;
  static constexpr std::size_t words{
      static_cast<std::size_t>(PREC + rangeBits - 1) / rangeBits};
  std::size_t elements{harvest.Elements()};
  SubscriptValue at[maxRank];
  harvest.GetLowerBounds(at);
  {
    CriticalSection critical{lock};
    for (std::size_t j{0}; j < elements; ++j) {
      while (true) {
        Int fraction{GetNextValue()};
        if constexpr (words > 1) {
          for (std::size_t k{1}; k < words; ++k) {
            static constexpr auto rangeMask{
                (GeneratedWord{1} << rangeBits) - 1};
            GeneratedWord word{(GetNextValue() - generator.min()) & rangeMask};
            fraction = (fraction << rangeBits) | word;
          }
        }
        fraction >>= words * rangeBits - PREC;
        REAL next{std::ldexp(static_cast<REAL>(fraction), -(PREC + 1))};
        if (next >= 0.0 && next < 1.0) {
          *harvest.Element<REAL>(at) = next;
          break;
        }
      }
      harvest.IncrementSubscripts(at);
    }
  }
}

extern "C" {

void RTNAME(RandomInit)(bool repeatable, bool /*image_distinct*/) {
  // TODO: multiple images and image_distinct: add image number
  {
    CriticalSection critical{lock};
    if (repeatable) {
      generator.seed(0);
    } else {
      generator.seed(std::time(nullptr));
    }
  }
}

void RTNAME(RandomNumber)(
    const Descriptor &harvest, const char *source, int line) {
  Terminator terminator{source, line};
  auto typeCode{harvest.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, typeCode && typeCode->first == TypeCategory::Real);
  int kind{typeCode->second};
  switch (kind) {
  // TODO: REAL (2 & 3)
  case 4:
    Generate<CppTypeFor<TypeCategory::Real, 4>, 24>(harvest);
    return;
  case 8:
    Generate<CppTypeFor<TypeCategory::Real, 8>, 53>(harvest);
    return;
  case 10:
    if constexpr (HasCppTypeFor<TypeCategory::Real, 10>) {
#if LDBL_MANT_DIG == 64
      Generate<CppTypeFor<TypeCategory::Real, 10>, 64>(harvest);
      return;
#endif
    }
    break;
  case 16:
    if constexpr (HasCppTypeFor<TypeCategory::Real, 16>) {
#if LDBL_MANT_DIG == 113
      Generate<CppTypeFor<TypeCategory::Real, 16>, 113>(harvest);
      return;
#endif
    }
    break;
  }
  terminator.Crash("not yet implemented: RANDOM_NUMBER(): REAL kind %d", kind);
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
  int kind{typeCode->second};
  switch (kind) {
  case 4:
    *size->OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>() = 1;
    break;
  case 8:
    *size->OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>() = 1;
    break;
  default:
    terminator.Crash(
        "not yet implemented: RANDOM_SEED(SIZE=): kind %d\n", kind);
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
  int kind{typeCode->second};
  GeneratedWord seed;
  switch (kind) {
  case 4:
    seed = *put->OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>();
    break;
  case 8:
    seed = *put->OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>();
    break;
  default:
    terminator.Crash("not yet implemented: RANDOM_SEED(PUT=): kind %d\n", kind);
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
  int kind{typeCode->second};
  GeneratedWord seed;
  {
    CriticalSection critical{lock};
    seed = GetNextValue();
    nextValue = seed;
  }
  switch (kind) {
  case 4:
    *get->OffsetElement<CppTypeFor<TypeCategory::Integer, 4>>() = seed;
    break;
  case 8:
    *get->OffsetElement<CppTypeFor<TypeCategory::Integer, 8>>() = seed;
    break;
  default:
    terminator.Crash("not yet implemented: RANDOM_SEED(GET=): kind %d\n", kind);
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
} // namespace Fortran::runtime
