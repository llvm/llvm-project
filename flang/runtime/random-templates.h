//===-- runtime/random-templates.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_RANDOM_TEMPLATES_H_
#define FORTRAN_RUNTIME_RANDOM_TEMPLATES_H_

#include "lock.h"
#include "numeric-templates.h"
#include "flang/Runtime/descriptor.h"
#include <algorithm>
#include <random>

namespace Fortran::runtime::random {

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

extern Lock lock;
extern Generator generator;
extern std::optional<GeneratedWord> nextValue;

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
        REAL next{
            LDEXPTy<REAL>::compute(static_cast<REAL>(fraction), -(PREC + 1))};
        if (next >= 0.0 && next < 1.0) {
          *harvest.Element<REAL>(at) = next;
          break;
        }
      }
      harvest.IncrementSubscripts(at);
    }
  }
}

} // namespace Fortran::runtime::random

#endif // FORTRAN_RUNTIME_RANDOM_TEMPLATES_H_
