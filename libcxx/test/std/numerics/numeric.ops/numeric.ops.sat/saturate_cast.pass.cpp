//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class R, class T>
//   constexpr R saturate_cast(T x) noexcept;                     // freestanding

#include <cassert>
#include <climits>
#include <concepts>
#include <limits>
#include <numeric>

#include "test_macros.h"

// Smaller to larger
static_assert(noexcept(std::saturate_cast<signed int>(std::numeric_limits<signed char>::max())));
static_assert(noexcept(std::saturate_cast<signed int>(std::numeric_limits<unsigned char>::max())));

static_assert(noexcept(std::saturate_cast<unsigned int>(std::numeric_limits<signed char>::max())));
static_assert(noexcept(std::saturate_cast<unsigned int>(std::numeric_limits<unsigned char>::max())));

// Same type
static_assert(noexcept(std::saturate_cast<signed long int>(std::numeric_limits<signed long int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned long int>(std::numeric_limits<unsigned long int>::max())));

// Larger to smaller
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<signed char>(std::numeric_limits<unsigned int>::max())));

static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<signed int>::max())));
static_assert(noexcept(std::saturate_cast<unsigned char>(std::numeric_limits<unsigned int>::max())));

template <typename IntegerT>
constexpr auto zero() {
  return IntegerT{0};
}

constexpr auto operator ""_C(unsigned long long int i) {
    return static_cast<signed char>(i);
}

constexpr auto operator ""_UC(unsigned long long int i) {
    return static_cast<unsigned char>(i);
}

constexpr auto operator ""_S(unsigned long long int i) {
    return static_cast<signed short int>(i);
}

constexpr auto operator ""_US(unsigned long long int i) {
    return static_cast<unsigned short int>(i);
}

constexpr bool test() {
  // clang-format off

#ifndef TEST_HAS_NO_INT128
  using SIntT = __int128_t;
  using UIntT = __uint128_t;
#else
  using SIntT = long long int;
  using UIntT = unsigned long long int;
#endif

  // Constants: biggest numbers

  constexpr auto big_sintMin  = std::numeric_limits<SIntT>::min();
  constexpr auto big_sintZero =                zero<SIntT>();
  constexpr auto big_sintMax  = std::numeric_limits<SIntT>::max();

  constexpr auto big_uintMin  = std::numeric_limits<UIntT>::min();
  constexpr auto big_uintZero =                zero<UIntT>();
  constexpr auto big_uintMax  = std::numeric_limits<UIntT>::max();

  // Constants: numeric limits

  constexpr auto std_scharMin  = std::numeric_limits<signed char>::min();
  constexpr auto std_scharZero =                zero<signed char>();
  constexpr auto std_scharMax  = std::numeric_limits<signed char>::max();

  constexpr auto std_ucharMin  = std::numeric_limits<unsigned char>::min();
  constexpr auto std_ucharZero =                zero<unsigned char>();
  constexpr auto std_ucharMax  = std::numeric_limits<unsigned char>::max();

  constexpr auto std_ssintMin  = std::numeric_limits<signed short int>::min();
  constexpr auto std_ssintZero =                zero<signed short int>();
  constexpr auto std_ssintMax  = std::numeric_limits<signed short int>::max();

  constexpr auto std_usintMin  = std::numeric_limits<unsigned short int>::min();
  constexpr auto std_usintZero =                zero<unsigned short int>();
  constexpr auto std_usintMax  = std::numeric_limits<unsigned short int>::max();

  constexpr auto std_sintMin   = std::numeric_limits<signed int>::min();
  constexpr auto std_sintZero  =                zero<signed int>();
  constexpr auto std_sintMax   = std::numeric_limits<signed int>::max();

  constexpr auto std_uintMin   = std::numeric_limits<unsigned int>::min();
  constexpr auto std_uintZero  =                zero<unsigned int>();
  constexpr auto std_uintMax   = std::numeric_limits<unsigned int>::max();

  constexpr auto std_slMin     = std::numeric_limits<signed long int>::min();
  constexpr auto std_slZero    =                zero<signed long int>();
  constexpr auto std_slMax     = std::numeric_limits<signed long int>::max();

  constexpr auto std_ulMin     = std::numeric_limits<unsigned long int>::min();
  constexpr auto std_ulZero    =                zero<unsigned long int>();
  constexpr auto std_ulMax     = std::numeric_limits<unsigned long int>::max();

  constexpr auto std_sllMin    = std::numeric_limits<signed long long int>::min();
  constexpr auto std_sllZero   =                zero<signed long long int>();
  constexpr auto std_sllMax    = std::numeric_limits<signed long long int>::max();

  constexpr auto std_ullMin    = std::numeric_limits<unsigned long long int>::min();
  constexpr auto std_ullZero   =                zero<unsigned long long int>();
  constexpr auto std_ullMax    = std::numeric_limits<unsigned long long int>::max();
  
  // signed char

  assert(std::saturate_cast<signed char>(std_scharMin)  == std_scharMin);
  assert(std::saturate_cast<signed char>(std_scharZero) == std_scharZero);
  assert(std::saturate_cast<signed char>(std_scharMax)  == std_scharMax);

  assert(std::saturate_cast<signed char>(std_ucharMin)  == std_scharZero);
  assert(std::saturate_cast<signed char>(std_ucharZero) == std_scharZero);
  assert(std::saturate_cast<signed char>(std_ucharMax)  == std_scharMax);

  assert(std::saturate_cast<signed char>(big_sintMin)   == std_scharMin);  // saturated
  assert(std::saturate_cast<signed char>(big_sintZero)  == std_scharZero);
  assert(std::saturate_cast<signed char>(big_sintMax)   == std_scharMax);  // saturated

  assert(std::saturate_cast<signed char>(big_uintMin)   == std_scharZero);
  assert(std::saturate_cast<signed char>(big_uintZero)  == std_scharZero);
  assert(std::saturate_cast<signed char>(big_uintMax)   == std_scharMax);  // saturated

  // short

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_scharMin); }
  assert(std::saturate_cast<signed short int>(std_scharMin)  == static_cast<signed short int>(std_scharMin));
  assert(std::saturate_cast<signed short int>(std_scharZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_scharMax)  == static_cast<signed short int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_ucharMin); }
  assert(std::saturate_cast<signed short int>(std_ucharMin)  == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_ucharZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_ucharMax)  == static_cast<signed short int>(std_ucharMax));

   // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_ssintMin); }
  assert(std::saturate_cast<signed short int>(std_ssintMin)  == std_ssintMin);
  assert(std::saturate_cast<signed short int>(std_ssintZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_ssintMax)  == std_ssintMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(std_usintMin); }
  assert(std::saturate_cast<signed short int>(std_usintMin)  == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_usintZero) == std_ssintZero);
  assert(std::saturate_cast<signed short int>(std_usintMax)  == std_ssintMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(big_sintMin); }
  assert(std::saturate_cast<signed short int>(big_sintMin)   == std_ssintMin);  // saturated
  assert(std::saturate_cast<signed short int>(big_sintZero)  == std_ssintZero);
  assert(std::saturate_cast<signed short int>(big_sintMax)   == std_ssintMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed short int> decltype(auto) _ = std::saturate_cast<signed short int>(big_uintMin); }
  assert(std::saturate_cast<signed short int>(big_uintMin)   == std_ssintZero);
  assert(std::saturate_cast<signed short int>(big_uintZero)  == std_ssintZero);
  assert(std::saturate_cast<signed short int>(big_uintMax)   == std_ssintMax);  // saturated

  // int

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(std_scharMin); }
  assert(std::saturate_cast<signed int>(std_scharMin)  == static_cast<signed int>(std_scharMin));
  assert(std::saturate_cast<signed int>(std_scharZero) == std_sintZero);
  assert(std::saturate_cast<signed int>(std_scharMax)  == static_cast<signed int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(std_ucharMin); }
  assert(std::saturate_cast<signed int>(std_ucharMin)  == std_sintZero);
  assert(std::saturate_cast<signed int>(std_ucharZero) == std_sintZero);
  assert(std::saturate_cast<signed int>(std_ucharMax)  == static_cast<signed int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(std_sintMin); }
  assert(std::saturate_cast<signed int>(std_sintMin)   == std_sintMin);
  assert(std::saturate_cast<signed int>(std_sintZero)  == std_sintZero);
  assert(std::saturate_cast<signed int>(std_sintMax)   == std_sintMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(std_uintMin); }
  assert(std::saturate_cast<signed int>(std_uintMin)   == std_sintZero);
  assert(std::saturate_cast<signed int>(std_uintZero)  == std_sintZero);
  assert(std::saturate_cast<signed int>(std_uintMax)   == std_sintMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(big_sintMin); }
  assert(std::saturate_cast<signed int>(big_sintMin)   == std_sintMin);  // saturated
  assert(std::saturate_cast<signed int>(big_sintZero)  == std_sintZero);
  assert(std::saturate_cast<signed int>(big_sintMax)   == std_sintMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed int> decltype(auto) _ = std::saturate_cast<signed int>(big_uintMin); }
  assert(std::saturate_cast<signed int>(big_uintMin)   == std_sintZero);
  assert(std::saturate_cast<signed int>(big_uintZero)  == std_sintZero);
  assert(std::saturate_cast<signed int>(big_uintMax)   == std_sintMax);  // saturated

  // long

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(std_scharMin); }
  assert(std::saturate_cast<signed long int>(std_scharMin)  == static_cast<signed long int>(std_scharMin));
  assert(std::saturate_cast<signed long int>(std_scharZero) == std_slZero);
  assert(std::saturate_cast<signed long int>(std_scharMax)  == static_cast<signed long int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(std_ucharMin); }
  assert(std::saturate_cast<signed long int>(std_ucharMin)  == std_slZero);
  assert(std::saturate_cast<signed long int>(std_ucharZero) == std_slZero);
  assert(std::saturate_cast<signed long int>(std_ucharMax)  == static_cast<signed long int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(std_slMin); }
  assert(std::saturate_cast<signed long int>(std_slMin)     == std_slMin);
  assert(std::saturate_cast<signed long int>(std_slZero)    == std_slZero);
  assert(std::saturate_cast<signed long int>(std_slMax)     == std_slMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(std_ulMin); }
  assert(std::saturate_cast<signed long int>(std_ulMin)     == std_slZero);
  assert(std::saturate_cast<signed long int>(std_ulZero)    == std_slZero);
  assert(std::saturate_cast<signed long int>(std_ulMax)     == std_slMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(big_sintMin); }
  assert(std::saturate_cast<signed long int>(big_sintMin)   == std_slMin);  // saturated
  assert(std::saturate_cast<signed long int>(big_sintZero)  == std_slZero);
  assert(std::saturate_cast<signed long int>(big_sintMax)   == std_slMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long int> decltype(auto) _ = std::saturate_cast<signed long int>(big_uintMin); }
  assert(std::saturate_cast<signed long int>(big_uintMin)   == std_slZero);
  assert(std::saturate_cast<signed long int>(big_uintZero)  == std_slZero);
  assert(std::saturate_cast<signed long int>(big_uintMax)   == std_slMax);  // saturated

  // long long

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(std_scharMin); }
  assert(std::saturate_cast<signed long long int>(std_scharMin)  == static_cast<signed long long int>(std_scharMin));
  assert(std::saturate_cast<signed long long int>(std_scharZero) == std_sllZero);
  assert(std::saturate_cast<signed long long int>(std_scharMax)  == static_cast<signed long long int>(std_scharMax));

   // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]]   std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(std_ucharMin); }
  assert(std::saturate_cast<signed long long int>(std_ucharMin)  == std_sllZero);
  assert(std::saturate_cast<signed long long int>(std_ucharZero) == std_sllZero);
  assert(std::saturate_cast<signed long long int>(std_ucharMax)  == static_cast<signed long long int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(std_sllMin); }
  assert(std::saturate_cast<signed long long int>(std_sllMin)     == std_sllMin);
  assert(std::saturate_cast<signed long long int>(std_sllZero)    == std_sllZero);
  assert(std::saturate_cast<signed long long int>(std_sllMax)     == std_sllMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(std_ullMin); }
  assert(std::saturate_cast<signed long long int>(std_ullMin)     == std_sllZero);
  assert(std::saturate_cast<signed long long int>(std_ullZero)    == std_sllZero);
  assert(std::saturate_cast<signed long long int>(std_ullMax)     == std_sllMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(big_sintMin); }
  assert(std::saturate_cast<signed long long int>(big_sintMin)    == std_sllMin);  // (128-bit) saturated
  assert(std::saturate_cast<signed long long int>(big_sintZero)   == std_sllZero);
  assert(std::saturate_cast<signed long long int>(big_sintMax)    == std_sllMax);  // (128-bit) saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<signed long long int> decltype(auto) _ = std::saturate_cast<signed long long int>(big_uintMin); }
  assert(std::saturate_cast<signed long long int>(big_uintMin)    == std_sllZero);
  assert(std::saturate_cast<signed long long int>(big_uintZero)   == std_sllZero);
  assert(std::saturate_cast<signed long long int>(big_uintMax)    == std_sllMax);  // (128-bit) saturated

#ifndef TEST_HAS_NO_INT128
  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<__int128_t> decltype(auto) _ = std::saturate_cast<__int128_t>(std_scharMin); }
  assert(std::saturate_cast<__int128_t>(std_scharMin)  == static_cast<__int128_t>(std_scharMin));
  assert(std::saturate_cast<__int128_t>(std_scharZero) == big_sintZero);
  assert(std::saturate_cast<__int128_t>(std_scharMax)  == static_cast<__int128_t>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<__int128_t> decltype(auto) _ = std::saturate_cast<__int128_t>(std_ucharMin); }
  assert(std::saturate_cast<__int128_t>(std_ucharMin)  == static_cast<__int128_t>(std_ucharMin));
  assert(std::saturate_cast<__int128_t>(std_ucharZero) == big_sintZero);
  assert(std::saturate_cast<__int128_t>(std_ucharMax)  == static_cast<__int128_t>(std_ucharMax));

  assert(std::saturate_cast<__int128_t>(big_sintMin)   == big_sintMin);
  assert(std::saturate_cast<__int128_t>(big_sintZero)  == big_sintZero);
  assert(std::saturate_cast<__int128_t>(big_sintMax)   == big_sintMax);

  assert(std::saturate_cast<__int128_t>(big_uintMin)   == big_sintZero);
  assert(std::saturate_cast<__int128_t>(big_uintZero)  == big_sintZero);
  assert(std::saturate_cast<__int128_t>(big_uintMax)   == big_sintMax);  // saturated
#endif

  // unsigned char

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(std_scharMin); }
  assert(std::saturate_cast<unsigned char>(std_scharMin)  == std_ucharMin);
  assert(std::saturate_cast<unsigned char>(std_scharZero) == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(std_scharMax)  == static_cast<unsigned char>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(std_ucharMin); }
  assert(std::saturate_cast<unsigned char>(std_ucharMin)  == std_ucharMin);
  assert(std::saturate_cast<unsigned char>(std_ucharZero) == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(std_ucharMax)  == std_ucharMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(big_sintMin); }
  assert(std::saturate_cast<unsigned char>(big_sintMin)   == std_ucharMin);  // saturated
  assert(std::saturate_cast<unsigned char>(big_sintZero)  == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(big_sintMax)   == std_ucharMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned char> decltype(auto) _ = std::saturate_cast<unsigned char>(big_uintMin); }
  assert(std::saturate_cast<unsigned char>(big_uintMin)   == std_ucharMin);
  assert(std::saturate_cast<unsigned char>(big_uintZero)  == std_ucharZero);
  assert(std::saturate_cast<unsigned char>(big_uintMax)   == std_ucharMax);  // saturated

  // short

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(std_scharMin); }
  assert(std::saturate_cast<unsigned short int>(std_scharMin)  == std_usintMin);
  assert(std::saturate_cast<unsigned short int>(std_scharZero) == std_usintZero);
  assert(std::saturate_cast<unsigned short int>(std_scharMax)  == static_cast<unsigned short int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(std_ucharMin); }
  assert(std::saturate_cast<unsigned short int>(std_ucharMin)  == std_usintMin);
  assert(std::saturate_cast<unsigned short int>(std_ucharZero) == std_usintZero);
  assert(std::saturate_cast<unsigned short int>(std_ucharMax)  == static_cast<unsigned short int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(std_scharMin); }
  assert(std::saturate_cast<unsigned short int>(std_ssintMin)  == std_usintMin);
  assert(std::saturate_cast<unsigned short int>(std_ssintZero) == std_usintZero);
  assert(std::saturate_cast<unsigned short int>(std_ssintMax)  == static_cast<unsigned short int>(std_ssintMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(std_ucharMin); }
  assert(std::saturate_cast<unsigned short int>(std_usintMin)  == std_usintMin);
  assert(std::saturate_cast<unsigned short int>(std_usintZero) == std_usintZero);
  assert(std::saturate_cast<unsigned short int>(std_usintMax)  == std_usintMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(big_sintMin); }
  assert(std::saturate_cast<unsigned short int>(big_sintMin)   == std_usintMin);  // saturated
  assert(std::saturate_cast<unsigned short int>(big_sintZero)  == std_usintZero);
  assert(std::saturate_cast<unsigned short int>(big_sintMax)   == std_usintMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned short int> decltype(auto) _ = std::saturate_cast<unsigned short int>(big_uintMin); }
  assert(std::saturate_cast<unsigned short int>(big_uintMin)   == std_usintMin);
  assert(std::saturate_cast<unsigned short int>(big_uintZero)  == std_usintZero);
  assert(std::saturate_cast<unsigned short int>(big_uintMax)   == std_usintMax);  // saturated

  // unsigned int

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(std_scharMin); }
  assert(std::saturate_cast<unsigned int>(std_scharMin)  == std_usintMin);
  assert(std::saturate_cast<unsigned int>(std_scharZero) == std_usintZero);
  assert(std::saturate_cast<unsigned int>(std_scharMax)  == static_cast<unsigned int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(std_ucharMin); }
  assert(std::saturate_cast<unsigned int>(std_ucharMin)  == std_uintMin);
  assert(std::saturate_cast<unsigned int>(std_ucharZero) == std_uintZero);
  assert(std::saturate_cast<unsigned int>(std_ucharMax)  == static_cast<unsigned int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(std_sintMin); }
  assert(std::saturate_cast<unsigned int>(std_sintMin)   == std_uintMin);
  assert(std::saturate_cast<unsigned int>(std_sintZero)  == std_uintZero);
  assert(std::saturate_cast<unsigned int>(std_sintMax)   == static_cast<unsigned int>(std_sintMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(std_uintMin); }
  assert(std::saturate_cast<unsigned int>(std_uintMin)   == std_uintMin);
  assert(std::saturate_cast<unsigned int>(std_uintZero)  == std_uintZero);
  assert(std::saturate_cast<unsigned int>(std_uintMax)   == std_uintMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(big_sintMin); }
  assert(std::saturate_cast<unsigned int>(big_sintMin)   == std_uintMin);  // saturated
  assert(std::saturate_cast<unsigned int>(big_sintZero)  == std_uintZero);
  assert(std::saturate_cast<unsigned int>(big_sintMax)   == std_uintMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned int> decltype(auto) _ = std::saturate_cast<unsigned int>(big_uintMin); }
  assert(std::saturate_cast<unsigned int>(big_uintMin)   == std_uintMin);
  assert(std::saturate_cast<unsigned int>(big_uintZero)  == std_uintZero);
  assert(std::saturate_cast<unsigned int>(big_uintMax)   == std_uintMax);  // saturated

  // unsigned long

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(std_scharMin); }
  assert(std::saturate_cast<unsigned long int>(std_scharMin)  == std_ulMin);
  assert(std::saturate_cast<unsigned long int>(std_scharZero) == std_ulZero);
  assert(std::saturate_cast<unsigned long int>(std_scharMax)  == static_cast<unsigned long int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(std_ucharMin); }
  assert(std::saturate_cast<unsigned long int>(std_ucharMin)  == std_ulMin);
  assert(std::saturate_cast<unsigned long int>(std_ucharZero) == std_ulZero);
  assert(std::saturate_cast<unsigned long int>(std_ucharMax)  == static_cast<unsigned long int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(std_slMin); }
  assert(std::saturate_cast<unsigned long int>(std_slMin)     == std_ulMin);
  assert(std::saturate_cast<unsigned long int>(std_slZero)    == std_ulZero);
  assert(std::saturate_cast<unsigned long int>(std_slMax)     == static_cast<unsigned long int>(std_slMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(std_ulMin); }
  assert(std::saturate_cast<unsigned long int>(std_ulMin)     == std_ulMin);
  assert(std::saturate_cast<unsigned long int>(std_ulZero)    == std_ulZero);
  assert(std::saturate_cast<unsigned long int>(std_ulMax)     == std_ulMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(big_sintMin); }
  assert(std::saturate_cast<unsigned long int>(big_sintMin)   == std_ulMin);  // saturated
  assert(std::saturate_cast<unsigned long int>(big_sintZero)  == std_ulZero);
  assert(std::saturate_cast<unsigned long int>(big_sintMax)   == std_ulMax);  // saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long int> decltype(auto) _ = std::saturate_cast<unsigned long int>(big_uintMin); }
  assert(std::saturate_cast<unsigned long int>(big_uintMin)   == std_ulMin);
  assert(std::saturate_cast<unsigned long int>(big_uintZero)  == std_ulZero);
  assert(std::saturate_cast<unsigned long int>(big_uintMax)   == std_ulMax);  // saturated

  // unsigned long long

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(std_scharMin); }
  assert(std::saturate_cast<unsigned long long int>(std_scharMin)  == std_ullMin);
  assert(std::saturate_cast<unsigned long long int>(std_scharZero) == std_ullZero);
  assert(std::saturate_cast<unsigned long long int>(std_scharMax)  == static_cast<unsigned long long int>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(std_ucharMin); }
  assert(std::saturate_cast<unsigned long long int>(std_ucharMin)  == std_ullMin);
  assert(std::saturate_cast<unsigned long long int>(std_ucharZero) == std_ullZero);
  assert(std::saturate_cast<unsigned long long int>(std_ucharMax)  == static_cast<unsigned long long int>(std_ucharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(std_sllMin); }
  assert(std::saturate_cast<unsigned long long int>(std_sllMin)    == std_ullMin);
  assert(std::saturate_cast<unsigned long long int>(std_sllZero)   == std_ullZero);
  assert(std::saturate_cast<unsigned long long int>(std_sllMax)    == static_cast<unsigned long long int>(std_sllMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(std_ullMin); }
  assert(std::saturate_cast<unsigned long long int>(std_ullMin)    == std_ullMin);
  assert(std::saturate_cast<unsigned long long int>(std_ullZero)   == std_ullZero);
  assert(std::saturate_cast<unsigned long long int>(std_ullMax)    == std_ullMax);

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(big_sintMin); }
  assert(std::saturate_cast<unsigned long long int>(big_sintMin)   == std_ullMin);  // (128-bit) saturated
  assert(std::saturate_cast<unsigned long long int>(big_sintZero)  == std_ullZero);
  assert(std::saturate_cast<unsigned long long int>(big_sintMax)   == std_ullMax);  // (128-bit) saturated

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<unsigned long long int> decltype(auto) _ = std::saturate_cast<unsigned long long int>(big_uintMin); }
  assert(std::saturate_cast<unsigned long long int>(big_uintMin)   == std_ullMin);
  assert(std::saturate_cast<unsigned long long int>(big_uintZero)  == std_ullZero);
  assert(std::saturate_cast<unsigned long long int>(big_uintMax)   == std_ullMax);  // (128-bit) saturated

#ifndef TEST_HAS_NO_INT128
  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<__uint128_t> decltype(auto) _ = std::saturate_cast<__uint128_t>(std_scharMin); }
  assert(std::saturate_cast<__uint128_t>(std_scharMin)  == big_uintMin);
  assert(std::saturate_cast<__uint128_t>(std_scharZero) == big_uintZero);
  assert(std::saturate_cast<__uint128_t>(std_scharMax)  == static_cast<__uint128_t>(std_scharMax));

  // TODO(LLVM-20) remove [[maybe_unused]] and `{}` scope since all supported compilers support "Placeholder variables with no name"
  { [[maybe_unused]] std::same_as<__uint128_t> decltype(auto) _ = std::saturate_cast<__uint128_t>(std_ucharMin); }
  assert(std::saturate_cast<__uint128_t>(std_ucharMin)  == big_uintMin);
  assert(std::saturate_cast<__uint128_t>(std_ucharZero) == big_uintZero);
  assert(std::saturate_cast<__uint128_t>(std_ucharMax)  == static_cast<__uint128_t>(std_ucharMax));

  assert(std::saturate_cast<__uint128_t>(big_sintMin)   == big_uintMin); // saturated
  assert(std::saturate_cast<__uint128_t>(big_sintZero)  == big_uintZero);
  assert(std::saturate_cast<__uint128_t>(big_sintMax)   == static_cast<__uint128_t>(big_sintMax));

  assert(std::saturate_cast<__uint128_t>(big_uintMin)   == big_uintMin);
  assert(std::saturate_cast<__uint128_t>(big_uintZero)  == big_uintZero);
  assert(std::saturate_cast<__uint128_t>(big_uintMax)   == big_uintMax);
#endif

  // clang-format on

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
