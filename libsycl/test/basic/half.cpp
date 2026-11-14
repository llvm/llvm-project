//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Host-side checks of sycl::half: storage format, conversions, operators and
// std::numeric_limits.

#include "Inputs/half_common.hpp"

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <type_traits>
#include <unordered_set>

using sycl::half;
using Limits = std::numeric_limits<half>;

static std::uint16_t bits(half H) {
  std::uint16_t Result;
  std::memcpy(&Result, &H, sizeof(Result));
  return Result;
}

static half fromBits(std::uint16_t Bits) {
  half Result;
  std::memcpy(&Result, &Bits, sizeof(Result));
  return Result;
}

static std::uint32_t bits(float F) {
  std::uint32_t Result;
  std::memcpy(&Result, &F, sizeof(Result));
  return Result;
}

static float floatFromBits(std::uint32_t Bits) {
  float Result;
  std::memcpy(&Result, &Bits, sizeof(Result));
  return Result;
}

static_assert(std::is_same_v<decltype(std::fmin(half{}, half{})), float>);
static_assert(std::is_same_v<decltype(std::sqrt(half{})), float>);

// Argument-dependent lookup for half doesn't search namespace sycl, so
// unqualified calls resolve as they would for float.
namespace sycl {
constexpr int adlProbe(half) { return 1; }
} // namespace sycl
constexpr int adlProbe(float) { return 2; }
static_assert(adlProbe(half{}) == 2);

// half is usable in unions and as a trivially copyable kernel argument.
union HalfOrBits {
  std::uint16_t Bits;
  half Value;
};
static_assert(std::is_trivially_default_constructible_v<HalfOrBits>);
static_assert(std::is_trivially_copyable_v<half>);

// constexpr usage.
static_assert(half(1.5f) + half(2.5f) == 4);
static_assert(half(5.0f) - half(2.0f) == 3 && half(5.0f) / half(2.0f) == 2.5f);
static_assert(half(3.0f) * 2 == 6.0f);
static_assert(-half(2.0f) < 0);
static_assert(half(1.0f) + half_test::EnumValue == 4);
static_assert(half_test::ConvertsToInt{} < half(3.0f));
static_assert(Limits::max() == 65504.0f);
static_assert(Limits::lowest() == -65504.0f);
static_assert(Limits::min() == 6.103515625e-05f);
static_assert(Limits::epsilon() == 9.765625e-04f);
static_assert(Limits::round_error() == 0.5f);
static_assert(Limits::denorm_min() == 5.9604644775390625e-08f);
static_assert(Limits::digits == 11 && Limits::digits10 == 3 &&
              Limits::max_digits10 == 5 && Limits::radix == 2);
static_assert(Limits::min_exponent == -13 && Limits::min_exponent10 == -4 &&
              Limits::max_exponent == 16 && Limits::max_exponent10 == 4);
static_assert(Limits::is_specialized && Limits::is_signed &&
              !Limits::is_integer && !Limits::is_exact && Limits::is_iec559 &&
              Limits::is_bounded && !Limits::is_modulo);
static_assert(Limits::has_infinity && Limits::has_quiet_NaN &&
              Limits::has_signaling_NaN &&
              Limits::has_denorm == std::denorm_present &&
              Limits::round_style == std::round_to_nearest);

// Comparisons with integers are done in float, so the integer isn't rounded
// to half first: in half, 65519 would round to max() and 70000 to infinity,
// and both comparisons below would be false.
static_assert(Limits::max() < 65519);
static_assert(70000 < Limits::infinity());

void testStorageFormat() {
  // IEEE 754-2008 binary16 encodings.
  assert(bits(half(1.0f)) == 0x3C00);
  assert(bits(half(-2.0f)) == 0xC000);
  assert(bits(Limits::max()) == 0x7BFF);
  assert(bits(Limits::infinity()) == 0x7C00);
  assert(bits(Limits::denorm_min()) == 0x0001);
  assert(bits(-half(0.0f)) == 0x8000);

  // A signaling NaN keeps its quiet bit clear.
  std::uint16_t SNaN = bits(Limits::signaling_NaN());
  assert((SNaN & 0x7C00) == 0x7C00 && (SNaN & 0x03FF) != 0);
  assert((SNaN & 0x0200) == 0);
  assert(bits(Limits::quiet_NaN()) & 0x0200);
}

// Every half value survives a conversion to float and back. NaNs stay quiet
// NaNs of the same sign; their payload isn't guaranteed to be preserved.
void testRoundTrip() {
  for (std::uint32_t I = 0; I <= 0xFFFF; ++I) {
    half H = fromBits(static_cast<std::uint16_t>(I));
    float F = H;
    half Back = F;
    if (std::isnan(F)) {
      assert(std::isnan(static_cast<float>(Back)));
      assert((bits(Back) & 0x8000) == (I & 0x8000));
      assert(bits(Back) & 0x0200);
    } else {
      assert(bits(Back) == I);
    }
  }
}

void testConversions() {
  // Round to nearest, ties to even: 2049 is halfway between 2048 and 2050.
  assert(half(2049.0f) == 2048);
  assert(half(2051.0f) == 2052);
  assert(static_cast<int>(half(-7.0f)) == -7);
  double D = half(0.5f);
  assert(D == 0.5);

  // float to half.
  assert(bits(half(75514.0f)) == 0x7C00);
  assert(bits(half(-75514.0f)) == 0xFC00);
  assert(bits(half(0.0f)) == 0x0000);
  assert(bits(half(-0.0f)) == 0x8000);
  assert(bits(half(1.9999f)) == 0x4000);
  assert(bits(half(9.8e-45f)) == 0x0000);
  assert(bits(half(-9.8e-45f)) == 0x8000);
  assert(bits(half(floatFromBits(0x38200000))) == 0x0280);
  for (std::uint32_t NaN : {0x7FC00000u, 0x7F800001u, 0xFF800001u}) {
    half H = floatFromBits(NaN);
    assert(std::isnan(static_cast<float>(H)));
    assert((bits(H) & 0x8000) == ((NaN >> 16) & 0x8000));
  }

  // half to float.
  assert(bits(static_cast<float>(fromBits(0x7A00))) == 0x47400000);
  assert(bits(static_cast<float>(fromBits(0xFA00))) == 0xC7400000);
  assert(bits(static_cast<float>(fromBits(0x0000))) == 0x00000000);
  assert(bits(static_cast<float>(fromBits(0x8000))) == 0x80000000);
  assert(bits(static_cast<float>(fromBits(0x7A03))) == 0x47406000);
  assert(bits(static_cast<float>(fromBits(0x0005))) == 0x34A00000);
}

void testOperators() {
  half H = 1.0f;
  H += 2;
  H *= half(2.0f);
  H -= 0.5f;
  H /= 2.0;
  assert(H == 2.75f);

  // Compound assignment converts the operand to half first: it rounds to 2^-11,
  // so 1 + 2^-11 is a tie that rounds to 1. Computed in float, it would round
  // up.
  half One = 1.0f;
  One += 0.00048828125f + 0.0000001f;
  assert(One == 1);

  // Operands that convert implicitly to an arithmetic type.
  half FromClass = 1.0f;
  FromClass += half_test::ConvertsToInt{};
  FromClass *= sycl::id<1>{2};
  assert(FromClass == 6);

  assert(++H == 3.75f);
  assert(H-- == 3.75f && H == 2.75f);

  // Mixed-type comparisons in both orders.
  assert(H > 2 && 3 > H && H != 2.0 && 2.75 == H);
  assert(!(Limits::quiet_NaN() == Limits::quiet_NaN()));

  // Arithmetic in half precision rounds to binary16 (1 + 2^-11 -> 1).
  assert(half(1.0f) + half(0.00048828125f) == 1);
  // Arithmetic with float keeps float precision.
  assert(half(1.0f) + 0.00048828125f != 1.0f);

  // Overflow and underflow.
  assert(bits(half(55504.0f) * 3) == 0x7C00);
  assert(bits(half(-55504.0f) * 3) == 0xFC00);
  assert(bits(half(8.1035e-05f) / half(3.0f)) == 0x01C5);
}

void testStreams() {
  std::stringstream SS;
  SS << half(1.5f);
  assert(SS.str() == "1.5");
  half H;
  SS >> H;
  assert(H == 1.5f);

  // As for float, a failed extraction stores zero.
  std::stringstream Bad("abc");
  half B = 1.0f;
  Bad >> B;
  assert(Bad.fail() && B == 0);
}

void testHash() {
  std::hash<half> Hash;
  assert(Hash(half(0.0f)) == Hash(half(-0.0f)));
  assert(Hash(half(1.5f)) == Hash(half(1.5f)));

  std::unordered_set<half> Set = {half(1.0f), half(2.0f), half(0.0f)};
  assert(Set.size() == 3);
  assert(Set.count(half(2.0f)) == 1 && Set.count(half(3.0f)) == 0);
  assert(Set.count(half(-0.0f)) == 1);
}

int main() {
  half_test::checkAllOperatorTypes();
  half_test::checkConstexprLimits();
  testStorageFormat();
  testRoundTrip();
  testConversions();
  testOperators();
  testStreams();
  testHash();
  return 0;
}
