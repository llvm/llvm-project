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
// Host-side checks of sycl::half: storage format, conversions, operator result
// types and std::numeric_limits.

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <type_traits>

using sycl::half;
using Limits = std::numeric_limits<half>;

static std::uint16_t bits(half H) {
  std::uint16_t Result;
  std::memcpy(&Result, &H, sizeof(Result));
  return Result;
}

// Operator result types follow the usual arithmetic conversions.
static_assert(std::is_same_v<decltype(half{} + half{}), half>);
static_assert(std::is_same_v<decltype(half{} + 1), half>);
static_assert(std::is_same_v<decltype(1u * half{}), half>);
static_assert(std::is_same_v<decltype(half{} - 1.0f), float>);
static_assert(std::is_same_v<decltype(1.0 / half{}), double>);
static_assert(std::is_same_v<decltype(half{} + 1.0L), long double>);
static_assert(std::is_same_v<decltype(-half{}), half>);
static_assert(std::is_same_v<decltype(+half{}), half>);

// Operands that convert implicitly to an arithmetic type behave as with the
// built-in operators.
enum UnscopedEnum { EnumValue = 3 };
struct ConvertsToInt {
  constexpr operator int() const { return 2; }
};
static_assert(std::is_same_v<decltype(half{} * EnumValue), half>);
static_assert(std::is_same_v<decltype(ConvertsToInt{} + half{}), half>);
static_assert(std::is_same_v<decltype(half{} * sycl::id<1>{}), half>);
static_assert(half(1.0f) + EnumValue == 4 && ConvertsToInt{} < half(3.0f));
static_assert(std::is_same_v<decltype(half{} < 1), bool>);
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
static_assert(std::is_trivially_default_constructible_v<half>);
static_assert(std::is_trivially_copyable_v<half>);

// constexpr usage.
static_assert(half(1.5f) + half(2.5f) == 4);
static_assert(half(3.0f) * 2 == 6.0f);
static_assert(-half(2.0f) < 0);
static_assert(Limits::max() == 65504.0f);
static_assert(Limits::lowest() == -65504.0f);
static_assert(Limits::min() == 6.103515625e-05f);
static_assert(Limits::epsilon() == 9.765625e-04f);
static_assert(Limits::denorm_min() == 5.9604644775390625e-08f);
static_assert(Limits::digits == 11 && Limits::max_exponent == 16);
static_assert(Limits::is_specialized && Limits::is_iec559);

// Integers outside the half range compare exactly instead of rounding to
// infinity.
static_assert(Limits::max() < 65519);
static_assert(Limits::infinity() != 70000 && 70000 < Limits::infinity());

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

void testConversions() {
  // Round to nearest, ties to even: 2049 is halfway between 2048 and 2050.
  assert(half(2049.0f) == 2048);
  assert(half(2051.0f) == 2052);
  assert(std::isinf(static_cast<float>(half(1.0e6f))));
  assert(std::isnan(static_cast<float>(half(NAN))));
  assert(static_cast<int>(half(-7.0f)) == -7);
  double D = half(0.5f);
  assert(D == 0.5);
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

  assert(++H == 3.75f);
  assert(H-- == 3.75f && H == 2.75f);

  // Mixed-type comparisons in both orders.
  assert(H > 2 && 3 > H && H != 2.0 && 2.75 == H);
  assert(!(Limits::quiet_NaN() == Limits::quiet_NaN()));

  // Arithmetic in half precision rounds to binary16 (1 + 2^-11 -> 1).
  assert(half(1.0f) + half(0.00048828125f) == 1);
  // Arithmetic with float keeps float precision.
  assert(half(1.0f) + 0.00048828125f != 1.0f);
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

int main() {
  testStorageFormat();
  testConversions();
  testOperators();
  testStreams();
  return 0;
}
