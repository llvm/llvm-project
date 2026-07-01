//===-- lib/Evaluate/integer-value.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/integer-value.h"
#include "flang/Evaluate/integer.h"
#include <cstring>
#include <string>

namespace Fortran::evaluate::value {

namespace {

// Converts a fixed-width value::Integer to an APInt of the same width,
// preserving its raw bit pattern.
template <typename T> llvm::APInt IntegerToAP(const T &n) {
  constexpr unsigned w = T::bits;
  constexpr unsigned nWords = (w + 63) / 64;
  std::uint64_t words[nWords];
  T cur{n};
  for (unsigned i = 0; i < nWords; ++i) {
    words[i] = cur.template ToUInt<std::uint64_t>();
    cur = cur.SHIFTR(64);
  }
  return llvm::APInt(w, llvm::ArrayRef<std::uint64_t>(words, nWords));
}

} // namespace

// ============================================================================
// IntegerValue out-of-line definitions.
// ============================================================================

// --- Construction / raw bytes -----------------------------------------------

IntegerValue IntegerValue::FromRawBytes(const void *raw, int kind) {
  if (kind == 0) {
    // Matches the historical default case (used by InitialImage):
    // zero-initialized widest raw word.
    return Wrap(llvm::APInt(128, 0));
  }
  unsigned bits{BitsForKind(kind)};
  unsigned nWords{(bits + 63) / 64};
  llvm::SmallVector<std::uint64_t, 4> words(nWords, 0);
  std::size_t copyBytes{static_cast<std::size_t>((bits + 7) / 8)};
  std::memcpy(words.data(), raw, copyBytes);
  return Wrap(llvm::APInt(bits, llvm::ArrayRef(words.data(), nWords)));
}

bool IntegerValue::StoreRawBytes(void *to, int kind) const {
  unsigned outBits{BitsForKind(kind)};
  llvm::APInt out{
      IsMonostate() ? llvm::APInt(outBits, 0) : ap().zextOrTrunc(outBits)};
  std::size_t payloadBytes{static_cast<std::size_t>((outBits + 7) / 8)};
  if (std::memcmp(to, out.getRawData(), payloadBytes) != 0) {
    std::memcpy(to, out.getRawData(), payloadBytes);
    return true;
  }
  return false;
}

// --- Integral conversions ---------------------------------------------------

std::uint64_t IntegerValue::ToUInt64() const {
  if (IsMonostate()) {
    return 0;
  }
  return ap().getRawData()[0]; // low 64 bits, zero-extended
}

std::int64_t IntegerValue::ToInt64() const {
  if (IsMonostate()) {
    return 0;
  }
  return ap().sextOrTrunc(64).getSExtValue();
}

std::uint64_t IntegerValue::ToUInt() const {
  if (IsMonostate()) {
    return 0;
  }
  return ap().getRawData()[0];
}

// --- Kind helpers -----------------------------------------------------------

unsigned IntegerValue::BitsForKind(int kind) {
  switch (kind) {
  case 1:
    return 8;
  case 2:
    return 16;
  case 3:
    return 16; // bfloat16 shares 16-bit storage with REAL(2)
  case 4:
    return 32;
  case 8:
    return 64;
  case 10:
    return 80;
  case 16:
    return 128;
  default:
    llvm_unreachable("arbritrary bits not yet supported");
    return 32;
  }
}

// Returns this value re-interpreted with a different kind (sign-preserving).
IntegerValue IntegerValue::ConvertToKind(int kind) const {
  unsigned w{BitsForKind(kind)};
  if (IsMonostate()) {
    return Wrap(llvm::APInt(w, 0));
  }
  return Wrap(ap().sextOrTrunc(w));
}

// --- Kind-parameterized ranges ----------------------------------------------

int IntegerValue::RANGE(int kind) { return DecimalRange(kind * 8 - 1); }

int IntegerValue::UnsignedRANGE(int kind) { return DecimalRange(kind * 8); }

// --- Equality ---------------------------------------------------------------

bool IntegerValue::operator==(const IntegerValue &y) const {
  if (IsMonostate() && y.IsMonostate()) {
    return true;
  }
  if (IsMonostate() != y.IsMonostate()) {
    llvm_unreachable("uncomparable integers");
    return false;
  }
  if (width() != y.width()) {
    llvm_unreachable("uncomparable integers");
    return false;
  }
  return ap() == y.ap();
}

// --- Kind / raw access ------------------------------------------------------

int IntegerValue::kind() const {
  if (IsMonostate()) {
    llvm_unreachable("default-initialized value representing 0 with unknown "
                     "width does not know its kind. By definition");
    return 0;
  }
  return static_cast<int>(width()) / 8;
}

int IntegerValue::bits() const {
  if (IsMonostate()) {
    return 0;
  }
  return static_cast<int>(width());
}

IntegerValue IntegerValue::Zero(int kind) {
  return Wrap(llvm::APInt(BitsForKind(kind), 0));
}

// --- Predicates and comparisons ---------------------------------------------

bool IntegerValue::IsZero() const {
  if (IsMonostate()) {
    return true; // uninitialized int representing 0 is zero
  }
  return ap().isZero();
}

bool IntegerValue::IsNegative() const {
  if (IsMonostate()) {
    return false; // uninitialized int representing 0 is not negative
  }
  return ap().isNegative();
}

Ordering IntegerValue::CompareSigned(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return Ordering::Equal;
  }
  // PAPAYA: Use APInt::compareSigned
  llvm::APInt b{coerce(y)};
  if (ap().slt(b)) {
    return Ordering::Less;
  }
  if (ap().sgt(b)) {
    return Ordering::Greater;
  }
  return Ordering::Equal;
}

Ordering IntegerValue::CompareUnsigned(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints; cast bitwidth first");
    return Ordering::Equal;
  }
  // PAPAYA: Use APInt::compare
  llvm::APInt b{coerce(y)};
  if (ap().ult(b)) {
    return Ordering::Less;
  }
  if (ap().ugt(b)) {
    return Ordering::Greater;
  }
  return Ordering::Equal;
}

Ordering IntegerValue::CompareToZeroSigned() const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return Ordering::Equal;
  }
  if (ap().isNegative()) {
    return Ordering::Less;
  }
  if (ap().isZero()) {
    return Ordering::Equal;
  }
  return Ordering::Greater;
}

bool IntegerValue::BGE(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return false;
  }
  return ap().uge(coerce(y));
}

bool IntegerValue::BGT(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return false;
  }
  return ap().ugt(coerce(y));
}

bool IntegerValue::BLE(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return false;
  }
  return ap().ule(coerce(y));
}

bool IntegerValue::BLT(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("uncomparable ints");
    return false;
  }
  return ap().ult(coerce(y));
}

// --- Arithmetic -------------------------------------------------------------

typename IntegerValue::ValueWithOverflow IntegerValue::Negate() const {
  if (IsMonostate()) {
    return ValueWithOverflow{}; // negation of uninitialized int 0 is zero
  }
  // Overflow occurs only for the most negative value (its negation is itself).
  return {Wrap(-ap()), ap().isMinSignedValue()};
}

typename IntegerValue::ValueWithOverflow IntegerValue::ABS() const {
  if (IsMonostate()) {
    return ValueWithOverflow{}; // absolute of uninitialized int 0 is zero
  }
  if (ap().isNegative()) {
    return Negate();
  }
  return {*this, false};
}

typename IntegerValue::ValueWithCarry IntegerValue::AddUnsigned(
    const IntegerValue &y, bool carryIn) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithCarry{};
  }
  bool carry1{false}, carry2{false};
  llvm::APInt sum{ap().uadd_ov(coerce(y), carry1)};
  if (carryIn) {
    sum = sum.uadd_ov(llvm::APInt(width(), 1), carry2);
  }
  return {Wrap(sum), carry1 || carry2};
}

typename IntegerValue::ValueWithOverflow IntegerValue::AddSigned(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return ValueWithOverflow{};
  }
  bool overflow{false};
  llvm::APInt sum{ap().sadd_ov(coerce(y), overflow)};
  return {Wrap(sum), overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::SubtractSigned(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  bool overflow{false};
  llvm::APInt diff{ap().ssub_ov(coerce(y), overflow)};
  return {Wrap(diff), overflow};
}

typename IntegerValue::ValueWithOverflow IntegerValue::AddUnsignedToOverflow(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  bool carry{false};
  llvm::APInt sum{ap().uadd_ov(coerce(y), carry)};
  return {Wrap(sum), carry};
}

typename IntegerValue::ValueWithOverflow IntegerValue::DIM(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  // DIM(X,Y) = MAX(X-Y, 0)
  if (CompareSigned(y) != Ordering::Greater) {
    return {Wrap(llvm::APInt(width(), 0)), false};
  }
  return SubtractSigned(y);
}

typename IntegerValue::ValueWithOverflow IntegerValue::SIGN(
    const IntegerValue &sign) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  bool toNegative{sign.IsNegative()};
  if (toNegative == ap().isNegative()) {
    return {*this, false};
  }
  if (toNegative) {
    return Negate();
  }
  return ABS();
}

typename IntegerValue::Product IntegerValue::MultiplySigned(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return Product{};
  }
  unsigned w{width()};
  llvm::APInt full{ap().sext(2 * w) * coerce(y).sext(2 * w)};
  llvm::APInt lower{full.trunc(w)};
  llvm::APInt upper{full.extractBits(w, w)};
  bool overflow{lower.isNegative()
          ? static_cast<unsigned>(upper.popcount()) != w
          : !upper.isZero()};
  return {Wrap(upper), Wrap(lower), overflow};
}

typename IntegerValue::Product IntegerValue::MultiplyUnsigned(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return Product{};
  }
  unsigned w{width()};
  llvm::APInt full{ap().zext(2 * w) * coerce(y).zext(2 * w)};
  llvm::APInt lower{full.trunc(w)};
  llvm::APInt upper{full.extractBits(w, w)};
  return {Wrap(upper), Wrap(lower), false};
}

typename IntegerValue::QuotientWithRemainder IntegerValue::DivideSigned(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return QuotientWithRemainder{};
  }
  unsigned w{width()};
  llvm::APInt a{ap()};
  llvm::APInt b{coerce(y)};
  if (b.isZero()) {
    // Division by zero saturates toward the sign of the dividend.
    llvm::APInt q{a.isNegative() ? llvm::APInt::getSignedMinValue(w)
                                 : llvm::APInt::getSignedMaxValue(w)};
    return {Wrap(q), Wrap(llvm::APInt(w, 0)), true, false};
  }
  if (a.isMinSignedValue() && b.isAllOnes()) {
    // The sole signed overflow case: most-negative / -1.
    return {Wrap(a), Wrap(llvm::APInt(w, 0)), false, true};
  }
  llvm::APInt q, r;
  llvm::APInt::sdivrem(a, b, q, r);
  return {Wrap(q), Wrap(r), false, false};
}

typename IntegerValue::QuotientWithRemainder IntegerValue::DivideUnsigned(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return QuotientWithRemainder{};
  }
  unsigned w{width()};
  llvm::APInt a{ap()};
  llvm::APInt b{coerce(y)};
  if (b.isZero()) {
    return {
        Wrap(llvm::APInt::getAllOnes(w)), Wrap(llvm::APInt(w, 0)), true, false};
  }
  llvm::APInt q, r;
  llvm::APInt::udivrem(a, b, q, r);
  return {Wrap(q), Wrap(r), false, false};
}

typename IntegerValue::ValueWithOverflow IntegerValue::MODULO(
    const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return ValueWithOverflow{};
  }
  // Result has the sign of the divisor argument.
  bool distinctSigns{ap().isNegative() != coerce(y).isNegative()};
  QuotientWithRemainder divided{DivideSigned(y)};
  if (distinctSigns && !divided.remainder.IsZero()) {
    return {divided.remainder.AddUnsigned(y).value, divided.overflow};
  }
  return {divided.remainder, divided.overflow};
}

typename IntegerValue::PowerWithErrors IntegerValue::Power(
    const IntegerValue &e) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return PowerWithErrors{};
  }
  unsigned w{width()};
  IntegerValue exponent{Wrap(coerce(e))};
  IntegerValue one{Wrap(llvm::APInt(w, 1))};
  IntegerValue negativeOne{Wrap(llvm::APInt::getAllOnes(w))};
  PowerWithErrors result;
  result.power = one;
  if (exponent.IsZero()) {
    // x**0 -> 1 (including 0**0, which other Fortrans also define as 1).
    result.zeroToZero = IsZero();
  } else if (exponent.IsNegative()) {
    if (IsZero()) {
      result.divisionByZero = true;
      result.power = Wrap(llvm::APInt::getSignedMaxValue(w));
    } else if (CompareSigned(one) == Ordering::Equal) {
      result.power = *this; // 1**x -> 1
    } else if (CompareSigned(negativeOne) == Ordering::Equal) {
      if (exponent.BTEST(0)) {
        result.power = *this; // (-1)**x -> -1 if x is odd
      }
    } else {
      result.power = Wrap(llvm::APInt(w, 0)); // |j| > 1 and k < 0 -> 0
    }
  } else {
    IntegerValue shifted{*this};
    int nbits{static_cast<int>(w) - exponent.LEADZ()};
    for (int j{0}; j < nbits; ++j) {
      if (exponent.BTEST(j)) {
        Product product{result.power.MultiplySigned(shifted)};
        result.power = product.lower;
        result.overflow |= product.SignedMultiplicationOverflowed();
      }
      if (j + 1 < nbits) {
        Product squared{shifted.MultiplySigned(shifted)};
        result.overflow |= squared.SignedMultiplicationOverflowed();
        shifted = squared.lower;
      }
    }
  }
  return result;
}

// --- Bitwise ----------------------------------------------------------------

IntegerValue IntegerValue::NOT() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  return Wrap(~ap());
}

IntegerValue IntegerValue::IAND(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatiable ints");
    return IntegerValue{};
  }
  return Wrap(ap() & coerce(y));
}

IntegerValue IntegerValue::IOR(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  return Wrap(ap() | coerce(y));
}

IntegerValue IntegerValue::IEOR(const IntegerValue &y) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  return Wrap(ap() ^ coerce(y));
}

IntegerValue IntegerValue::MERGE_BITS(
    const IntegerValue &y, const IntegerValue &mask) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  llvm::APInt m{coerce(mask)};
  return Wrap((ap() & m) | (coerce(y) & ~m));
}

// --- Shifts and bit queries -------------------------------------------------

IntegerValue IntegerValue::ISHFT(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  return count < 0 ? SHIFTR(-count) : SHIFTL(count);
}

IntegerValue IntegerValue::SHIFTL(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  unsigned w{width()};
  if (count <= 0) {
    return *this;
  }
  if (static_cast<unsigned>(count) >= w) {
    return Wrap(llvm::APInt(w, 0));
  }
  return Wrap(ap().shl(static_cast<unsigned>(count)));
}

IntegerValue IntegerValue::SHIFTR(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  unsigned w{width()};
  if (count <= 0) {
    return *this;
  }
  if (static_cast<unsigned>(count) >= w) {
    return Wrap(llvm::APInt(w, 0));
  }
  return Wrap(ap().lshr(static_cast<unsigned>(count)));
}

IntegerValue IntegerValue::SHIFTA(int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  unsigned w{width()};
  if (count <= 0) {
    return *this;
  }
  if (static_cast<unsigned>(count) >= w) {
    // Saturates to all sign bits.
    return Wrap(
        ap().isNegative() ? llvm::APInt::getAllOnes(w) : llvm::APInt(w, 0));
  }
  return Wrap(ap().ashr(static_cast<unsigned>(count)));
}

IntegerValue IntegerValue::ISHFTC(int count, int size) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  int w{static_cast<int>(width())};
  if (size <= 0) {
    size = w;
  }
  if (count == 0 || size <= 0) {
    return *this;
  }
  if (size > w) {
    size = w;
  }
  count %= size;
  if (count == 0) {
    return *this;
  }
  int middleBits{size - count}, leastBits{count};
  if (count < 0) {
    middleBits = -count;
    leastBits = size + count;
  }
  if (size == w) {
    return SHIFTL(leastBits).IOR(SHIFTR(middleBits));
  }
  IntegerValue unchanged{IAND(Wrap(llvm::APInt::getHighBitsSet(w, w - size)))};
  IntegerValue middle{
      IAND(Wrap(llvm::APInt::getLowBitsSet(w, middleBits))).SHIFTL(leastBits)};
  IntegerValue least{
      SHIFTR(middleBits).IAND(Wrap(llvm::APInt::getLowBitsSet(w, leastBits)))};
  return unchanged.IOR(middle).IOR(least);
}

IntegerValue IntegerValue::IBITS(int pos, int size) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  int w{static_cast<int>(width())};
  int clamped{size < 0 ? 0 : (size > w ? w : size)};
  return SHIFTR(pos).IAND(Wrap(llvm::APInt::getLowBitsSet(w, clamped)));
}

IntegerValue IntegerValue::IBSET(int pos) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  if (pos < 0 || pos >= static_cast<int>(width())) {
    return *this;
  }
  llvm::APInt r{ap()};
  r.setBit(static_cast<unsigned>(pos));
  return Wrap(r);
}

IntegerValue IntegerValue::IBCLR(int pos) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  if (pos < 0 || pos >= static_cast<int>(width())) {
    return *this;
  }
  llvm::APInt r{ap()};
  r.clearBit(static_cast<unsigned>(pos));
  return Wrap(r);
}

IntegerValue IntegerValue::DSHIFTL(const IntegerValue &fill, int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  return ShiftLWithFill(fill, count);
}

IntegerValue IntegerValue::DSHIFTR(const IntegerValue &v2, int count) const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return IntegerValue{};
  }
  // DSHIFTR(I,J): shift I:J right; the *first* argument is the left fill.
  return v2.ShiftRWithFill(*this, count);
}

bool IntegerValue::BTEST(int pos) const {
  if (IsMonostate()) {
    return false; // uninitialized int representing 0 has no bits set
  }
  if (pos < 0 || pos >= static_cast<int>(width())) {
    return false;
  }
  return ap()[static_cast<unsigned>(pos)];
}

int IntegerValue::LEADZ() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return 0;
  }
  return static_cast<int>(ap().countl_zero());
}

int IntegerValue::TRAILZ() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return 0;
  }
  return static_cast<int>(ap().countr_zero());
}

int IntegerValue::POPCNT() const {
  if (IsMonostate()) {
    return 0; // uninitialized int representing 0 has no bits set
  }
  return static_cast<int>(ap().popcount());
}

bool IntegerValue::POPPAR() const {
  if (IsMonostate()) {
    llvm_unreachable("incompatible ints");
    return false;
  }
  return ap().popcount() & 1;
}

// Double shifts with explicit fill (mirrors value::Integer::SHIFT*WithFill).
IntegerValue IntegerValue::ShiftLWithFill(
    const IntegerValue &fill, int count) const {
  int w{static_cast<int>(width())};
  if (count <= 0) {
    return *this;
  }
  if (count >= 2 * w) {
    return Wrap(llvm::APInt(static_cast<unsigned>(w), 0));
  }
  if (count > w) {
    return fill.SHIFTL(count - w);
  }
  if (count == w) {
    return fill;
  }
  return SHIFTL(count).IOR(fill.SHIFTR(w - count));
}

IntegerValue IntegerValue::ShiftRWithFill(
    const IntegerValue &fill, int count) const {
  int w{static_cast<int>(width())};
  if (count <= 0) {
    return *this;
  }
  if (count >= 2 * w) {
    return Wrap(llvm::APInt(static_cast<unsigned>(w), 0));
  }
  if (count > w) {
    return fill.SHIFTR(count - w);
  }
  if (count == w) {
    return fill;
  }
  return SHIFTR(count).IOR(fill.SHIFTL(w - count));
}

// --- Kind-parameterized constants -------------------------------------------

namespace {
IntegerValue MakeMASKR(int places, unsigned bits) {
  if (places <= 0) {
    return IntegerValue::FromAPInt(llvm::APInt(bits, 0));
  }
  if (static_cast<unsigned>(places) >= bits) {
    return IntegerValue::FromAPInt(llvm::APInt::getAllOnes(bits));
  }
  return IntegerValue::FromAPInt(llvm::APInt::getLowBitsSet(bits, places));
}

IntegerValue MakeMASKL(int places, unsigned bits) {
  if (places <= 0) {
    return IntegerValue::FromAPInt(llvm::APInt(bits, 0));
  }
  if (static_cast<unsigned>(places) >= bits) {
    return MakeMASKR(static_cast<int>(bits), bits);
  }
  return IntegerValue::FromAPInt(llvm::APInt::getHighBitsSet(bits, places));
}
} // namespace

IntegerValue IntegerValue::MASKL(int places, int kind) {
  return MakeMASKL(places, BitsForKind(kind));
}

IntegerValue IntegerValue::MASKR(int places, int kind) {
  return MakeMASKR(places, BitsForKind(kind));
}

IntegerValue IntegerValue::HUGE(int kind) {
  unsigned bits{BitsForKind(kind)};
  return MakeMASKR(static_cast<int>(bits) - 1, bits);
}

IntegerValue IntegerValue::Least(int kind) {
  return MakeMASKL(1, BitsForKind(kind));
}

// --- Formatting -------------------------------------------------------------

std::string IntegerValue::SignedDecimal() const {
  if (IsMonostate()) {
    return "0";
  }
  llvm::SmallVector<char, 32> buffer;
  ap().toString(buffer, 10, /*Signed=*/true, /*formatAsCLiteral=*/false);
  return std::string(buffer.data(), buffer.size());
}

std::string IntegerValue::UnsignedDecimal() const {
  if (IsMonostate()) {
    return "0";
  }
  llvm::SmallVector<char, 32> buffer;
  ap().toString(buffer, 10, /*Signed=*/false, /*formatAsCLiteral=*/false);
  return std::string(buffer.data(), buffer.size());
}

std::string IntegerValue::Hexadecimal() const {
  if (IsMonostate()) {
    return "0";
  }
  llvm::SmallVector<char, 32> buffer;
  ap().toString(buffer, 16, /*Signed=*/false, /*formatAsCLiteral=*/false);
  // value::Integer::Hexadecimal emits lower-case digits with no "0x" prefix.
  std::string result(buffer.data(), buffer.size());
  for (char &c : result) {
    if (c >= 'A' && c <= 'F') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return result;
}

// --- Private helpers --------------------------------------------------------

typename IntegerValue::ValueWithOverflow IntegerValue::Read(
    const char *&pp, int base, bool isSigned, int toBits) {
  auto pick = [&](auto target) -> ValueWithOverflow {
    using T = decltype(target);
    auto r{T::Read(pp, base, isSigned)};
    return {Wrap(IntegerToAP(r.value)), r.overflow};
  };
  switch (toBits) {
  case 8:
    return pick(Integer<8>{});
  case 16:
    return pick(Integer<16>{});
  case 32:
    return pick(Integer<32>{});
  case 64:
    return pick(Integer<64>{});
  case 80:
    return pick(X87IntegerContainer{});
  case 128:
    return pick(Integer<128>{});
  default:
    llvm_unreachable("arbitrary precisions not supported");
    return pick(Integer<128>{});
  }
}

IntegerValue::ValueWithOverflow IntegerValue::ConvertAP(
    const llvm::APInt &src, int toBits, bool isSigned) {
  unsigned to{static_cast<unsigned>(toBits)};
  llvm::APInt value{isSigned ? src.sextOrTrunc(to) : src.zextOrTrunc(to)};
  bool overflow;
  if (isSigned) {
    overflow = src.getBitWidth() > to && value.sext(src.getBitWidth()) != src;
  } else {
    overflow = src.getActiveBits() > to;
  }
  return {Wrap(value), overflow};
}

IntegerValue::ValueWithOverflow IntegerValue::ConvertSigned(
    const IntegerValue &from, int toBits) {
  if (from.IsMonostate()) {
    return {};
  }
  return ConvertAP(from.ap(), toBits, /*isSigned=*/true);
}

IntegerValue::ValueWithOverflow IntegerValue::ConvertUnsigned(
    const IntegerValue &from, int toBits) {
  if (from.IsMonostate()) {
    return {};
  }
  return ConvertAP(from.ap(), toBits, /*isSigned=*/false);
}

} // namespace Fortran::evaluate::value
