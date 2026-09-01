//===-- flang/unittests/Evaluate/IntegerValueTest.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/Fortran-consts.h"
#include "flang/Common/template.h"
#include "flang/Common/type-kinds.h"
#include "flang/Common/uint128.h"
#include "flang/Evaluate/integer-value.h"
#include "flang/Evaluate/typekind-traits.h"
#include "llvm/ADT/Sequence.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace Fortran::common;
using namespace Fortran::evaluate;
using namespace Fortran::evaluate::value;

namespace {

//===----------------------------------------------------------------------===//
// Parameterization over the INTEGER kinds
//===----------------------------------------------------------------------===//

using IntegerTypedKinds = testing::Types<TypeKind<TypeCategory::Integer, 1>,
    TypeKind<TypeCategory::Integer, 2>, TypeKind<TypeCategory::Integer, 4>,
    TypeKind<TypeCategory::Integer, 8>, TypeKind<TypeCategory::Integer, 16>>;
template <typename Target>
inline constexpr std::size_t IntKindPos =
    type_index<Target, IntegerTypedKinds>::value;
struct KindName {
  template <typename TK> static std::string GetName(int) {
    return "INTEGER_" + std::to_string(TK::kind);
  }
};

template <typename T> class IntegerValueTypedKind : public testing::Test {};
TYPED_TEST_SUITE(IntegerValueTypedKind, IntegerTypedKinds, KindName);

class IntegerValueKind : public testing::TestWithParam<int> {};
INSTANTIATE_TEST_SUITE_P(IntegerValueKind, IntegerValueKind,
    testing::ValuesIn(KindsByType<TypeCategory::Integer>::kinds),
    [](const testing::TestParamInfo<int> &info) {
      return "INTEGER_" + std::to_string(info.param);
    });

//===----------------------------------------------------------------------===//
// Construction, assignment and kind inquiries
//===----------------------------------------------------------------------===//

TEST(IntegerValue, Monostate) {
  IntegerValue x;
  EXPECT_TRUE(x.IsMonostate());
  EXPECT_TRUE(x.IsZero());
  EXPECT_FALSE(x.IsNegative());
  EXPECT_EQ(0u, x.ToUInt64());
  EXPECT_EQ(0, x.ToInt64());
  EXPECT_EQ(0, x.POPCNT());
  EXPECT_FALSE(x.BTEST(0));
  EXPECT_EQ("0", x.SignedDecimal());
  EXPECT_EQ("0", x.UnsignedDecimal());
  EXPECT_EQ("0", x.Hexadecimal());
}

TYPED_TEST(IntegerValueTypedKind, ConstructFromIntegral) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue positive{kind, 42};
  EXPECT_EQ(42, positive.ToInt64());
  IntegerValue negative{kind, -42};
  EXPECT_EQ(-42, negative.ToInt64());

  // The signedness of the C++ operand decides between sign- and zero-extension.
  IntegerValue sext{kind, int8_t{-1}};
  EXPECT_EQ(SignedT{-1}, sext.ToSInt<SignedT>());
  EXPECT_EQ(std::numeric_limits<UnsignedT>::max(), sext.ToUInt<UnsignedT>());
  IntegerValue zext{kind, uint8_t{255}};
  EXPECT_EQ(UnsignedT{255}, zext.ToUInt<UnsignedT>());

  // A value too wide for the kind is truncated silently.
  constexpr uint64_t w{0x123456789abcdefu};
  IntegerValue wide{kind, w};
  EXPECT_EQ(UnsignedT(w), wide.ToUInt<UnsignedT>());
  EXPECT_EQ(SignedT(w), wide.ToSInt<SignedT>());
  EXPECT_EQ(UnsignedT(w), wide.ToUInt<UnsignedT>());
  EXPECT_EQ(SignedT(w), wide.ToSInt<SignedT>());
}

TEST_P(IntegerValueKind, CopyAndMove) {
  const int kind{GetParam()};
  const IntegerValue x{IntegerValue::HUGE(kind)};

  IntegerValue copyConstructed{x};
  EXPECT_EQ(kind, copyConstructed.kind());
  EXPECT_EQ(x, copyConstructed);

  IntegerValue copyAssigned;
  copyAssigned = x;
  EXPECT_EQ(kind, copyAssigned.kind());
  EXPECT_EQ(x, copyAssigned);

  IntegerValue moveConstructed{std::move(copyConstructed)};
  EXPECT_EQ(kind, moveConstructed.kind());
  EXPECT_EQ(x, moveConstructed);

  IntegerValue moveAssigned;
  moveAssigned = std::move(copyAssigned);
  EXPECT_EQ(kind, moveAssigned.kind());
  EXPECT_EQ(x, moveAssigned);
}

TYPED_TEST(IntegerValueTypedKind, KindCheckingConstructors) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue x{kind, 7};
  IntegerValue copied{kind, x};
  EXPECT_EQ(SignedT(7), copied.ToSInt<SignedT>());
  EXPECT_EQ(UnsignedT(7), copied.ToUInt<UnsignedT>());

  IntegerValue y{kind, 7};
  IntegerValue moved{kind, std::move(y)};
  EXPECT_EQ(kind, moved.kind());
  EXPECT_EQ(SignedT(7), moved.ToSInt<SignedT>());
  EXPECT_EQ(UnsignedT(7), moved.ToUInt<UnsignedT>());
}

TYPED_TEST(IntegerValueTypedKind, Zero) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(kind, zero.kind());
  EXPECT_FALSE(zero.IsMonostate());
  EXPECT_TRUE(zero.IsZero());
  EXPECT_EQ(SignedT(0), zero.ToSInt<SignedT>());
}

TEST(IntegerValue, Bits) {
  EXPECT_EQ(8, IntegerValue::bits(1));
  EXPECT_EQ(16, IntegerValue::bits(2));
  EXPECT_EQ(16, IntegerValue::bits(3));
  EXPECT_EQ(32, IntegerValue::bits(4));
  EXPECT_EQ(64, IntegerValue::bits(8));
  EXPECT_EQ(128, IntegerValue::bits(10)); // 80 significant bits, 128 stored
  EXPECT_EQ(128, IntegerValue::bits(16));

  IntegerValue v{4, 0};
  EXPECT_EQ(32, v.bits());
}

TEST(IntegerValue, BytesStored) {
  EXPECT_EQ(1u, IntegerValue::bytesStored(1));
  EXPECT_EQ(2u, IntegerValue::bytesStored(2));
  EXPECT_EQ(2u, IntegerValue::bytesStored(3));
  EXPECT_EQ(4u, IntegerValue::bytesStored(4));
  EXPECT_EQ(8u, IntegerValue::bytesStored(8));
  EXPECT_EQ(16u, IntegerValue::bytesStored(10));
  EXPECT_EQ(16u, IntegerValue::bytesStored(16));

  IntegerValue v{4, 0};
  EXPECT_EQ(4u, v.bytesStored());
}

TYPED_TEST(IntegerValueTypedKind, DIGITS) {
  constexpr int kind{TypeParam::kind};
  EXPECT_EQ(TypeParam::bits - 1, IntegerValue::DIGITS(kind));
}

TEST(IntegerValue, RANGE) {
  EXPECT_EQ(2, IntegerValue::RANGE(1));
  EXPECT_EQ(4, IntegerValue::RANGE(2));
  EXPECT_EQ(9, IntegerValue::RANGE(4));
  EXPECT_EQ(18, IntegerValue::RANGE(8));
  EXPECT_EQ(38, IntegerValue::RANGE(16));
}

TEST(IntegerValue, UnsignedRANGE) {
  EXPECT_EQ(2, IntegerValue::UnsignedRANGE(1));
  EXPECT_EQ(4, IntegerValue::UnsignedRANGE(2));
  EXPECT_EQ(9, IntegerValue::UnsignedRANGE(4));
  EXPECT_EQ(19, IntegerValue::UnsignedRANGE(8));
  EXPECT_EQ(38, IntegerValue::UnsignedRANGE(16));
}

//===----------------------------------------------------------------------===//
// Formatting and parsing
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, UnsignedDecimal) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ("0", zero.UnsignedDecimal());

  IntegerValue one{kind, 1};
  EXPECT_EQ("1", one.UnsignedDecimal());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ("42", theanswer.UnsignedDecimal());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  static constexpr const char *maxstr[]{"255", "65535", "4294967295",
      "18446744073709551615", "340282366920938463463374607431768211455"};
  EXPECT_EQ(maxstr[IntKindPos<TypeParam>], maxv.UnsignedDecimal());

  IntegerValue beforemaxv{kind, std::numeric_limits<UnsignedT>::max() - 1};
  static constexpr const char *beforemaxstr[]{"254", "65534", "4294967294",
      "18446744073709551614", "340282366920938463463374607431768211454"};
  EXPECT_EQ(beforemaxstr[IntKindPos<TypeParam>], beforemaxv.UnsignedDecimal());

  IntegerValue hugev{kind, IntegerValue::HUGE(kind)};
  static constexpr const char *hugestr[]{"127", "32767", "2147483647",
      "9223372036854775807", "170141183460469231731687303715884105727"};
  EXPECT_EQ(hugestr[IntKindPos<TypeParam>], hugev.UnsignedDecimal());

  IntegerValue leastv{kind, IntegerValue::Least(kind)};
  static constexpr const char *leaststr[]{"128", "32768", "2147483648",
      "9223372036854775808", "170141183460469231731687303715884105728"};
  EXPECT_EQ(leaststr[IntKindPos<TypeParam>], leastv.UnsignedDecimal());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  static constexpr const char *patternstr[]{
      "239", "52719", "2309737967", "81985529216486895", "81985529216486895"};
  EXPECT_EQ(patternstr[IntKindPos<TypeParam>], patternv.UnsignedDecimal());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  static constexpr const char *invpatternstr[]{"16", "12816", "1985229328",
      "18364758544493064720", "340282366920938463463292621902551724560"};
  EXPECT_EQ(
      invpatternstr[IntKindPos<TypeParam>], invpatternv.UnsignedDecimal());
}

TYPED_TEST(IntegerValueTypedKind, SignedDecimal) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ("0", zero.SignedDecimal());

  IntegerValue one{kind, 1};
  EXPECT_EQ("1", one.SignedDecimal());

  IntegerValue minusone{kind, -1};
  EXPECT_EQ("-1", minusone.SignedDecimal());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ("42", theanswer.SignedDecimal());

  IntegerValue maxv{kind, std::numeric_limits<SignedT>::max()};
  static constexpr const char *maxstr[]{"127", "32767", "2147483647",
      "9223372036854775807", "170141183460469231731687303715884105727"};
  EXPECT_EQ(maxstr[IntKindPos<TypeParam>], maxv.SignedDecimal());

  IntegerValue beforemaxv{kind, std::numeric_limits<SignedT>::max() - 1};
  static constexpr const char *beforemaxstr[]{"126", "32766", "2147483646",
      "9223372036854775806", "170141183460469231731687303715884105726"};
  EXPECT_EQ(beforemaxstr[IntKindPos<TypeParam>], beforemaxv.SignedDecimal());

  IntegerValue hugev{kind, IntegerValue::HUGE(kind)};
  static constexpr const char *hugestr[]{"127", "32767", "2147483647",
      "9223372036854775807", "170141183460469231731687303715884105727"};
  EXPECT_EQ(hugestr[IntKindPos<TypeParam>], hugev.SignedDecimal());

  IntegerValue leastv{kind, IntegerValue::Least(kind)};
  static constexpr const char *leaststr[]{"-128", "-32768", "-2147483648",
      "-9223372036854775808", "-170141183460469231731687303715884105728"};
  EXPECT_EQ(leaststr[IntKindPos<TypeParam>], leastv.SignedDecimal());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  static constexpr const char *patternstr[]{
      "-17", "-12817", "-1985229329", "81985529216486895", "81985529216486895"};
  EXPECT_EQ(patternstr[IntKindPos<TypeParam>], patternv.SignedDecimal());

  IntegerValue invpatternv{kind, ~SignedT(0x0123456789abcdefull)};
  static constexpr const char *invpatternstr[]{
      "16", "12816", "1985229328", "-81985529216486896", "-81985529216486896"};
  EXPECT_EQ(invpatternstr[IntKindPos<TypeParam>], invpatternv.SignedDecimal());
}

TYPED_TEST(IntegerValueTypedKind, Hexadecimal) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ("0", zero.Hexadecimal());

  IntegerValue one{kind, 1};
  EXPECT_EQ("1", one.Hexadecimal());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ("2a", theanswer.Hexadecimal());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  static constexpr const char *maxstr[]{"ff", "ffff", "ffffffff",
      "ffffffffffffffff", "ffffffffffffffffffffffffffffffff"};
  EXPECT_EQ(maxstr[IntKindPos<TypeParam>], maxv.Hexadecimal());

  IntegerValue beforemaxv{kind, std::numeric_limits<UnsignedT>::max() - 1};
  static constexpr const char *beforemaxstr[]{"fe", "fffe", "fffffffe",
      "fffffffffffffffe", "fffffffffffffffffffffffffffffffe"};
  EXPECT_EQ(beforemaxstr[IntKindPos<TypeParam>], beforemaxv.Hexadecimal());

  IntegerValue hugev{kind, IntegerValue::HUGE(kind)};
  static constexpr const char *hugestr[]{"7f", "7fff", "7fffffff",
      "7fffffffffffffff", "7fffffffffffffffffffffffffffffff"};
  EXPECT_EQ(hugestr[IntKindPos<TypeParam>], hugev.Hexadecimal());

  IntegerValue leastv{kind, IntegerValue::Least(kind)};
  static constexpr const char *leaststr[]{"80", "8000", "80000000",
      "8000000000000000", "80000000000000000000000000000000"};
  EXPECT_EQ(leaststr[IntKindPos<TypeParam>], leastv.Hexadecimal());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  static constexpr const char *patternstr[]{
      "ef", "cdef", "89abcdef", "123456789abcdef", "123456789abcdef"};
  EXPECT_EQ(patternstr[IntKindPos<TypeParam>], patternv.Hexadecimal());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  static constexpr const char *invpatternstr[]{"10", "3210", "76543210",
      "fedcba9876543210", "fffffffffffffffffedcba9876543210"};
  EXPECT_EQ(invpatternstr[IntKindPos<TypeParam>], invpatternv.Hexadecimal());
}

TYPED_TEST(IntegerValueTypedKind, Read) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  {
    // Leading blanks are skipped and trailing text is left for the caller.
    const char *p{"  42tail"};
    auto decimal{IntegerValue::Read(kind, p, 10, /*isSigned=*/false)};
    EXPECT_FALSE(decimal.overflow);
    EXPECT_EQ(kind, decimal.value.kind());
    EXPECT_EQ(UnsignedT(42), decimal.value.ToUInt<UnsignedT>());
    EXPECT_STREQ("tail", p);
  }

  {
    const char *p{"  -42tail"};
    auto decimal{IntegerValue::Read(kind, p, 10, /*isSigned=*/true)};
    EXPECT_FALSE(decimal.overflow);
    EXPECT_EQ(kind, decimal.value.kind());
    EXPECT_EQ(SignedT(-42), decimal.value.ToSInt<SignedT>());
    EXPECT_STREQ("tail", p);
  }

  {
    const char *p{"-42"};
    auto decimal{IntegerValue::Read(kind, p, 10, /*isSigned=*/false)};
    EXPECT_FALSE(decimal.overflow);
    EXPECT_EQ(kind, decimal.value.kind());
    EXPECT_EQ(UnsignedT(-42), decimal.value.ToUInt<UnsignedT>());
    EXPECT_STREQ("", p);
  }

  {
    // More f's than can fit into the largest unsigned int
    const char *p = "fffffffffffffffffffffffffffffffff";
    auto unsignedRead{
        IntegerValue::Read(kind, p, /*base=*/16, /*isSigned=*/false)};
    EXPECT_TRUE(unsignedRead.overflow);
    EXPECT_EQ(kind, unsignedRead.value.kind());
    EXPECT_EQ(std::numeric_limits<UnsignedT>::max(),
        unsignedRead.value.ToUInt<UnsignedT>());
    EXPECT_EQ(p[0], '\0');
  }

  {
    // Fits unsigned representations, but not signed
    static constexpr const char *signedstr[]{"ff", "ffff", "ffffffff",
        "ffffffffffffffff", "ffffffffffffffffffffffffffffffff"};
    const char *p = signedstr[IntKindPos<TypeParam>];
    auto signedRead{
        IntegerValue::Read(kind, p, /*base=*/16, /*isSigned=*/true)};
    EXPECT_TRUE(signedRead.overflow);
    EXPECT_EQ(kind, signedRead.value.kind());
    EXPECT_EQ(SignedT(-1), signedRead.value.ToSInt<SignedT>());
    EXPECT_EQ(p[0], '\0');
  }
}

//===----------------------------------------------------------------------===//
// Bit masks and kind-specific constants
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, MASKL) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};
  static constexpr UnsignedT nobits{0};
  static constexpr UnsignedT allbits{UnsignedT(~UnsignedT(0))};

  IntegerValue signBit{IntegerValue::MASKL(kind, 1)};
  EXPECT_EQ(kind, signBit.kind());
  EXPECT_EQ(1, signBit.POPCNT());
  EXPECT_TRUE(signBit.IsNegative());
  EXPECT_EQ(0, signBit.LEADZ());

  IntegerValue maskedunderflow{IntegerValue::MASKL(kind, -1)};
  EXPECT_EQ(kind, maskedunderflow.kind());
  EXPECT_EQ(nobits, maskedunderflow.ToUInt<UnsignedT>());

  IntegerValue nomask{IntegerValue::MASKL(kind, 0)};
  EXPECT_EQ(kind, nomask.kind());
  EXPECT_EQ(nobits, nomask.ToUInt<UnsignedT>());

  for (auto places : llvm::seq<int>(1, bits)) {
    IntegerValue masked{IntegerValue::MASKL(kind, places)};
    UnsignedT reference =
        UnsignedT(UnsignedT(~UnsignedT(0)) << (bits - places));
    EXPECT_EQ(kind, masked.kind());
    EXPECT_EQ(reference, masked.ToUInt<UnsignedT>()) << "places=" << places;
  }

  IntegerValue fullmask{IntegerValue::MASKL(kind, bits)};
  EXPECT_EQ(kind, fullmask.kind());
  EXPECT_EQ(allbits, fullmask.ToUInt<UnsignedT>());

  IntegerValue maskedoverflow{IntegerValue::MASKL(kind, bits + 1)};
  EXPECT_EQ(kind, maskedoverflow.kind());
  EXPECT_EQ(allbits, maskedoverflow.ToUInt<UnsignedT>());
}

TYPED_TEST(IntegerValueTypedKind, MASKR) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};
  static constexpr UnsignedT nobits{0};
  static constexpr UnsignedT allbits{UnsignedT(~UnsignedT(0))};

  IntegerValue maskedunderflow{IntegerValue::MASKR(kind, -1)};
  EXPECT_EQ(kind, maskedunderflow.kind());
  EXPECT_EQ(nobits, maskedunderflow.ToUInt<UnsignedT>());

  IntegerValue nomask{IntegerValue::MASKR(kind, 0)};
  EXPECT_EQ(kind, nomask.kind());
  EXPECT_EQ(nobits, nomask.ToUInt<UnsignedT>());

  for (auto places : llvm::seq<int>(1, bits)) {
    IntegerValue masked{IntegerValue::MASKR(kind, places)};
    UnsignedT reference = allbits >> (bits - places);
    EXPECT_EQ(kind, masked.kind());
    EXPECT_EQ(reference, masked.ToUInt<UnsignedT>()) << "places=" << places;
  }

  IntegerValue fullmask{IntegerValue::MASKR(kind, bits)};
  EXPECT_EQ(kind, fullmask.kind());
  EXPECT_EQ(allbits, fullmask.ToUInt<UnsignedT>());

  IntegerValue maskedoverflow{IntegerValue::MASKR(kind, bits + 1)};
  EXPECT_EQ(kind, maskedoverflow.kind());
  EXPECT_EQ(allbits, maskedoverflow.ToUInt<UnsignedT>());
}

TYPED_TEST(IntegerValueTypedKind, HUGE) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue huge{IntegerValue::HUGE(kind)};
  EXPECT_EQ(kind, huge.kind());
  EXPECT_EQ(std::numeric_limits<SignedT>::max(), huge.ToSInt<SignedT>());
}

TYPED_TEST(IntegerValueTypedKind, Least) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue least{IntegerValue::Least(kind)};
  EXPECT_EQ(kind, least.kind());
  EXPECT_EQ(std::numeric_limits<SignedT>::min(), least.ToSInt<SignedT>());
}

//===----------------------------------------------------------------------===//
// Predicates and comparisons
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, IsZero) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_TRUE(zero.IsZero());

  IntegerValue one{kind, 1};
  EXPECT_FALSE(one.IsZero());

  IntegerValue negone{kind, -1};
  EXPECT_FALSE(negone.IsZero());

  IntegerValue theanswer{kind, 42};
  EXPECT_FALSE(theanswer.IsZero());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_FALSE(maxv.IsZero());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_TRUE(minv.IsZero());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_FALSE(smaxv.IsZero());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_FALSE(smaxv.IsZero());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_FALSE(patternv.IsZero());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_FALSE(invpatternv.IsZero());
}

TYPED_TEST(IntegerValueTypedKind, IsNegative) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_FALSE(zero.IsNegative());

  IntegerValue one{kind, 1};
  EXPECT_FALSE(one.IsNegative());

  IntegerValue negone{kind, -1};
  EXPECT_TRUE(negone.IsNegative());

  IntegerValue theanswer{kind, 42};
  EXPECT_FALSE(theanswer.IsNegative());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_TRUE(maxv.IsNegative());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_FALSE(minv.IsNegative());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_FALSE(smaxv.IsNegative());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_FALSE(smaxv.IsNegative());

  IntegerValue patternv{kind, 0x7FFFFFFF7FFF7F7Full};
  EXPECT_FALSE(patternv.IsNegative());

  IntegerValue invpatternv{kind, ~UnsignedT(0x7FFFFFFF7FFF7F7Full)};
  EXPECT_TRUE(invpatternv.IsNegative());
}

TYPED_TEST(IntegerValueTypedKind, CompareToZeroSigned) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ(Ordering::Equal, zero.CompareToZeroSigned());

  IntegerValue one{kind, 1};
  EXPECT_EQ(Ordering::Greater, one.CompareToZeroSigned());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(Ordering::Less, negone.CompareToZeroSigned());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(Ordering::Greater, theanswer.CompareToZeroSigned());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_EQ(Ordering::Less, maxv.CompareToZeroSigned());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_EQ(Ordering::Equal, minv.CompareToZeroSigned());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(Ordering::Greater, smaxv.CompareToZeroSigned());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(Ordering::Less, sminv.CompareToZeroSigned());

  IntegerValue patternv{kind, 0x7FFFFFFF7FFF7F7Full};
  EXPECT_EQ(Ordering::Greater, patternv.CompareToZeroSigned());

  IntegerValue invpatternv{kind, ~UnsignedT(0x7FFFFFFF7FFF7F7Full)};
  EXPECT_EQ(Ordering::Less, invpatternv.CompareToZeroSigned());
}

TYPED_TEST(IntegerValueTypedKind, LEADZ) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ(bits, zero.LEADZ());

  IntegerValue one{kind, 1};
  EXPECT_EQ(bits - 1, one.LEADZ());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(0, negone.LEADZ());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(bits - 6, theanswer.LEADZ());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_EQ(0, maxv.LEADZ());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_EQ(bits, minv.LEADZ());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(1, smaxv.LEADZ());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(0, sminv.LEADZ());

  IntegerValue patternv{kind, 0x7FFFFFFF7FFF7F7Full};
  EXPECT_EQ((kind == 16) ? 65 : 1, patternv.LEADZ());

  IntegerValue invpatternv{kind, ~UnsignedT(0x7FFFFFFF7FFF7F7Full)};
  EXPECT_EQ(0, invpatternv.LEADZ());
}

TYPED_TEST(IntegerValueTypedKind, POPCNT) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ(0, zero.POPCNT());

  IntegerValue one{kind, 1};
  EXPECT_EQ(1, one.POPCNT());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(bits, negone.POPCNT());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(3, theanswer.POPCNT());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_EQ(bits, maxv.POPCNT());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_EQ(0, minv.POPCNT());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(bits - 1, smaxv.POPCNT());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(1, sminv.POPCNT());

  IntegerValue patternv{kind, 0x7FFFFFFF7FFF7F7Full};
  const int kindPos{IntKindPos<TypeParam>};
  EXPECT_EQ((kind == 16) ? 60 : bits - kindPos - 1, patternv.POPCNT());

  IntegerValue invpatternv{kind, ~UnsignedT(0x7FFFFFFF7FFF7F7Full)};
  EXPECT_EQ((kind == 16) ? 68 : 1 + kindPos, invpatternv.POPCNT());
}

TYPED_TEST(IntegerValueTypedKind, POPPAR) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_FALSE(zero.POPPAR());

  IntegerValue one{kind, 1};
  EXPECT_TRUE(one.POPPAR());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(false, negone.POPPAR());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(true, theanswer.POPPAR());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_EQ(bits & 1, maxv.POPPAR());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_EQ(0, minv.POPPAR());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(bits % 2 == 0, smaxv.POPPAR());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(true, sminv.POPPAR());

  IntegerValue patternv{kind, 0x5555555555555554ull};
  EXPECT_TRUE(patternv.POPPAR());

  IntegerValue invpatternv{kind, 0xAAAAAAAAAAAAAAABull};
  EXPECT_TRUE(invpatternv.POPPAR());
}

TYPED_TEST(IntegerValueTypedKind, TRAILZ) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue zero{IntegerValue::Zero(TypeParam::kind)};
  EXPECT_EQ(bits, zero.TRAILZ());

  IntegerValue one{kind, 1};
  EXPECT_EQ(0, one.TRAILZ());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(0, negone.TRAILZ());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(1, theanswer.TRAILZ());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_EQ(0, maxv.TRAILZ());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_EQ(bits, minv.TRAILZ());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(0, smaxv.TRAILZ());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(bits - 1, sminv.TRAILZ());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_EQ(0, patternv.TRAILZ());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_EQ(4, invpatternv.TRAILZ());
}

TYPED_TEST(IntegerValueTypedKind, BTEST) {
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue zero{IntegerValue::Zero(kind)};
  for (auto place : llvm::seq<int>(-2, bits + 2)) {
    EXPECT_FALSE(zero.BTEST(place)) << "place=" << place;
  }

  // Out-of-range positions read as clear.
  IntegerValue negone{kind, -1};
  EXPECT_FALSE(negone.BTEST(-1));
  for (auto place : llvm::seq<int>(0, bits)) {
    EXPECT_TRUE(negone.BTEST(place)) << "place=" << place;
  }
  EXPECT_FALSE(negone.BTEST(bits));
}

TYPED_TEST(IntegerValueTypedKind, CompareUnsigned) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(Ordering::Equal, zero.CompareUnsigned(zero));

  IntegerValue one{kind, 1};
  EXPECT_EQ(Ordering::Less, zero.CompareUnsigned(one));
  EXPECT_EQ(Ordering::Greater, one.CompareUnsigned(zero));

  // -1 is the largest unsigned value.
  IntegerValue negone{kind, -1};
  EXPECT_EQ(Ordering::Less, one.CompareUnsigned(negone));
  EXPECT_EQ(Ordering::Greater, negone.CompareUnsigned(one));

  // As an unsigned pattern, the sign bit outweighs the rest of the word.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(Ordering::Less, smaxv.CompareUnsigned(sminv));

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(Ordering::Equal, theanswer.CompareUnsigned(theanswer));
  EXPECT_EQ(Ordering::Less, one.CompareUnsigned(theanswer));
}

TYPED_TEST(IntegerValueTypedKind, CompareSigned) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(Ordering::Equal, zero.CompareSigned(zero));

  IntegerValue one{kind, 1};
  EXPECT_EQ(Ordering::Less, zero.CompareSigned(one));
  EXPECT_EQ(Ordering::Greater, one.CompareSigned(zero));

  IntegerValue negone{kind, -1};
  EXPECT_EQ(Ordering::Less, negone.CompareSigned(one));
  EXPECT_EQ(Ordering::Greater, one.CompareSigned(negone));

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(Ordering::Greater, smaxv.CompareSigned(sminv));

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(Ordering::Equal, theanswer.CompareSigned(theanswer));
  EXPECT_EQ(Ordering::Less, one.CompareSigned(theanswer));
}

TYPED_TEST(IntegerValueTypedKind, BitwiseComparisons) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue one{kind, 1};
  EXPECT_FALSE(zero.BGE(one));
  EXPECT_FALSE(zero.BGT(one));
  EXPECT_TRUE(zero.BLE(one));
  EXPECT_TRUE(zero.BLT(one));

  IntegerValue negone{kind, -1};
  EXPECT_TRUE(negone.BGE(one));
  EXPECT_TRUE(negone.BGT(one));
  EXPECT_FALSE(negone.BLE(one));
  EXPECT_FALSE(negone.BLT(one));

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_FALSE(smaxv.BGE(sminv));
  EXPECT_FALSE(smaxv.BGT(sminv));
  EXPECT_TRUE(smaxv.BLE(sminv));
  EXPECT_TRUE(smaxv.BLT(sminv));

  IntegerValue theanswer{kind, 42};
  EXPECT_TRUE(theanswer.BGE(theanswer));
  EXPECT_FALSE(theanswer.BGT(theanswer));
  EXPECT_TRUE(theanswer.BLE(theanswer));
  EXPECT_FALSE(theanswer.BLT(theanswer));
}

TYPED_TEST(IntegerValueTypedKind, RelationalOperators) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_FALSE(zero < zero);
  EXPECT_TRUE(zero <= zero);
  EXPECT_TRUE(zero == zero);
  EXPECT_FALSE(zero != zero);
  EXPECT_TRUE(zero >= zero);
  EXPECT_FALSE(zero > zero);

  IntegerValue one{kind, 1};
  EXPECT_TRUE(zero < one);
  EXPECT_TRUE(zero <= one);
  EXPECT_FALSE(zero == one);
  EXPECT_TRUE(zero != one);
  EXPECT_FALSE(zero >= one);
  EXPECT_FALSE(zero > one);

  IntegerValue negone{kind, -1};
  EXPECT_TRUE(negone < one);
  EXPECT_TRUE(negone <= one);
  EXPECT_FALSE(negone == one);
  EXPECT_TRUE(negone != one);
  EXPECT_FALSE(negone >= one);
  EXPECT_FALSE(negone > one);

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_FALSE(smaxv < sminv);
  EXPECT_FALSE(smaxv <= sminv);
  EXPECT_FALSE(smaxv == sminv);
  EXPECT_TRUE(smaxv != sminv);
  EXPECT_TRUE(smaxv >= sminv);
  EXPECT_TRUE(smaxv > sminv);

  IntegerValue theanswer{kind, 42};
  EXPECT_TRUE(one < theanswer);
  EXPECT_TRUE(one <= theanswer);
  EXPECT_FALSE(one == theanswer);
  EXPECT_TRUE(one != theanswer);
  EXPECT_FALSE(one >= theanswer);
  EXPECT_FALSE(one > theanswer);
}

TYPED_TEST(IntegerValueTypedKind, ToUInt64) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(0u, zero.ToUInt64());

  IntegerValue one{kind, 1};
  EXPECT_EQ(1u, one.ToUInt64());

  // Only the least-significant 64 bits of a value survive conversion to a host
  // 64-bit integer; wider kinds can therefore lose information.
  IntegerValue negone{kind, -1};
  static constexpr uint64_t moneu64[]{255ull, 65535ull, 4294967295ull,
      18446744073709551615ull, 18446744073709551615ull};
  EXPECT_EQ(moneu64[IntKindPos<TypeParam>], negone.ToUInt64());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(42u, theanswer.ToUInt64());

  IntegerValue maxv{kind, std::numeric_limits<UnsignedT>::max()};
  EXPECT_EQ(moneu64[IntKindPos<TypeParam>], maxv.ToUInt64());

  IntegerValue minv{kind, std::numeric_limits<UnsignedT>::min()};
  EXPECT_EQ(0u, minv.ToUInt64());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  static constexpr uint64_t smaxu64[]{127ull, 32767ull, 2147483647ull,
      9223372036854775807ull, 18446744073709551615ull};
  EXPECT_EQ(smaxu64[IntKindPos<TypeParam>], smaxv.ToUInt64());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  static constexpr uint64_t sminu64[]{
      128ull, 32768ull, 2147483648ull, 9223372036854775808ull, 0ull};
  EXPECT_EQ(sminu64[IntKindPos<TypeParam>], sminv.ToUInt64());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  static constexpr uint64_t patternu64[]{239ull, 52719ull, 2309737967ull,
      81985529216486895ull, 81985529216486895ull};
  EXPECT_EQ(patternu64[IntKindPos<TypeParam>], patternv.ToUInt64());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  static constexpr uint64_t invpatternu64[]{16ull, 12816ull, 1985229328ull,
      18364758544493064720ull, 18364758544493064720ull};
  EXPECT_EQ(invpatternu64[IntKindPos<TypeParam>], invpatternv.ToUInt64());
}

TYPED_TEST(IntegerValueTypedKind, ToInt64) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(0, zero.ToInt64());
  EXPECT_EQ(0, zero.template ToSInt<int64_t>()); // a synonym

  IntegerValue one{kind, 1};
  EXPECT_EQ(1, one.ToInt64());

  // -1 is all-ones regardless of width, so its low 64 bits read back as -1.
  IntegerValue negone{kind, -1};
  EXPECT_EQ(-1, negone.ToInt64());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(42, theanswer.ToInt64());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  static constexpr int64_t smaxi64[]{
      127, 32767, 2147483647, 9223372036854775807ll, -1ll};
  EXPECT_EQ(smaxi64[IntKindPos<TypeParam>], smaxv.ToInt64());

  // For kinds up to 8 bytes, ToInt64() recovers the exact signed value.
  // For 16-byte kind, only the low 8 bytes survive, reread as signed.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  static constexpr int64_t smini64[]{
      -128, -32768, -2147483648ll, -9223372036854775807ll - 1, 0};
  EXPECT_EQ(smini64[IntKindPos<TypeParam>], sminv.ToInt64());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  static constexpr int64_t patterni64[]{
      -17, -12817, -1985229329, 81985529216486895ll, 81985529216486895ll};
  EXPECT_EQ(patterni64[IntKindPos<TypeParam>], patternv.ToInt64());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  static constexpr int64_t invpatterni64[]{
      16, 12816, 1985229328, -81985529216486896ll, -81985529216486896ll};
  EXPECT_EQ(invpatterni64[IntKindPos<TypeParam>], invpatternv.ToInt64());
}

TYPED_TEST(IntegerValueTypedKind, ToUInt) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  // Small values fit in every host width, regardless of kind.
  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(uint8_t{0}, zero.ToUInt<uint8_t>());
  EXPECT_EQ(uint16_t{0}, zero.ToUInt<uint16_t>());
  EXPECT_EQ(uint32_t{0}, zero.ToUInt<uint32_t>());
  EXPECT_EQ(uint64_t{0}, zero.ToUInt<uint64_t>());
  EXPECT_EQ(uint128_t{0}, zero.ToUInt<uint128_t>());

  IntegerValue one{kind, 1};
  EXPECT_EQ(uint8_t{1}, one.ToUInt<uint8_t>());
  EXPECT_EQ(uint16_t{1}, one.ToUInt<uint16_t>());
  EXPECT_EQ(uint32_t{1}, one.ToUInt<uint32_t>());
  EXPECT_EQ(uint64_t{1}, one.ToUInt<uint64_t>());
  EXPECT_EQ(uint128_t{1}, one.ToUInt<uint128_t>());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(uint8_t{42}, theanswer.ToUInt<uint8_t>());
  EXPECT_EQ(uint16_t{42}, theanswer.ToUInt<uint16_t>());
  EXPECT_EQ(uint32_t{42}, theanswer.ToUInt<uint32_t>());
  EXPECT_EQ(uint64_t{42}, theanswer.ToUInt<uint64_t>());
  EXPECT_EQ(uint128_t{42}, theanswer.ToUInt<uint128_t>());

  // -1 is all-ones within the kind's own width. A host type at least as wide
  // as the kind therefore also reads back all-ones, but a host type wider
  // than a narrower kind sees that kind's value zero-extended instead.
  // A width as wide as the widest kind (16) always sees the exact value.
  IntegerValue negone{kind, -1};
  EXPECT_EQ(uint8_t{0xff}, negone.ToUInt<uint8_t>());
  EXPECT_EQ(UnsignedT(0xffffu), negone.ToUInt<uint16_t>());
  EXPECT_EQ(UnsignedT(0xffffffffu), negone.ToUInt<uint32_t>());
  EXPECT_EQ(uint128_t{std::numeric_limits<UnsignedT>::max()},
      negone.ToUInt<uint128_t>());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(UnsignedT(std::numeric_limits<SignedT>::max()),
      smaxv.ToUInt<UnsignedT>());
  EXPECT_EQ(uint128_t{std::numeric_limits<SignedT>::max()},
      smaxv.ToUInt<uint128_t>());

  // Least truncates to zero in any host width narrower than the kind.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(kind == 1 ? 0x80u : 0, sminv.ToUInt<uint8_t>());
  EXPECT_EQ(uint128_t{UnsignedT(std::numeric_limits<SignedT>::min())},
      sminv.ToUInt<uint128_t>());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_EQ(uint8_t{0xef}, patternv.ToUInt<uint8_t>());
  EXPECT_EQ(UnsignedT(0xcdefu), patternv.ToUInt<uint16_t>());
  EXPECT_EQ(UnsignedT(0x89abcdefu), patternv.ToUInt<uint32_t>());
  EXPECT_EQ(uint128_t{UnsignedT(0x0123456789abcdefull)},
      patternv.ToUInt<uint128_t>());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_EQ(uint8_t{0x10}, invpatternv.ToUInt<uint8_t>());
  EXPECT_EQ(UnsignedT(0x3210u), invpatternv.ToUInt<uint16_t>());
  EXPECT_EQ(invpatternv.ToUInt64(), invpatternv.ToUInt<uint64_t>()); // synonym
  // The inner UnsignedT cast undoes ~'s integer promotion for narrow kinds
  // before widening, so only the kind's own bits are zero-extended.
  EXPECT_EQ(uint128_t{UnsignedT(~UnsignedT(0x0123456789abcdefull))},
      invpatternv.ToUInt<uint128_t>());
}

TYPED_TEST(IntegerValueTypedKind, ToSInt) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  // Small values fit in every host width, regardless of kind.
  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(int8_t{0}, zero.ToSInt<int8_t>());
  EXPECT_EQ(int16_t{0}, zero.ToSInt<int16_t>());
  EXPECT_EQ(int32_t{0}, zero.ToSInt<int32_t>());
  EXPECT_EQ(int64_t{0}, zero.ToSInt<int64_t>());
  EXPECT_EQ(int128_t{0}, zero.ToSInt<int128_t>());

  IntegerValue one{kind, 1};
  EXPECT_EQ(int8_t{1}, one.ToSInt<int8_t>());
  EXPECT_EQ(int16_t{1}, one.ToSInt<int16_t>());
  EXPECT_EQ(int32_t{1}, one.ToSInt<int32_t>());
  EXPECT_EQ(int64_t{1}, one.ToSInt<int64_t>());
  EXPECT_EQ(int128_t{1}, one.ToSInt<int128_t>());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(int8_t{42}, theanswer.ToSInt<int8_t>());
  EXPECT_EQ(int16_t{42}, theanswer.ToSInt<int16_t>());
  EXPECT_EQ(int32_t{42}, theanswer.ToSInt<int32_t>());
  EXPECT_EQ(int64_t{42}, theanswer.ToSInt<int64_t>());
  EXPECT_EQ(int128_t{42}, theanswer.ToSInt<int128_t>());

  // -1 is all-ones at every width, in every kind.
  IntegerValue negone{kind, -1};
  EXPECT_EQ(int8_t{-1}, negone.ToSInt<int8_t>());
  EXPECT_EQ(int16_t(-1), negone.ToSInt<int16_t>());
  EXPECT_EQ(int32_t(-1), negone.ToSInt<int32_t>());
  EXPECT_EQ(int64_t(-1), negone.ToSInt<int64_t>());
  EXPECT_EQ(int128_t{-1}, negone.ToSInt<int128_t>());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(std::numeric_limits<SignedT>::max(), smaxv.ToSInt<SignedT>());
  EXPECT_EQ(
      int8_t(std::numeric_limits<SignedT>::max()), smaxv.ToSInt<int8_t>());
  EXPECT_EQ(
      int16_t(std::numeric_limits<SignedT>::max()), smaxv.ToSInt<int16_t>());
  EXPECT_EQ(
      int32_t(std::numeric_limits<SignedT>::max()), smaxv.ToSInt<int32_t>());
  EXPECT_EQ(
      int64_t(std::numeric_limits<SignedT>::max()), smaxv.ToSInt<int64_t>());
  EXPECT_EQ(
      int128_t{std::numeric_limits<SignedT>::max()}, smaxv.ToSInt<int128_t>());

  // Least's sign bit only survives in a host width no narrower than the
  // kind; a narrower width truncates it away, along with the sign. Widening
  // to the widest kind's own width (16) always sign-extends the true value.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(kind == 1 ? -128 : 0, sminv.ToSInt<int8_t>());
  EXPECT_EQ(
      int8_t(std::numeric_limits<SignedT>::min()), sminv.ToSInt<int8_t>());
  EXPECT_EQ(
      int16_t(std::numeric_limits<SignedT>::min()), sminv.ToSInt<int16_t>());
  EXPECT_EQ(
      int32_t(std::numeric_limits<SignedT>::min()), sminv.ToSInt<int32_t>());
  EXPECT_EQ(
      int64_t(std::numeric_limits<SignedT>::min()), sminv.ToSInt<int64_t>());
  EXPECT_EQ(
      int128_t{std::numeric_limits<SignedT>::min()}, sminv.ToSInt<int128_t>());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  SignedT patternvref{SignedT(UnsignedT(0x0123456789abcdefull))};
  EXPECT_EQ(int8_t(patternvref), patternv.ToSInt<int8_t>());
  EXPECT_EQ(int16_t(patternvref), patternv.ToSInt<int16_t>());
  EXPECT_EQ(int32_t(patternvref), patternv.ToSInt<int32_t>());
  EXPECT_EQ(int64_t(patternvref), patternv.ToSInt<int64_t>());
  EXPECT_EQ(int128_t{patternvref}, patternv.ToSInt<int128_t>());

  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  SignedT invpatternvref{SignedT(~UnsignedT(0x0123456789abcdefull))};
  EXPECT_EQ(int8_t(invpatternvref), invpatternv.ToSInt<int8_t>());
  EXPECT_EQ(int16_t(invpatternvref), invpatternv.ToSInt<int16_t>());
  EXPECT_EQ(int32_t(invpatternvref), invpatternv.ToSInt<int32_t>());
  EXPECT_EQ(int64_t(invpatternvref), invpatternv.ToSInt<int64_t>());
  EXPECT_EQ(int128_t{invpatternvref}, invpatternv.ToSInt<int128_t>());
}

//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, NOT) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(UnsignedT(~UnsignedT{0}), zero.NOT().ToUInt<UnsignedT>());

  IntegerValue one{kind, 1};
  EXPECT_EQ(UnsignedT(~UnsignedT{1}), one.NOT().ToUInt<UnsignedT>());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(UnsignedT{0}, negone.NOT().ToUInt<UnsignedT>());

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(UnsignedT(~UnsignedT{42}), theanswer.NOT().ToUInt<UnsignedT>());

  // Complementing HUGE (a leading zero followed by all ones) yields Least.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(sminv, smaxv.NOT());
  EXPECT_EQ(smaxv, sminv.NOT());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_EQ(invpatternv, patternv.NOT());
  EXPECT_EQ(patternv, invpatternv.NOT());
}

TYPED_TEST(IntegerValueTypedKind, IAND) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(theanswer, theanswer.IAND(theanswer));

  // A pattern and its complement share no set bits.
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_TRUE(patternv.IAND(invpatternv).IsZero());

  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_TRUE(zero.IAND(patternv).IsZero());

  // ANDing with all-ones is the identity.
  IntegerValue negone{kind, -1};
  EXPECT_EQ(patternv, negone.IAND(patternv));
}

TYPED_TEST(IntegerValueTypedKind, IOR) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(theanswer, theanswer.IOR(theanswer));

  IntegerValue negone{kind, -1};
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_EQ(negone, negone.IOR(patternv));

  // A pattern and its complement together cover every bit.
  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_EQ(negone, patternv.IOR(invpatternv));

  // ORing with zero is the identity; ORing with all-ones saturates.
  IntegerValue zero{IntegerValue::Zero(kind)};
  EXPECT_EQ(patternv, zero.IOR(patternv));
}

TYPED_TEST(IntegerValueTypedKind, IEOR) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};
  IntegerValue zero{IntegerValue::Zero(kind)};

  IntegerValue theanswer{kind, 42};
  EXPECT_TRUE(theanswer.IEOR(theanswer).IsZero());

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_EQ(patternv, zero.IEOR(patternv));

  IntegerValue negone{kind, -1};
  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  EXPECT_EQ(negone, patternv.IEOR(invpatternv));
  EXPECT_EQ(invpatternv, negone.IEOR(patternv));
}

TYPED_TEST(IntegerValueTypedKind, MERGE_BITS) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue negone{kind, -1};
  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};

  EXPECT_EQ(patternv, negone.MERGE_BITS(zero, patternv));
  EXPECT_EQ(invpatternv, zero.MERGE_BITS(negone, patternv));
  EXPECT_EQ(patternv, patternv.MERGE_BITS(invpatternv, negone));
  EXPECT_EQ(invpatternv, patternv.MERGE_BITS(invpatternv, zero));
}

TYPED_TEST(IntegerValueTypedKind, MAX) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue one{kind, 1};
  EXPECT_EQ(one, zero.MAX(one));

  IntegerValue negone{kind, -1};
  EXPECT_EQ(one, negone.MAX(one));

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(smaxv, smaxv.MAX(sminv));

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(theanswer, theanswer.MAX(theanswer));
}

TYPED_TEST(IntegerValueTypedKind, MIN) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue one{kind, 1};
  EXPECT_EQ(zero, zero.MIN(one));

  IntegerValue negone{kind, -1};
  EXPECT_EQ(negone, negone.MIN(one));

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(sminv, smaxv.MIN(sminv));

  IntegerValue theanswer{kind, 42};
  EXPECT_EQ(theanswer, theanswer.MIN(theanswer));
}

TYPED_TEST(IntegerValueTypedKind, IBCLR) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue theanswer{kind, 0b101010};
  EXPECT_EQ(40, theanswer.IBCLR(1).ToInt64()); // clears the bit worth 2
  EXPECT_EQ(34, theanswer.IBCLR(3).ToInt64()); // clears the bit worth 8
  EXPECT_EQ(42, theanswer.IBCLR(0).ToInt64()); // bit 0 is already clear
  // Out-of-range positions are ignored.
  EXPECT_EQ(theanswer, theanswer.IBCLR(-1));
  EXPECT_EQ(theanswer, theanswer.IBCLR(bits));

  IntegerValue negone{kind, -1};
  EXPECT_EQ(UnsignedT(~UnsignedT{1}), negone.IBCLR(0).ToUInt<UnsignedT>());
  // Clearing the sign bit of all-ones yields HUGE.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(smaxv, negone.IBCLR(bits - 1));
}

TYPED_TEST(IntegerValueTypedKind, IBSET) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue theanswer{kind, 42}; // 0b101010
  EXPECT_EQ(43, theanswer.IBSET(0).ToInt64()); // sets the bit worth 1
  EXPECT_EQ(46, theanswer.IBSET(2).ToInt64()); // sets the bit worth 4
  EXPECT_EQ(42, theanswer.IBSET(1).ToInt64()); // bit 1 is already set
  EXPECT_EQ(theanswer, theanswer.IBSET(-1));
  EXPECT_EQ(theanswer, theanswer.IBSET(bits));

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue one{kind, 1};
  EXPECT_EQ(one, zero.IBSET(0));
  // Setting the sign bit of zero yields Least.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(sminv, zero.IBSET(bits - 1));
}

TYPED_TEST(IntegerValueTypedKind, IBITS) {
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  // 0x...ef: low byte is 0b11101111.
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_EQ(0xf, patternv.IBITS(0, 4).ToInt64());
  EXPECT_EQ(0xe, patternv.IBITS(4, 4).ToInt64());
  // Bit fields are unsigned; for kind 1 this extracts the whole byte, whose
  // top bit would read as negative through the signed accessor.
  EXPECT_EQ(0xefull, patternv.IBITS(0, 8).ToUInt64());
  EXPECT_TRUE(patternv.IBITS(0, 0).IsZero());
  // A zero-based field spanning the full width extracts the whole value.
  EXPECT_EQ(patternv, patternv.IBITS(0, bits));
}

//===----------------------------------------------------------------------===//
// Shifts
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, ISHFT) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  // A positive count shifts left; a negative count shifts right.
  IntegerValue one{kind, 1};
  EXPECT_EQ(1, one.ISHFT(0).ToInt64());
  EXPECT_EQ(2, one.ISHFT(1).ToInt64());
  EXPECT_EQ(0, one.ISHFT(-1).ToInt64());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(sminv, one.ISHFT(bits - 1));
  EXPECT_TRUE(one.ISHFT(bits).IsZero());
  EXPECT_TRUE(one.ISHFT(bits + 1).IsZero());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(UnsignedT(~UnsignedT{1}), negone.ISHFT(1).ToUInt<UnsignedT>());

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(smaxv, negone.ISHFT(-1));
  EXPECT_TRUE(negone.ISHFT(bits).IsZero());
  EXPECT_TRUE(negone.ISHFT(-bits).IsZero());
}

TYPED_TEST(IntegerValueTypedKind, SHIFTL) {
  using UnsignedT = typename TypeParam::UnsignedT;
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue one{kind, 1};
  EXPECT_EQ(1, one.SHIFTL(-1).ToInt64()); // nonpositive count: no shift
  EXPECT_EQ(1, one.SHIFTL(0).ToInt64());
  EXPECT_EQ(2, one.SHIFTL(1).ToInt64());
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(sminv, one.SHIFTL(bits - 1));
  EXPECT_TRUE(one.SHIFTL(bits).IsZero());
  EXPECT_TRUE(one.SHIFTL(bits + 1).IsZero());

  IntegerValue negone{kind, -1};
  EXPECT_EQ(negone, negone.SHIFTL(0));
  EXPECT_EQ(UnsignedT(~UnsignedT{1}), negone.SHIFTL(1).ToUInt<UnsignedT>());
}

TYPED_TEST(IntegerValueTypedKind, SHIFTR) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue negone{kind, -1};
  EXPECT_EQ(negone, negone.SHIFTR(-1)); // nonpositive count: no shift
  EXPECT_EQ(negone, negone.SHIFTR(0));

  // Zero fill, so a negative value becomes positive.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_EQ(smaxv, negone.SHIFTR(1));
  EXPECT_TRUE(negone.SHIFTR(bits).IsZero());
  EXPECT_TRUE(negone.SHIFTR(bits + 1).IsZero());

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(1, sminv.SHIFTR(bits - 1).ToInt64());
}

TYPED_TEST(IntegerValueTypedKind, SHIFTA) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue negone{kind, -1};
  EXPECT_EQ(negone, negone.SHIFTA(-1)); // nonpositive count: no shift
  EXPECT_EQ(negone, negone.SHIFTA(0));
  // Sign fill keeps a negative value negative.
  EXPECT_EQ(negone, negone.SHIFTA(1));
  EXPECT_EQ(negone, negone.SHIFTA(bits - 1));

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(negone, sminv.SHIFTA(bits - 1));

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  EXPECT_TRUE(smaxv.SHIFTA(bits - 1).IsZero());
}

TYPED_TEST(IntegerValueTypedKind, ISHFTC) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  // Rotating a uniform bit pattern leaves it unchanged.
  IntegerValue negone{kind, -1};
  EXPECT_EQ(negone, negone.ISHFTC(1));
  EXPECT_EQ(negone, negone.ISHFTC(-1));

  // Rotating the single set bit off one end wraps it to the other.
  IntegerValue one{kind, 1};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  EXPECT_EQ(2, one.ISHFTC(1).ToInt64());
  EXPECT_EQ(sminv, one.ISHFTC(-1));
  EXPECT_EQ(one, sminv.ISHFTC(1));

  // A full-word rotation by the width is the identity.
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  EXPECT_EQ(patternv, patternv.ISHFTC(bits));

  // Rotating within a narrower field of least-significant bits leaves the
  // higher-order bits unchanged; a nonpositive size selects the full width.
  EXPECT_EQ(2, one.ISHFTC(1, 4).ToInt64());
  EXPECT_EQ(8, one.ISHFTC(-1, 4).ToInt64());
  EXPECT_EQ(2, one.ISHFTC(1, 0).ToInt64());
}

TYPED_TEST(IntegerValueTypedKind, DSHIFTL) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};
  const UnsignedT i{UnsignedT(0x0123456789abcdefull)};
  const UnsignedT j{UnsignedT(~i)};
  IntegerValue a{kind, i}, b{kind, j};

  // The leading `bits` of the doubled-width value i:j shifted left by count.
  EXPECT_EQ(a, a.DSHIFTL(b, 0)); // count==0 selects i unchanged
  EXPECT_EQ(b, a.DSHIFTL(b, bits)); // count==bits selects j unchanged
  EXPECT_TRUE(a.DSHIFTL(b, 2 * bits).IsZero()); // shifted entirely out

  constexpr int half{bits / 2};
  const UnsignedT expected{
      UnsignedT(UnsignedT(i << half) | UnsignedT(j >> (bits - half)))};
  EXPECT_EQ(expected, a.DSHIFTL(b, half).ToUInt<UnsignedT>());
}

TYPED_TEST(IntegerValueTypedKind, DSHIFTR) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};
  const UnsignedT i{UnsignedT(0x0123456789abcdefull)};
  const UnsignedT j{UnsignedT(~i)};
  IntegerValue a{kind, i}, b{kind, j};

  // The trailing `bits` of the doubled-width value i:j shifted right by
  // count.
  EXPECT_EQ(b, a.DSHIFTR(b, 0)); // count==0 selects j unchanged
  EXPECT_EQ(a, a.DSHIFTR(b, bits)); // count==bits selects i unchanged
  EXPECT_TRUE(a.DSHIFTR(b, 2 * bits).IsZero()); // shifted entirely out

  constexpr int half{bits / 2};
  const UnsignedT expected(
      UnsignedT(j >> half) | UnsignedT(i << (bits - half)));
  EXPECT_EQ(expected, a.DSHIFTR(b, half).ToUInt<UnsignedT>());
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, Negate) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  auto negZero{zero.Negate()};
  EXPECT_TRUE(negZero.value.IsZero());
  EXPECT_FALSE(negZero.overflow);

  IntegerValue one{kind, 1};
  IntegerValue negone{kind, -1};
  auto negOne{one.Negate()};
  EXPECT_EQ(negone, negOne.value);
  EXPECT_FALSE(negOne.overflow);
  auto negMOne{negone.Negate()};
  EXPECT_EQ(one, negMOne.value);
  EXPECT_FALSE(negMOne.overflow);

  IntegerValue theanswer{kind, 42};
  auto negAnswer{theanswer.Negate()};
  EXPECT_EQ(-42, negAnswer.value.ToInt64());
  EXPECT_FALSE(negAnswer.overflow);

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  auto negHuge{smaxv.Negate()};
  EXPECT_EQ(-smaxv.ToInt64(), negHuge.value.ToInt64());
  EXPECT_FALSE(negHuge.overflow);

  // Only the most negative number cannot be negated; it wraps back to
  // itself.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  auto negLeast{sminv.Negate()};
  EXPECT_EQ(sminv, negLeast.value);
  EXPECT_TRUE(negLeast.overflow);
}

TYPED_TEST(IntegerValueTypedKind, ABS) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  auto absZero{zero.ABS()};
  EXPECT_TRUE(absZero.value.IsZero());
  EXPECT_FALSE(absZero.overflow);

  IntegerValue one{kind, 1};
  auto absOne{one.ABS()};
  EXPECT_EQ(one, absOne.value);
  EXPECT_FALSE(absOne.overflow);

  IntegerValue negone{kind, -1};
  auto absMOne{negone.ABS()};
  EXPECT_EQ(one, absMOne.value);
  EXPECT_FALSE(absMOne.overflow);

  IntegerValue theanswer{kind, 42};
  auto absAnswer{theanswer.ABS()};
  EXPECT_EQ(42, absAnswer.value.ToInt64());
  EXPECT_FALSE(absAnswer.overflow);

  // HUGE is already nonnegative.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  auto absHuge{smaxv.ABS()};
  EXPECT_EQ(smaxv, absHuge.value);
  EXPECT_FALSE(absHuge.overflow);

  // Taking the magnitude of the most negative number overflows; it stays
  // unchanged.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  auto absLeast{sminv.ABS()};
  EXPECT_EQ(sminv, absLeast.value);
  EXPECT_TRUE(absLeast.overflow);
}

TYPED_TEST(IntegerValueTypedKind, AddUnsigned) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  auto zeroPlusZero{zero.AddUnsigned(zero)};
  EXPECT_TRUE(zeroPlusZero.value.IsZero());
  EXPECT_FALSE(zeroPlusZero.carry);

  // All-ones plus one wraps around to zero with a carry out.
  IntegerValue negone{kind, -1};
  IntegerValue one{kind, 1};
  auto wrapped{negone.AddUnsigned(one)};
  EXPECT_TRUE(wrapped.value.IsZero());
  EXPECT_TRUE(wrapped.carry);
  // A carry in has the same effect as adding one.
  auto wrappedByCarryIn{negone.AddUnsigned(zero, /*carryIn=*/true)};
  EXPECT_TRUE(wrappedByCarryIn.value.IsZero());
  EXPECT_TRUE(wrappedByCarryIn.carry);

  IntegerValue theanswer{kind, 42};
  auto doubled{theanswer.AddUnsigned(theanswer)};
  EXPECT_EQ(84, doubled.value.ToInt64());
  EXPECT_FALSE(doubled.carry);

  // A pattern and its complement sum exactly to all-ones.
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  IntegerValue invpatternv{kind, ~UnsignedT(0x0123456789abcdefull)};
  auto complementary{patternv.AddUnsigned(invpatternv)};
  EXPECT_EQ(negone, complementary.value);
  EXPECT_FALSE(complementary.carry);
  auto complementaryPlusOne{
      patternv.AddUnsigned(invpatternv, /*carryIn=*/true)};
  EXPECT_TRUE(complementaryPlusOne.value.IsZero());
  EXPECT_TRUE(complementaryPlusOne.carry);
}

TYPED_TEST(IntegerValueTypedKind, AddSigned) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  auto zeroPlusZero{zero.AddSigned(zero)};
  EXPECT_TRUE(zeroPlusZero.value.IsZero());
  EXPECT_FALSE(zeroPlusZero.overflow);

  // Operands of unlike sign can never overflow.
  IntegerValue one{kind, 1};
  IntegerValue negone{kind, -1};
  auto onePlusMOne{one.AddSigned(negone)};
  EXPECT_TRUE(onePlusMOne.value.IsZero());
  EXPECT_FALSE(onePlusMOne.overflow);

  IntegerValue theanswer{kind, 42};
  auto doubled{theanswer.AddSigned(theanswer)};
  EXPECT_EQ(84, doubled.value.ToInt64());
  EXPECT_FALSE(doubled.overflow);

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  // HUGE+1 overflows and wraps around to the most negative number.
  auto hugePlusOne{smaxv.AddSigned(one)};
  EXPECT_EQ(sminv, hugePlusOne.value);
  EXPECT_TRUE(hugePlusOne.overflow);

  // Least-1 underflows and wraps around to HUGE.
  auto leastMinusOne{sminv.AddSigned(negone)};
  EXPECT_EQ(smaxv, leastMinusOne.value);
  EXPECT_TRUE(leastMinusOne.overflow);
}

TYPED_TEST(IntegerValueTypedKind, SubtractSigned) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  auto zeroMinusZero{zero.SubtractSigned(zero)};
  EXPECT_TRUE(zeroMinusZero.value.IsZero());
  EXPECT_FALSE(zeroMinusZero.overflow);

  IntegerValue theanswer{kind, 42};
  auto selfMinusSelf{theanswer.SubtractSigned(theanswer)};
  EXPECT_TRUE(selfMinusSelf.value.IsZero());
  EXPECT_FALSE(selfMinusSelf.overflow);

  IntegerValue one{kind, 1};
  IntegerValue negone{kind, -1};
  auto oneMinusMOne{one.SubtractSigned(negone)};
  EXPECT_EQ(2, oneMinusMOne.value.ToInt64());
  EXPECT_FALSE(oneMinusMOne.overflow);

  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  // HUGE-(-1) overflows and wraps around to the most negative number.
  auto hugeMinusMOne{smaxv.SubtractSigned(negone)};
  EXPECT_EQ(sminv, hugeMinusMOne.value);
  EXPECT_TRUE(hugeMinusMOne.overflow);

  // Least-1 underflows and wraps around to HUGE.
  auto leastMinusOne{sminv.SubtractSigned(one)};
  EXPECT_EQ(smaxv, leastMinusOne.value);
  EXPECT_TRUE(leastMinusOne.overflow);
}

TYPED_TEST(IntegerValueTypedKind, DIM) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue one{kind, 1};
  IntegerValue theanswer{kind, 42};
  // x <= y clamps at zero rather than going negative.
  auto smallMinusBig{one.DIM(theanswer)};
  EXPECT_TRUE(smallMinusBig.value.IsZero());
  EXPECT_FALSE(smallMinusBig.overflow);
  EXPECT_EQ(kind, smallMinusBig.value.kind());

  auto selfMinusSelf{theanswer.DIM(theanswer)};
  EXPECT_TRUE(selfMinusSelf.value.IsZero());
  EXPECT_FALSE(selfMinusSelf.overflow);

  auto bigMinusSmall{theanswer.DIM(one)};
  EXPECT_EQ(41, bigMinusSmall.value.ToInt64());
  EXPECT_FALSE(bigMinusSmall.overflow);

  // HUGE-Least overflows the representable range.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  IntegerValue negone{kind, -1};
  auto hugeMinusLeast{smaxv.DIM(sminv)};
  EXPECT_EQ(negone, hugeMinusLeast.value);
  EXPECT_TRUE(hugeMinusLeast.overflow);

  auto leastMinusHuge{sminv.DIM(smaxv)};
  EXPECT_TRUE(leastMinusHuge.value.IsZero());
  EXPECT_FALSE(leastMinusHuge.overflow);
}

TYPED_TEST(IntegerValueTypedKind, SIGN) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue one{kind, 1};
  IntegerValue negone{kind, -1};
  IntegerValue theanswer{kind, 42};

  // Same sign as the second operand: the value is unchanged.
  auto samePos{one.SIGN(theanswer)};
  EXPECT_EQ(one, samePos.value);
  EXPECT_FALSE(samePos.overflow);
  auto sameNeg{negone.SIGN(negone)};
  EXPECT_EQ(negone, sameNeg.value);
  EXPECT_FALSE(sameNeg.overflow);

  // Differing sign: the value is negated.
  auto flipToNeg{one.SIGN(negone)};
  EXPECT_EQ(negone, flipToNeg.value);
  EXPECT_FALSE(flipToNeg.overflow);
  auto flipToPos{negone.SIGN(one)};
  EXPECT_EQ(one, flipToPos.value);
  EXPECT_FALSE(flipToPos.overflow);

  // Negating the most negative number overflows and wraps back to itself.
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  auto flipLeast{sminv.SIGN(smaxv)};
  EXPECT_EQ(sminv, flipLeast.value);
  EXPECT_TRUE(flipLeast.overflow);
  auto sameLeast{sminv.SIGN(sminv)};
  EXPECT_EQ(sminv, sameLeast.value);
  EXPECT_FALSE(sameLeast.overflow);
}

TYPED_TEST(IntegerValueTypedKind, MultiplyUnsigned) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};
  constexpr int bits{TypeParam::bits};

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue patternv{kind, 0x0123456789abcdefull};
  auto zeroProduct{zero.MultiplyUnsigned(patternv)};
  EXPECT_TRUE(zeroProduct.lower.IsZero());
  EXPECT_TRUE(zeroProduct.upper.IsZero());
  EXPECT_FALSE(zeroProduct.overflow);

  IntegerValue one{kind, 1};
  auto identityProduct{one.MultiplyUnsigned(patternv)};
  EXPECT_EQ(patternv, identityProduct.lower);
  EXPECT_TRUE(identityProduct.upper.IsZero());
  EXPECT_FALSE(identityProduct.overflow);

  IntegerValue theanswer{kind, 42};
  auto answerSquared{theanswer.MultiplyUnsigned(theanswer)};
  EXPECT_EQ(UnsignedT(42 * 42), answerSquared.lower.ToUInt<UnsignedT>());
  EXPECT_FALSE(answerSquared.overflow);

  // All-ones squared: (2^bits-1)^2 == 1 (mod 2^bits), with the high half
  // holding the rest of the product.
  IntegerValue negone{kind, -1};
  auto moneSquared{negone.MultiplyUnsigned(negone)};
  EXPECT_EQ(1, moneSquared.lower.ToInt64());
  EXPECT_FALSE(moneSquared.overflow);
  // Only up to INTEGER(8) is there a host type wide enough to check the
  // full product directly.
  if constexpr (bits <= 64) {
    using Wide = HostUnsignedIntType<2 * bits>;
    const Wide allOnes{UnsignedT(~UnsignedT{0})};
    const Wide wide{Wide(allOnes * allOnes)};
    EXPECT_EQ(UnsignedT(wide >> bits), moneSquared.upper.ToUInt<UnsignedT>());
  }
}

TYPED_TEST(IntegerValueTypedKind, MultiplySigned) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue theanswer{kind, 42};
  auto zeroProduct{zero.MultiplySigned(theanswer)};
  EXPECT_TRUE(zeroProduct.lower.IsZero());
  EXPECT_TRUE(zeroProduct.upper.IsZero());
  EXPECT_FALSE(zeroProduct.overflow);

  IntegerValue one{kind, 1};
  auto identityProduct{one.MultiplySigned(theanswer)};
  EXPECT_EQ(theanswer, identityProduct.lower);
  EXPECT_TRUE(identityProduct.upper.IsZero()); // theanswer is positive
  EXPECT_FALSE(identityProduct.overflow);

  IntegerValue negone{kind, -1};
  auto negated{negone.MultiplySigned(one)};
  EXPECT_EQ(negone, negated.lower);
  EXPECT_EQ(negone, negated.upper); // sign-extended
  EXPECT_FALSE(negated.overflow);

  // Least*-1 overflows: the true product is one past the representable
  // range, and wraps back to the bit pattern of Least itself.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  auto negatedLeast{sminv.MultiplySigned(negone)};
  EXPECT_EQ(sminv, negatedLeast.lower);
  EXPECT_TRUE(negatedLeast.upper.IsZero());
  EXPECT_TRUE(negatedLeast.overflow);
  EXPECT_TRUE(negatedLeast.SignedMultiplicationOverflowed());
}

TYPED_TEST(IntegerValueTypedKind, DivideUnsigned) {
  using UnsignedT = typename TypeParam::UnsignedT;
  constexpr int kind{TypeParam::kind};

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue negone{kind, -1};
  auto byZero{patternv.DivideUnsigned(zero)};
  EXPECT_TRUE(byZero.divisionByZero);
  EXPECT_EQ(negone, byZero.quotient);
  EXPECT_TRUE(byZero.remainder.IsZero());

  IntegerValue theanswer{kind, 42};
  IntegerValue one{kind, 1};
  auto byOne{theanswer.DivideUnsigned(one)};
  EXPECT_FALSE(byOne.divisionByZero);
  EXPECT_EQ(theanswer, byOne.quotient);
  EXPECT_TRUE(byOne.remainder.IsZero());

  auto bySelf{theanswer.DivideUnsigned(theanswer)};
  EXPECT_FALSE(bySelf.divisionByZero);
  EXPECT_EQ(one, bySelf.quotient);
  EXPECT_TRUE(bySelf.remainder.IsZero());

  // All-ones is the largest unsigned value.
  const UnsignedT allOnes{UnsignedT(~UnsignedT{0})};
  auto moneByAnswer{negone.DivideUnsigned(theanswer)};
  EXPECT_FALSE(moneByAnswer.divisionByZero);
  EXPECT_EQ(UnsignedT(allOnes / UnsignedT{42}),
      moneByAnswer.quotient.ToUInt<UnsignedT>());
  EXPECT_EQ(UnsignedT(allOnes % UnsignedT{42}),
      moneByAnswer.remainder.ToUInt<UnsignedT>());
}

TYPED_TEST(IntegerValueTypedKind, DivideSigned) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};

  // A nonzero remainder has the sign of the dividend: this is MOD, not MODULO.
  struct {
    int64_t x, y, quotient, remainder;
  } cases[]{
      {8, 5, 1, 3},
      {-8, 5, -1, -3},
      {8, -5, -1, 3},
      {-8, -5, 1, -3},
  };
  for (auto &c : cases) {
    auto r{IntegerValue(kind, c.x).DivideSigned(IntegerValue{kind, c.y})};
    EXPECT_FALSE(r.divisionByZero);
    EXPECT_FALSE(r.overflow);
    EXPECT_EQ(c.quotient, r.quotient.ToInt64()) << c.x << '/' << c.y;
    EXPECT_EQ(c.remainder, r.remainder.ToInt64()) << c.x << '/' << c.y;
  }

  IntegerValue zero{IntegerValue::Zero(kind)};
  IntegerValue theanswer{kind, 42};
  IntegerValue negone{kind, -1};
  IntegerValue smaxv{kind, std::numeric_limits<SignedT>::max()};
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};

  // Division by zero saturates in the direction of the dividend's sign.
  auto positiveByZero{theanswer.DivideSigned(zero)};
  EXPECT_TRUE(positiveByZero.divisionByZero);
  EXPECT_EQ(smaxv, positiveByZero.quotient);
  EXPECT_TRUE(positiveByZero.remainder.IsZero());

  auto negativeByZero{negone.DivideSigned(zero)};
  EXPECT_TRUE(negativeByZero.divisionByZero);
  EXPECT_EQ(sminv, negativeByZero.quotient);
  EXPECT_TRUE(negativeByZero.remainder.IsZero());

  // The most negative number divided by -1 is the sole overflow case.
  auto leastByMOne{sminv.DivideSigned(negone)};
  EXPECT_FALSE(leastByMOne.divisionByZero);
  EXPECT_TRUE(leastByMOne.overflow);
  EXPECT_EQ(sminv, leastByMOne.quotient);
  EXPECT_TRUE(leastByMOne.remainder.IsZero());
}

TYPED_TEST(IntegerValueTypedKind, MODULO) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  // The result has the sign of the divisor.
  struct {
    int64_t x, y, modulo;
  } cases[]{
      {8, 5, 3},
      {-8, 5, 2},
      {8, -5, -2},
      {-8, -5, -3},
  };
  for (auto &c : cases) {
    auto r{IntegerValue(kind, c.x).MODULO(IntegerValue{kind, c.y})};
    EXPECT_FALSE(r.overflow);
    EXPECT_EQ(c.modulo, r.value.ToInt64()) << c.x << " mod " << c.y;
  }

  IntegerValue one{kind, 1};
  IntegerValue theanswer{kind, 42};
  auto exact{theanswer.MODULO(one)};
  EXPECT_FALSE(exact.overflow);
  EXPECT_TRUE(exact.value.IsZero());

  // -1 mod 42: the result takes the sign of the (positive) divisor.
  IntegerValue negone{kind, -1};
  auto negByPos{negone.MODULO(theanswer)};
  EXPECT_FALSE(negByPos.overflow);
  EXPECT_EQ(41, negByPos.value.ToInt64());

  // Least mod -1 is exactly zero, but MODULO still reports the overflow
  // that occurs while computing the underlying Least/-1 quotient.
  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  auto leastByMOne{sminv.MODULO(negone)};
  EXPECT_TRUE(leastByMOne.overflow);
  EXPECT_TRUE(leastByMOne.value.IsZero());
}

TYPED_TEST(IntegerValueTypedKind, Power) {
  constexpr int kind{TypeParam::kind};
  IntegerValue three{kind, 3};
  IntegerValue two{kind, 2};
  auto square{three.Power(two)};
  EXPECT_FALSE(square.overflow);
  EXPECT_FALSE(square.divisionByZero);
  EXPECT_FALSE(square.zeroToZero);
  EXPECT_EQ(9, square.power.ToInt64());

  // x**0 is 1; 0**0 is 1 too, but additionally reports zeroToZero.
  IntegerValue seven{kind, 7};
  IntegerValue zero{IntegerValue::Zero(kind)};
  auto zeroth{seven.Power(zero)};
  EXPECT_EQ(1, zeroth.power.ToInt64());
  EXPECT_FALSE(zeroth.zeroToZero);
  auto zeroToZero{zero.Power(zero)};
  EXPECT_EQ(1, zeroToZero.power.ToInt64());
  EXPECT_TRUE(zeroToZero.zeroToZero);

  // 0**-1 divides by zero.
  IntegerValue minusOne{kind, -1};
  auto zeroToMinusOne{zero.Power(minusOne)};
  EXPECT_TRUE(zeroToMinusOne.divisionByZero);

  // Negative exponents truncate towards zero for other bases.
  auto twoToMinusOne{two.Power(minusOne)};
  EXPECT_TRUE(twoToMinusOne.power.IsZero());
  IntegerValue one{kind, 1};
  IntegerValue minusThree{kind, -3};
  auto oneToMinusThree{one.Power(minusThree)};
  EXPECT_EQ(1, oneToMinusThree.power.ToInt64());
  auto minusOneToMinusThree{minusOne.Power(minusThree)};
  EXPECT_EQ(-1, minusOneToMinusThree.power.ToInt64());
  IntegerValue minusTwo{kind, -2};
  auto minusOneToMinusTwo{minusOne.Power(minusTwo)};
  EXPECT_EQ(1, minusOneToMinusTwo.power.ToInt64());

  IntegerValue huge{IntegerValue::HUGE(kind)};
  auto hugeSquared{huge.Power(two)};
  EXPECT_TRUE(hugeSquared.overflow);
}

//===----------------------------------------------------------------------===//
// Raw storage
//===----------------------------------------------------------------------===//

TYPED_TEST(IntegerValueTypedKind, RawBytesRoundTrip) {
  using SignedT = typename TypeParam::SignedT;
  constexpr int kind{TypeParam::kind};
  ASSERT_EQ(
      IntegerValue::bytesStored(kind), IntegerValue::Zero(kind).bytesStored());

  IntegerValue zero{IntegerValue::Zero(kind)};
  char zeroBuffer[16]{};
  bool zeroChanged{false};
  zero.StoreRawBytes(zeroBuffer, zero.bytesStored(), &zeroChanged);
  EXPECT_FALSE(zeroChanged); // storing a zero never sets the "changed" flag
  IntegerValue zeroRestored{
      IntegerValue::FromRawBytes(kind, zeroBuffer, zero.bytesStored())};
  EXPECT_EQ(kind, zeroRestored.kind());
  EXPECT_EQ(zero, zeroRestored);

  IntegerValue negone{kind, -1};
  char moneBuffer[16]{};
  bool moneChanged{false};
  negone.StoreRawBytes(moneBuffer, negone.bytesStored(), &moneChanged);
  EXPECT_TRUE(moneChanged);
  IntegerValue moneRestored{
      IntegerValue::FromRawBytes(kind, moneBuffer, negone.bytesStored())};
  EXPECT_EQ(kind, moneRestored.kind());
  EXPECT_EQ(negone, moneRestored);

  // Storing the same value again reports no change.
  moneChanged = false;
  negone.StoreRawBytes(moneBuffer, negone.bytesStored(), &moneChanged);
  EXPECT_FALSE(moneChanged);
  // Overwriting with a different value reports a change.
  IntegerValue theanswer{kind, 42};
  bool overwriteChanged{false};
  theanswer.StoreRawBytes(
      moneBuffer, theanswer.bytesStored(), &overwriteChanged);
  EXPECT_TRUE(overwriteChanged);
  IntegerValue answerRestored{
      IntegerValue::FromRawBytes(kind, moneBuffer, theanswer.bytesStored())};
  EXPECT_EQ(theanswer, answerRestored);

  IntegerValue patternv{kind, 0x0123456789abcdefull};
  char patternBuffer[16]{};
  bool patternChanged{false};
  patternv.StoreRawBytes(
      patternBuffer, patternv.bytesStored(), &patternChanged);
  EXPECT_TRUE(patternChanged);
  IntegerValue patternRestored{
      IntegerValue::FromRawBytes(kind, patternBuffer, patternv.bytesStored())};
  EXPECT_EQ(kind, patternRestored.kind());
  EXPECT_EQ(patternv, patternRestored);

  IntegerValue sminv{kind, std::numeric_limits<SignedT>::min()};
  char sminvBuffer[16]{};
  bool sminvChanged{false};
  sminv.StoreRawBytes(sminvBuffer, sminv.bytesStored(), &sminvChanged);
  EXPECT_TRUE(sminvChanged);
  IntegerValue sminvRestored{
      IntegerValue::FromRawBytes(kind, sminvBuffer, sminv.bytesStored())};
  EXPECT_EQ(kind, sminvRestored.kind());
  EXPECT_EQ(sminv, sminvRestored);
}

//===----------------------------------------------------------------------===//
// Operations between operands of different kinds
//
// A dyadic operation converts its argument to the receiver's kind, so these
// are parameterized over ordered pairs of kinds rather than over single kinds.
//===----------------------------------------------------------------------===//

class IntegerValueKindPair
    : public testing::TestWithParam<std::tuple<int, int>> {};

INSTANTIATE_TEST_SUITE_P(AllKindPairs, IntegerValueKindPair,
    testing::Combine(
        testing::ValuesIn(std::initializer_list<int> FORTRAN_INTEGER_KINDS),
        testing::ValuesIn(std::initializer_list<int> FORTRAN_INTEGER_KINDS)),
    [](const testing::TestParamInfo<std::tuple<int, int>> &info) {
      return "KIND" + std::to_string(std::get<0>(info.param)) + "AndKIND" +
          std::to_string(std::get<1>(info.param));
    });

TEST_P(IntegerValueKindPair, ConvertUnsigned) {
  const int from{std::get<0>(GetParam())}, to{std::get<1>(GetParam())};
  const int fromBits{IntegerValue::bits(from)}, toBits{IntegerValue::bits(to)};
  const int common{std::min(fromBits, toBits)};

  // All ones: zero-extended when widening, truncated (and flagged) otherwise.
  auto ones{IntegerValue::ConvertUnsigned(to, IntegerValue{from, -1})};
  EXPECT_EQ(to, ones.value.kind());
  EXPECT_EQ(toBits < fromBits, ones.overflow);
  EXPECT_EQ(IntegerValue::MASKR(to, common), ones.value);

  // A value that fits in either width converts exactly.
  auto exact{IntegerValue::ConvertUnsigned(to, IntegerValue{from, 0x34})};
  EXPECT_FALSE(exact.overflow);
  EXPECT_EQ(IntegerValue(to, 0x34), exact.value);

  auto zero{IntegerValue::ConvertUnsigned(to, IntegerValue::Zero(from))};
  EXPECT_FALSE(zero.overflow);
  EXPECT_TRUE(zero.value.IsZero());
}

TEST_P(IntegerValueKindPair, ConvertSigned) {
  const int from{std::get<0>(GetParam())}, to{std::get<1>(GetParam())};
  const int fromBits{IntegerValue::bits(from)}, toBits{IntegerValue::bits(to)};

  // All ones stays all ones: it sign-extends and truncates to itself.
  auto ones{IntegerValue::ConvertSigned(to, IntegerValue{from, -1})};
  EXPECT_EQ(to, ones.value.kind());
  EXPECT_FALSE(ones.overflow);
  EXPECT_EQ(IntegerValue(to, -1), ones.value);

  // Truncation that changes the value is flagged.
  auto huge{IntegerValue::ConvertSigned(to, IntegerValue::HUGE(from))};
  EXPECT_EQ(toBits < fromBits, huge.overflow);
  EXPECT_EQ(toBits < fromBits ? IntegerValue(to, -1)
                              : IntegerValue::MASKR(to, fromBits - 1),
      huge.value);
  auto exact{IntegerValue::ConvertSigned(to, IntegerValue{from, -56})};
  EXPECT_FALSE(exact.overflow);
  EXPECT_EQ(IntegerValue(to, -56), exact.value);
}

TEST_P(IntegerValueKindPair, MixedKindOperandsAreCoerced) {
  const int receiver{std::get<0>(GetParam())};
  const int other{std::get<1>(GetParam())};
  IntegerValue x{receiver, 0x5a};
  IntegerValue allOnes{other, -1};

  // The result takes the receiver's kind; the argument is converted to it,
  // preserving its sign.
  EXPECT_EQ(receiver, x.IOR(allOnes).kind());
  EXPECT_EQ(IntegerValue(receiver, -1), x.IOR(allOnes));
  EXPECT_EQ(x, x.IAND(allOnes));
  EXPECT_EQ(Ordering::Greater, x.CompareSigned(allOnes));
  EXPECT_EQ(IntegerValue(receiver, 0x5a - 1), x.AddSigned(allOnes).value);

  // A monostate operand behaves as a zero of the receiver's width.
  EXPECT_EQ(x, x.IOR(IntegerValue{}));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
TEST(IntegerValue, Dump) { IntegerValue(4, -1).dump(); }
#endif

// Not an IntegerValue method, but the ordering that its comparisons return;
// checked here because the legacy non-GTest test does so too.
TEST(Ordering, Reverse) {
  EXPECT_EQ(Ordering::Greater, Reverse(Ordering::Less));
  EXPECT_EQ(Ordering::Less, Reverse(Ordering::Greater));
  EXPECT_EQ(Ordering::Equal, Reverse(Ordering::Equal));
}

//===----------------------------------------------------------------------===//
// Exhaustive tests
//
// The tests above check a selection of bit patterns for every kind; these
// replicate the coverage of the legacy non-GTest test
// flang/unittests/Evaluate/integer.cpp by checking every value (and, for the
// dyadic operations, every pair of values) of a narrow kind.
//===----------------------------------------------------------------------===//

void ExhaustiveUnary(int kind) {
  const int bits{IntegerValue::bits(kind)};
  ASSERT_LE(bits, 16); // the reference arithmetic below assumes a narrow kind
  const uint64_t maxUnsigned{(uint64_t{1} << bits) - 1};

  IntegerValue zero{IntegerValue::Zero(kind)};
  ASSERT_TRUE(zero.IsZero());
  ASSERT_EQ(0u, zero.ToUInt64());
  ASSERT_EQ(0, zero.ToInt64());
  ASSERT_EQ(int64_t(maxUnsigned >> 1), IntegerValue::HUGE(kind).ToInt64());

  for (uint64_t x{0}; x <= maxUnsigned; ++x) {
    SCOPED_TRACE(testing::Message() << "kind=" << kind << " x=" << x);
    IntegerValue a{kind, x};
    ASSERT_EQ(x, a.ToUInt64());
    ASSERT_EQ(kind, a.kind());
    IntegerValue copy{a};
    ASSERT_EQ(x, copy.ToUInt64());
    copy = a;
    ASSERT_EQ(x, copy.ToUInt64());
    ASSERT_EQ(x == 0, a.IsZero());

    // Decimal and hexadecimal formatting round-trip through Read().
    std::string udec{a.UnsignedDecimal()};
    const char *p{udec.c_str()};
    auto readDecimal{IntegerValue::Read(kind, p, 10, /*isSigned=*/false)};
    ASSERT_FALSE(readDecimal.overflow);
    ASSERT_EQ(x, readDecimal.value.ToUInt64());
    ASSERT_EQ('\0', *p);
    std::string hex{a.Hexadecimal()};
    p = hex.c_str();
    auto readHex{IntegerValue::Read(kind, p, 16, /*isSigned=*/false)};
    ASSERT_FALSE(readHex.overflow);
    ASSERT_EQ(x, readHex.value.ToUInt64());
    ASSERT_EQ('\0', *p);

    ASSERT_EQ(x ^ maxUnsigned, a.NOT().ToUInt64());

    const bool isNegative{(x >> (bits - 1)) != 0};
    const bool isMostNegative{x == (uint64_t{1} << (bits - 1))};
    auto negated{a.Negate()};
    ASSERT_EQ(isMostNegative, negated.overflow);
    ASSERT_EQ((~x + 1) & maxUnsigned, negated.value.ToUInt64());
    auto abs{a.ABS()};
    ASSERT_EQ(isMostNegative, abs.overflow);
    ASSERT_EQ(isNegative ? (~x + 1) & maxUnsigned : x, abs.value.ToUInt64());

    const int lzbc{a.LEADZ()};
    ASSERT_GE(lzbc, 0);
    ASSERT_LE(lzbc, bits);
    ASSERT_EQ(x == 0, lzbc == bits);
    ASSERT_LT(x, uint64_t{1} << (bits - lzbc));
    ASSERT_GE(x + x + !x, uint64_t{1} << (bits - lzbc));

    int popcheck{0};
    for (int j{0}; j < bits; ++j) {
      popcheck += (x >> j) & 1;
    }
    ASSERT_EQ(popcheck, a.POPCNT());
    ASSERT_EQ((popcheck & 1) != 0, a.POPPAR());
    int trailcheck{0};
    for (; trailcheck < bits; ++trailcheck) {
      if ((x >> trailcheck) & 1) {
        break;
      }
    }
    ASSERT_EQ(trailcheck, a.TRAILZ());
    for (int j{0}; j < bits; ++j) {
      ASSERT_EQ(((x >> j) & 1) != 0, a.BTEST(j)) << "bit " << j;
    }

    const int64_t sx{a.ToInt64()};
    if (isNegative) {
      ASSERT_TRUE(a.IsNegative());
      ASSERT_LT(sx, 0);
      ASSERT_EQ(Ordering::Less, a.CompareToZeroSigned());
    } else {
      ASSERT_FALSE(a.IsNegative());
      ASSERT_GE(sx, 0);
      ASSERT_EQ(x == 0 ? Ordering::Equal : Ordering::Greater,
          a.CompareToZeroSigned());
    }
    ASSERT_EQ(x, uint64_t(sx) & maxUnsigned);

    for (int count{0}; count <= bits + 1; ++count) {
      const uint64_t left{(x << count) & maxUnsigned};
      ASSERT_EQ(left, a.SHIFTL(count).ToUInt64()) << "count=" << count;
      ASSERT_EQ(left, a.ISHFT(count).ToUInt64()) << "count=" << count;
      const uint64_t right{x >> count};
      ASSERT_EQ(right, a.SHIFTR(count).ToUInt64()) << "count=" << count;
      ASSERT_EQ(right, a.ISHFT(-count).ToUInt64()) << "count=" << count;
      const uint64_t fill{isNegative ? ~uint64_t{0} : 0};
      const uint64_t arithmetic{count >= bits
              ? fill & maxUnsigned
              : (right | ((fill << (bits - count)) & maxUnsigned))};
      ASSERT_EQ(arithmetic, a.SHIFTA(count).ToUInt64()) << "count=" << count;
    }
  }
}

TEST(IntegerValue, ExhaustiveUnaryKind1) { ExhaustiveUnary(1); }

TEST(IntegerValue, ExhaustiveUnaryKind2) { ExhaustiveUnary(2); }

TEST(IntegerValue, ExhaustiveDyadicKind1) {
  constexpr int kind{1};
  constexpr int bits{8};
  constexpr uint64_t maxUnsigned{0xff};
  constexpr int64_t maxPositiveSigned{0x7f};
  constexpr int64_t mostNegativeSigned{-0x80};

  for (uint64_t x{0}; x <= maxUnsigned; ++x) {
    IntegerValue a{kind, x};
    const int64_t sx{a.ToInt64()};
    for (uint64_t y{0}; y <= maxUnsigned; ++y) {
      SCOPED_TRACE(testing::Message() << "x=" << x << " y=" << y);
      IntegerValue b{kind, y};
      const int64_t sy{b.ToInt64()};

      ASSERT_EQ(x < y ? Ordering::Less
              : x > y ? Ordering::Greater
                      : Ordering::Equal,
          a.CompareUnsigned(b));
      ASSERT_EQ(x >= y, a.BGE(b));
      ASSERT_EQ(x > y, a.BGT(b));
      ASSERT_EQ(x <= y, a.BLE(b));
      ASSERT_EQ(x < y, a.BLT(b));
      ASSERT_EQ(sx < sy ? Ordering::Less
              : sx > sy ? Ordering::Greater
                        : Ordering::Equal,
          a.CompareSigned(b));
      ASSERT_EQ(sx < sy, a < b);
      ASSERT_EQ(sx == sy, a == b);

      ASSERT_EQ(x & y, a.IAND(b).ToUInt64());
      ASSERT_EQ(x | y, a.IOR(b).ToUInt64());
      ASSERT_EQ(x ^ y, a.IEOR(b).ToUInt64());
      ASSERT_EQ(std::max(sx, sy), a.MAX(b).ToInt64());
      ASSERT_EQ(std::min(sx, sy), a.MIN(b).ToInt64());

      auto sum{a.AddUnsigned(b)};
      ASSERT_EQ(x + y, sum.value.ToUInt64() + (uint64_t{sum.carry} << bits));
      auto ssum{a.AddSigned(b)};
      ASSERT_EQ(uint64_t(sx + sy) & maxUnsigned, ssum.value.ToUInt64());
      ASSERT_EQ(sx + sy < mostNegativeSigned || sx + sy > maxPositiveSigned,
          ssum.overflow);
      auto diff{a.SubtractSigned(b)};
      ASSERT_EQ(uint64_t(sx - sy) & maxUnsigned, diff.value.ToUInt64());
      ASSERT_EQ(sx - sy < mostNegativeSigned || sx - sy > maxPositiveSigned,
          diff.overflow);
      auto dim{a.DIM(b)};
      ASSERT_EQ(
          sx > sy ? uint64_t(sx - sy) & maxUnsigned : 0, dim.value.ToUInt64());
      auto sign{a.SIGN(b)};
      ASSERT_EQ(uint64_t(sy < 0 ? -std::abs(sx) : std::abs(sx)) & maxUnsigned,
          sign.value.ToUInt64());

      auto product{a.MultiplyUnsigned(b)};
      ASSERT_EQ(
          x * y, (product.upper.ToUInt64() << bits) | product.lower.ToUInt64());
      auto sproduct{a.MultiplySigned(b)};
      ASSERT_EQ(uint64_t(sx * sy) & maxUnsigned, sproduct.lower.ToUInt64());
      ASSERT_EQ(
          uint64_t((sx * sy) >> bits) & maxUnsigned, sproduct.upper.ToUInt64());

      auto quot{a.DivideUnsigned(b)};
      ASSERT_EQ(y == 0, quot.divisionByZero);
      if (y == 0) {
        ASSERT_EQ(maxUnsigned, quot.quotient.ToUInt64());
        ASSERT_TRUE(quot.remainder.IsZero());
      } else {
        ASSERT_EQ(x / y, quot.quotient.ToUInt64());
        ASSERT_EQ(x % y, quot.remainder.ToUInt64());
      }

      auto squot{a.DivideSigned(b)};
      const bool badCase{sx == mostNegativeSigned && sy == -1};
      ASSERT_EQ(y == 0, squot.divisionByZero);
      ASSERT_EQ(badCase, squot.overflow);
      if (y == 0) {
        ASSERT_EQ(sx >= 0 ? maxPositiveSigned : mostNegativeSigned,
            squot.quotient.ToInt64());
        ASSERT_TRUE(squot.remainder.IsZero());
      } else if (badCase) {
        ASSERT_EQ(sx, squot.quotient.ToInt64());
        ASSERT_TRUE(squot.remainder.IsZero());
      } else {
        ASSERT_EQ(sx / sy, squot.quotient.ToInt64());
        ASSERT_EQ(sx % sy, squot.remainder.ToInt64());
        int64_t modulo{sx % sy};
        if (modulo != 0 && ((sx < 0) != (sy < 0))) {
          modulo += sy;
        }
        ASSERT_EQ(uint64_t(modulo) & maxUnsigned, a.MODULO(b).value.ToUInt64());
      }
    }
  }
}

TYPED_TEST(IntegerValueTypedKind, Print) {
  constexpr int kind{TypeParam::kind};
  constexpr int pos{IntKindPos<TypeParam>};

  llvm::SmallString<128> buf;
  llvm::raw_svector_ostream os{buf};
  IntegerValue abc{kind, 42};
  abc.print(os);

  const char *results[]{"42_1", "42_2", "42_4", "42_8", "42_16"};
  EXPECT_EQ(results[pos], os.str());
}

} // namespace
