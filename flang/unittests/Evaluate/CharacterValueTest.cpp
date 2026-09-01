//===-- flang/unittests/Evaluate/CharacterValueTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/template.h"
#include "flang/Common/type-kinds.h"
#include "flang/Evaluate/character-value.h"
#include "flang/Evaluate/typekind-traits.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
#include <initializer_list>
#include <string>

using namespace Fortran::common;
using namespace Fortran::evaluate;
using namespace Fortran::evaluate::value;

namespace {

using CharacterTypedKinds = testing::Types<TypeKind<TypeCategory::Character, 1>,
    TypeKind<TypeCategory::Character, 2>, TypeKind<TypeCategory::Character, 4>>;
template <typename Target>
inline constexpr std::size_t CharacterKindPos =
    type_index<Target, CharacterTypedKinds>::value;
struct KindName {
  template <typename TP> static std::string GetName(int) {
    return "CHARACTER_" + std::to_string(TP::kind);
  }
};

template <typename T> class CharacterValueTypedKind : public testing::Test {};
TYPED_TEST_SUITE(CharacterValueTypedKind, CharacterTypedKinds, KindName);

class CharacterValueKind : public testing::TestWithParam<int> {};
INSTANTIATE_TEST_SUITE_P(CharacterValueKind, CharacterValueKind,
    testing::ValuesIn(CharacterKinds),
    [](const testing::TestParamInfo<int> &info) {
      return "CHARACTER_" + std::to_string(info.param);
    });

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static testing::AssertionResult CharsEqual(const char *expectedExpr,
    const char *valueExpr, llvm::StringRef expected, const CharacterValue &v) {
  std::string actual{v.ToStdString()};
  if (expected == actual) {
    return testing::AssertionSuccess();
  }
  return testing::AssertionFailure()
      << valueExpr << " is \"" << actual << "\", expected " << expectedExpr
      << " (\"" << expected << "\")";
}

#define EXPECT_CHARS_EQ(expected, value) \
  EXPECT_PRED_FORMAT2(CharsEqual, expected, value)

/// Writes one character of the value's own character type at "dst".
static void PutChar(int kind, void *dst, char32_t c) {
  CharacterValue::withCharProto(kind, [=](auto proto) {
    using CharT = std::decay_t<decltype(proto)>;
    CharT raw{static_cast<CharT>(c)};
    std::memcpy(dst, &raw, sizeof(raw));
  });
}

//===----------------------------------------------------------------------===//
// Construction, assignment and kind inquiries
//===----------------------------------------------------------------------===//

TEST(CharacterValue, Monostate) {
  CharacterValue v;
  EXPECT_TRUE(v.IsMonostate());

  // Monostate behaves like an empty string
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(0u, v.size());
  EXPECT_EQ(0u, v.length());

  // A monostate is converted to an empty string of any representation
  EXPECT_CHARS_EQ("", v);
  EXPECT_EQ(llvm::StringRef{}, *v.AsStringRef());
  EXPECT_EQ(std::string{}, *v.AsStdString());
  EXPECT_EQ(std::u16string{}, *v.AsU16String());
  EXPECT_EQ(std::u32string{}, *v.AsU32String());
  EXPECT_EQ(std::string{}, v.ToStdString());
}

TYPED_TEST(CharacterValueTypedKind, ConstructFromStdBasicString) {
  using CharT = typename TypeParam::CharT;
  using StringT = typename TypeParam::StringT;
  constexpr int kind{TypeParam::kind};

  CharT buffer[] = {'a', 'b', 'c', '\0'};
  CharacterValue v{kind, StringT{buffer}};

  EXPECT_FALSE(v.IsMonostate());
  EXPECT_EQ(StringT{buffer}, v.AsBasicString<CharT>());
}

TEST_P(CharacterValueKind, Zero) {
  const int kind{GetParam()};
  CharacterValue zero{CharacterValue::Zero(kind)};
  CharacterValue empty{kind, ""};
  CharacterValue monostate;

  EXPECT_FALSE(zero.IsMonostate());
  EXPECT_EQ(kind, zero.kind());
  EXPECT_TRUE(zero.empty());
  EXPECT_EQ(0u, zero.bytesStored());
  EXPECT_CHARS_EQ("", zero);
  EXPECT_EQ(empty, zero);
  EXPECT_EQ(monostate, empty);
}

TEST_P(CharacterValueKind, FillConstructor) {
  const int kind{GetParam()};

  CharacterValue v(kind, 3, U'x');
  EXPECT_FALSE(v.IsMonostate());
  EXPECT_EQ(kind, v.kind());
  EXPECT_EQ(3u, v.size());
  EXPECT_CHARS_EQ("xxx", v);

  // A zero-length fill is still kind-typed
  CharacterValue none(kind, 0, U'x');
  EXPECT_FALSE(none.IsMonostate());
  EXPECT_TRUE(none.empty());
  EXPECT_EQ(kind, none.kind());
}

TEST(CharacterValue, SubscriptWidensToChar32) {
  CharacterValue u{1, std::string{"\x80"}};
  EXPECT_EQ(char32_t('\x80'), u[0]);

  CharacterValue w{2, std::u16string{u"\u0100"}};
  EXPECT_EQ(char32_t{u'\u0100'}, w[0]);

  CharacterValue v{4, std::u32string{U"\U0001F600"}};
  EXPECT_EQ(U'\U0001F600', v[0]);
}

TEST_P(CharacterValueKind, CopyAndMove) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abc"};

  CharacterValue copyConstructed{v};
  EXPECT_EQ(kind, v.kind());
  EXPECT_TRUE(v == copyConstructed);

  CharacterValue moveConstructed{std::move(copyConstructed)};
  EXPECT_EQ(kind, moveConstructed.kind());
  EXPECT_TRUE(v == moveConstructed);

  CharacterValue copyAssigned;
  copyAssigned = v;
  EXPECT_EQ(kind, copyAssigned.kind());
  EXPECT_TRUE(v == copyAssigned);

  CharacterValue moveAssigned;
  moveAssigned = std::move(copyAssigned);
  EXPECT_EQ(kind, moveAssigned.kind());
  EXPECT_TRUE(v == moveAssigned);
}

TEST_P(CharacterValueKind, CharSize) {
  const int kind{GetParam()};

  CharacterValue v{kind, "abcd"};
  EXPECT_EQ(kind, v.kind());
  EXPECT_EQ(std::size_t(kind), v.charSize());
}

TEST_P(CharacterValueKind, SizeAndLength) {
  const int kind{GetParam()};

  CharacterValue v{kind, "hello"};
  EXPECT_FALSE(v.empty());
  EXPECT_EQ(5u, v.size());
  EXPECT_EQ(5u, v.length());
}

//===----------------------------------------------------------------------===//
// Conversions to host string types
//===----------------------------------------------------------------------===//

TEST_P(CharacterValueKind, AsStringConversions) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abc"};

  // Only the conversion matching the stored character type is available.
  EXPECT_EQ(kind == 1, v.AsStringRef().has_value());
  EXPECT_EQ(kind == 1, v.AsStdString().has_value());
  EXPECT_EQ(kind == 2, v.AsU16String().has_value());
  EXPECT_EQ(kind == 4, v.AsU32String().has_value());

  switch (kind) {
  case 1:
    EXPECT_EQ("abc", *v.AsStringRef());
    EXPECT_EQ("abc", *v.AsStdString());
    break;
  case 2:
    EXPECT_EQ(std::u16string{u"abc"}, *v.AsU16String());
    break;
  case 4:
    EXPECT_EQ(std::u32string{U"abc"}, *v.AsU32String());
    break;
  }
  EXPECT_EQ("abc", v.ToStdString());
}

TYPED_TEST(CharacterValueTypedKind, ToBasicString) {
  using CharT = typename TypeParam::CharT;
  using StringT = typename TypeParam::StringT;
  constexpr int kind{TypeParam::kind};

  const CharT data[] = {'a', 'b', 'c', '\0'};
  CharacterValue v1{kind, data};
  EXPECT_EQ(StringT{data}, v1.AsBasicString<CharT>());
}

TEST_P(CharacterValueKind, WithStdString) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abcde"};

  // The callable sees the concrete std::basic_string<> for the stored kind.
  EXPECT_EQ(5u, v.withStdString([](const auto &s) { return s.size(); }));
  EXPECT_EQ(std::size_t(kind), v.withStdString([](const auto &s) {
    return sizeof(typename std::decay_t<decltype(s)>::value_type);
  }));
}

TEST_P(CharacterValueKind, ToAscii) {
  const int kind{GetParam()};

  // conversion to possible kinds
  CharacterValue v{kind, "abc"};
  for (int to : std::initializer_list<int> FORTRAN_CHARACTER_KINDS) {
    CharacterValue converted{v.ToAscii(to)};
    EXPECT_EQ(to, converted.kind());
    EXPECT_CHARS_EQ("abc", converted);
  }

  // Conversion between kinds is defined only for 7-bit ASCII; anything else
  // yields an empty string.
  CharacterValue nonascii{4, std::u32string{U"a\u0100b"}};
  EXPECT_TRUE(nonascii.ToAscii(kind).empty());
  EXPECT_EQ(kind, nonascii.ToAscii(kind).kind());

  // Converting a monostate yields an empty string of the target kind.
  CharacterValue empty{CharacterValue{}.ToAscii(kind)};
  EXPECT_EQ(kind, empty.kind());
  EXPECT_TRUE(empty.empty());
}

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

TEST_P(CharacterValueKind, Compare) {
  const int kind{GetParam()};

  CharacterValue abc{kind, "abc"};
  CharacterValue abd{kind, "abd"};
  CharacterValue ab{kind, "ab"};
  CharacterValue ab_{kind, "ab "};
  CharacterValue empty{kind, ""};

  EXPECT_EQ(Ordering::Equal, abc.Compare(abc));
  EXPECT_EQ(Ordering::Less, abc.Compare(abd));
  EXPECT_EQ(Ordering::Greater, abd.Compare(abc));

  // Fortran CHARACTER comparison blank-pads the shorter operand, so a trailing
  // blank does not make a difference ...
  EXPECT_EQ(Ordering::Equal, ab.Compare(ab_));

  // ... whereas any other trailing character does.
  EXPECT_EQ(Ordering::Less, ab.Compare(abc));

  // A monostate compares as an empty string of the other operand's kind.
  CharacterValue monostate;
  EXPECT_EQ(Ordering::Equal, monostate.Compare(empty));
  EXPECT_EQ(Ordering::Less, monostate.Compare(abc));
  EXPECT_EQ(Ordering::Greater, abc.Compare(monostate));
}

TEST_P(CharacterValueKind, RelationalOperators) {
  const int kind{GetParam()};
  CharacterValue abc{kind, "abc"};
  CharacterValue abd{kind, "abd"};

  EXPECT_TRUE(abc == abc);
  EXPECT_FALSE(abc != abc);
  EXPECT_TRUE(abc != abd);
  EXPECT_TRUE(abc < abd);
  EXPECT_TRUE(abc <= abd);
  EXPECT_TRUE(abc <= abc);
  EXPECT_TRUE(abd > abc);
  EXPECT_TRUE(abd >= abc);
  EXPECT_TRUE(abc >= abc);
  EXPECT_FALSE(abd < abc);

  // The operators have std::basic_string semantics, which - unlike Compare() -
  // do not blank-pad the shorter operand.
  CharacterValue ab{kind, "ab"};
  CharacterValue ab_{kind, "ab "};
  EXPECT_TRUE(ab != ab_);
  EXPECT_TRUE(ab < ab_);

  // A monostate is an empty string here too.
  CharacterValue monostate;
  CharacterValue empty{kind, ""};
  EXPECT_TRUE(monostate == empty);
  EXPECT_TRUE(monostate < abc);
}

//===----------------------------------------------------------------------===//
// Mutation
//===----------------------------------------------------------------------===//

TEST_P(CharacterValueKind, AssignFill) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abc"};

  v.assign(kind, 2, 'z');
  EXPECT_EQ(kind, v.kind());
  EXPECT_CHARS_EQ("zz", v);

  // assign() also fixes the kind of a monostate, and can change the kind.
  CharacterValue fresh;
  fresh.assign(kind, 1, 'q');
  EXPECT_EQ(kind, fresh.kind());
  EXPECT_CHARS_EQ("q", fresh);
}

TEST_P(CharacterValueKind, AssignFromPointerAndLength) {
  CharacterValue v;

  // char
  v.assign("abcd", 3);
  EXPECT_EQ(1, v.kind());
  EXPECT_CHARS_EQ("abc", v);

  // char16_t
  v.assign(u"abcd", 2);
  EXPECT_EQ(2, v.kind());
  EXPECT_CHARS_EQ("ab", v);

  // char32_t
  v.assign(U"abcd", 4);
  EXPECT_EQ(4, v.kind());
  EXPECT_CHARS_EQ("abcd", v);
}

TEST_P(CharacterValueKind, Erase) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abcdef"};

  v.erase(3);
  EXPECT_EQ(kind, v.kind());
  EXPECT_CHARS_EQ("abc", v);

  v.erase(0);
  EXPECT_EQ(kind, v.kind());
  EXPECT_TRUE(v.empty());
}

TEST_P(CharacterValueKind, Append) {
  const int kind{GetParam()};
  CharacterValue v{kind, "ab"};

  v.append(3, '!');
  EXPECT_CHARS_EQ("ab!!!", v);

  v.append(0, '?');
  EXPECT_CHARS_EQ("ab!!!", v);
}

TEST_P(CharacterValueKind, Replace) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abcdef"};

  CharacterValue xy{kind, "XY"};
  EXPECT_EQ(&v, &v.replace(1, 2, xy));
  EXPECT_CHARS_EQ("aXYdef", v);

  // The replacement need not have the same length as the replaced substring.
  CharacterValue hyph{kind, "-"};
  v.replace(0, 3, hyph);
  EXPECT_CHARS_EQ("-def", v);
}

TEST_P(CharacterValueKind, Substr) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abcdef"};

  EXPECT_CHARS_EQ("cdef", v.substr(2));
  EXPECT_EQ(kind, v.substr(2).kind());
  EXPECT_CHARS_EQ("cd", v.substr(2, 2));

  // A length reaching past the end is clamped.
  EXPECT_CHARS_EQ("ef", v.substr(4, 100));
  EXPECT_TRUE(v.substr(6).empty());

  // The original is unchanged.
  EXPECT_CHARS_EQ("abcdef", v);
}

TEST_P(CharacterValueKind, Reserve) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abc"};

  // Reserving capacity does not change the value.
  v.reserve(100);
  EXPECT_EQ(kind, v.kind());
  EXPECT_CHARS_EQ("abc", v);
  EXPECT_EQ(3u, v.size());
}

TEST_P(CharacterValueKind, Subscript) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abc"};
  EXPECT_EQ(U'a', v[0]);
  EXPECT_EQ(U'b', v[1]);
  EXPECT_EQ(U'c', v[2]);
}

TEST_P(CharacterValueKind, Concatenation) {
  const int kind{GetParam()};
  CharacterValue ab{kind, "ab"};
  CharacterValue cd{kind, "cd"};
  CharacterValue empty = CharacterValue::Zero(kind);

  CharacterValue sum{ab + cd};
  EXPECT_EQ(kind, sum.kind());
  EXPECT_CHARS_EQ("abcd", sum);

  // Concatenating an empty string is the identity.
  EXPECT_CHARS_EQ("ab", ab + empty);
}

TEST_P(CharacterValueKind, AppendAssignString) {
  const int kind{GetParam()};

  CharacterValue v{kind, "ab"};
  CharacterValue cd{kind, "cd"};
  EXPECT_EQ(&v, &(v += cd));
  EXPECT_CHARS_EQ("abcd", v);
}

TEST_P(CharacterValueKind, AppendAssignChar) {
  const int kind{GetParam()};

  CharacterValue v{kind, "ab"};
  EXPECT_EQ(kind, v.kind());
  EXPECT_EQ(&v, &(v += 'c'));
  EXPECT_CHARS_EQ("abc", v);
}

//===----------------------------------------------------------------------===//
// Searching
//===----------------------------------------------------------------------===//

TEST(CharacterValue, Npos) {
  EXPECT_EQ(std::string::npos, CharacterValue::npos);
}

TEST_P(CharacterValueKind, Find) {
  const int kind{GetParam()};
  CharacterValue abcabc{kind, "abcabc"};
  CharacterValue bc{kind, "bc"};
  CharacterValue abc{kind, "abc"};
  CharacterValue empty{kind, ""};
  CharacterValue xyz{kind, "xyz"};
  CharacterValue a{kind, "a"};
  CharacterValue monostate;

  EXPECT_EQ(1u, abcabc.find(bc));
  EXPECT_EQ(0u, abcabc.find(abc));
  EXPECT_EQ(CharacterValue::npos, abcabc.find(xyz));

  // Find empty string at begnning
  EXPECT_EQ(0u, abcabc.find(empty));
  EXPECT_EQ(0u, abcabc.find(monostate));
  EXPECT_EQ(0u, empty.find(empty));
  EXPECT_EQ(0u, monostate.find(empty));
  EXPECT_EQ(0u, monostate.find(monostate));

  // Nothing is ever found in a value of unknown kind
  EXPECT_EQ(CharacterValue::npos, monostate.find(a));
}

TEST_P(CharacterValueKind, RFind) {
  const int kind{GetParam()};
  CharacterValue v{kind, "abcabc"};
  CharacterValue bc{kind, "bc"};
  CharacterValue abc{kind, "abc"};
  CharacterValue xyz{kind, "xyz"};

  EXPECT_EQ(4u, v.rfind(bc));
  EXPECT_EQ(3u, v.rfind(abc));
  EXPECT_EQ(CharacterValue::npos, v.rfind(xyz));
}

TEST_P(CharacterValueKind, FindFirstOf) {
  const int kind{GetParam()};
  CharacterValue v{kind, "hello"};
  CharacterValue le{kind, "le"};
  CharacterValue h{kind, "he"};
  CharacterValue xyz{kind, "xyz"};
  CharacterValue empty{kind, ""};

  EXPECT_EQ(1u, v.find_first_of(le));
  EXPECT_EQ(0u, v.find_first_of(h));
  EXPECT_EQ(CharacterValue::npos, v.find_first_of(xyz));
  EXPECT_EQ(CharacterValue::npos, v.find_first_of(empty));
}

TEST_P(CharacterValueKind, FindLastOf) {
  const int kind{GetParam()};
  CharacterValue v{kind, "hello"};
  CharacterValue le{kind, "le"};
  CharacterValue o{kind, "o"};
  CharacterValue xyz{kind, "xyz"};

  EXPECT_EQ(3u, v.find_last_of(le));
  EXPECT_EQ(4u, v.find_last_of(o));
  EXPECT_EQ(CharacterValue::npos, v.find_last_of(xyz));
}

TEST_P(CharacterValueKind, FindFirstNotOfCharacter) {
  const int kind{GetParam()};
  CharacterValue aab{kind, "aab"};
  CharacterValue aaa{kind, "aaa"};

  EXPECT_EQ(2u, aab.find_first_not_of(U'a'));
  EXPECT_EQ(0u, aab.find_first_not_of(U'b'));
  EXPECT_EQ(CharacterValue::npos, aaa.find_first_not_of(U'a'));
}

TEST_P(CharacterValueKind, FindLastNotOfCharacter) {
  const int kind{GetParam()};
  CharacterValue abb{kind, "abb"};
  CharacterValue bbb{kind, "bbb"};

  EXPECT_EQ(0u, abb.find_last_not_of(U'b'));
  EXPECT_EQ(2u, abb.find_last_not_of(U'a'));
  EXPECT_EQ(CharacterValue::npos, bbb.find_last_not_of(U'b'));
}

TEST_P(CharacterValueKind, FindFirstNotOfSet) {
  const int kind{GetParam()};
  CharacterValue v{kind, "aabbc"};
  CharacterValue ab{kind, "ab"};
  CharacterValue abc{kind, "abc"};
  CharacterValue xyz{kind, "xyz"};
  CharacterValue a{kind, "a"};
  CharacterValue empty{kind, ""};
  CharacterValue monostate;

  EXPECT_EQ(4u, v.find_first_not_of(ab));
  EXPECT_EQ(0u, v.find_first_not_of(xyz));
  EXPECT_EQ(CharacterValue::npos, v.find_first_not_of(abc));
  EXPECT_EQ(CharacterValue::npos, empty.find_first_not_of(a));
  EXPECT_EQ(CharacterValue::npos, monostate.find_first_not_of(a));
}

TEST_P(CharacterValueKind, FindLastNotOfSet) {
  const int kind{GetParam()};
  CharacterValue v{kind, "aabbc"};
  CharacterValue abc{kind, "abc"};
  CharacterValue bc{kind, "bc"};
  CharacterValue xyz{kind, "xyz"};
  CharacterValue a{kind, "a"};
  CharacterValue empty{kind, ""};
  CharacterValue monostate;

  EXPECT_EQ(1u, v.find_last_not_of(bc));
  EXPECT_EQ(4u, v.find_last_not_of(xyz));
  EXPECT_EQ(CharacterValue::npos, v.find_last_not_of(abc));
  EXPECT_EQ(CharacterValue::npos, empty.find_last_not_of(a));
  EXPECT_EQ(CharacterValue::npos, monostate.find_last_not_of(a));
}

//===----------------------------------------------------------------------===//
// Raw storage
//===----------------------------------------------------------------------===//

TEST_P(CharacterValueKind, Data) {
  const int kind{GetParam()};
  CharacterValue abc{kind, "abc"};
  CharacterValue same{abc};
  CharacterValue v{kind, "abc"};
  const CharacterValue &constRef{v};

  ASSERT_EQ(v.bytesStored(), std::size_t(3 * kind));
  ASSERT_EQ(same.bytesStored(), v.bytesStored());
  EXPECT_EQ(v.data(), static_cast<void *>(v.charData()));
  EXPECT_EQ(constRef.data(), static_cast<const void *>(constRef.charData()));
  EXPECT_EQ(0, std::memcmp(v.data(), same.data(), v.bytesStored()));

  // Writing through data() is visible in the value.
  PutChar(kind, v.data(), U'A');
  EXPECT_EQ(U'A', v[0]);
}

TYPED_TEST(CharacterValueTypedKind, At) {
  constexpr int kind{TypeParam::kind};
  CharacterValue v{kind, "abc"};
  const CharacterValue &constRef{v};

  EXPECT_EQ(v.data(), v.at(0));
  EXPECT_EQ(static_cast<void *>(v.charData() + 2 * v.charSize()), v.at(2));
  EXPECT_EQ(static_cast<const void *>(constRef.charData() + v.charSize()),
      constRef.at(1));

  // The character at that address is the one reported by operator[].
  PutChar(kind, v.at(1), U'Z');
  EXPECT_EQ(U'Z', v[1]);
  EXPECT_CHARS_EQ("aZc", v);
}

TYPED_TEST(CharacterValueTypedKind, StoreRawBytes) {
  using CharT = typename TypeParam::CharT;
  constexpr int kind{TypeParam::kind};
  CharacterValue v{kind, "abc"};

  CharT buffer[4]{};

  bool changed1{false};
  v.StoreRawBytes(buffer, 3 * sizeof(CharT), &changed1);
  EXPECT_TRUE(changed1);
  EXPECT_EQ(CharT{'a'}, buffer[0]);
  EXPECT_EQ(CharT{'b'}, buffer[1]);
  EXPECT_EQ(CharT{'c'}, buffer[2]);

  // Storing the same bytes again reports no change.
  bool changed2{false};
  v.StoreRawBytes(buffer, 3 * sizeof(CharT), &changed2);
  EXPECT_FALSE(changed2);

  // Storing fewer than available chars
  bool changed3{false};
  buffer[1] = 'X';
  buffer[2] = 'X';
  v.StoreRawBytes(buffer, 2 * sizeof(CharT), &changed3);
  EXPECT_TRUE(changed3);
  EXPECT_EQ(CharT{'b'}, buffer[1]);
  EXPECT_EQ(CharT{'X'}, buffer[2]);

  // A larger destination is zero-filled beyond the payload, and that padding
  // counts towards whether anything changed.
  bool changed4{false};
  buffer[3] = 'X';
  v.StoreRawBytes(buffer, 4 * sizeof(CharT), &changed4);
  EXPECT_TRUE(changed4);
  EXPECT_EQ(CharT{U' '}, buffer[3]);

  // No change reported even with padding
  bool changed5{false};
  v.StoreRawBytes(buffer, 4 * sizeof(CharT), &changed5);
  EXPECT_FALSE(changed5);
}

TYPED_TEST(CharacterValueTypedKind, FromRawBytes) {
  using CharT = typename TypeParam::CharT;
  constexpr int kind{TypeParam::kind};

  CharT data[] = {'a', 'b', 'c', '\0'};
  CharacterValue reference{kind, std::basic_string<CharT>(data)};

  CharacterValue restored{
      CharacterValue::FromRawBytes(kind, data, 3 * sizeof(CharT))};
  EXPECT_EQ(kind, restored.kind());
  EXPECT_EQ(reference, restored);

  // Read an empty string
  CharacterValue empty{CharacterValue::FromRawBytes(kind, data, 0)};
  EXPECT_EQ(kind, empty.kind());
  EXPECT_TRUE(empty.empty());
}

TYPED_TEST(CharacterValueTypedKind, Print) {
  using CharT = typename TypeParam::CharT;
  constexpr int kind{TypeParam::kind};
  constexpr int pos{CharacterKindPos<TypeParam>};

  llvm::SmallString<128> buf;
  llvm::raw_svector_ostream os{buf};
  const CharT data[] = {'a', 'b', 'c', '\0'};
  CharacterValue abc{kind, data};
  abc.print(os);

  const char *results[]{"1_\"abc\"", "2_\"abc\"", "4_\"abc\""};
  EXPECT_EQ(results[pos], os.str());
}

} // namespace
