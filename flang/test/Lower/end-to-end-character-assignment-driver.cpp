//===-- test/lower/test-character-assignment.cc -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#include <cassert>
#include <codecvt>
#include <iomanip>
#include <iostream>
#include <locale>
#include <string>
#include <type_traits>

// Driver to tests Fortran subroutine from character-assignment.f90

// So far lowering of fir::boxchar dummy to llvm does not layout character
// arguments like other compiler do for F77. Templates provides a patch for
// that.

using LenT = std::int64_t;
struct Fchar {
  char *data;
  LenT len;
};

template <typename F, typename... T>
void CallSubroutine(F f, Fchar s1, Fchar s2, T... args) {
  f(s1.data, s2.data, args..., s1.len, s2.len);
}

// Define structures to create and manipulate Fortran Character
// A canary is always added at the end of character storage so that
// invalid overwrites can be detected.
template <int K> struct CharStorage {};
template <> struct CharStorage<1> {
  using Type = std::string;
  static const Type canary;
};
const CharStorage<1>::Type CharStorage<1>::canary{"_CaNaRy"};

template <> struct CharStorage<2> {
  using Type = std::u16string;
  static const Type canary;
};
const CharStorage<2>::Type CharStorage<2>::canary{u"_CaNaRy"};

template <> struct CharStorage<4> {
  using Type = std::u32string;
  static const Type canary;
};
const CharStorage<4>::Type CharStorage<4>::canary{U"_CaNaRy"};

template <int Kind> struct FcharData {
  using String = typename CharStorage<Kind>::Type;
  using CharT = typename String::value_type;
  FcharData(String str)
      : data{str + CharStorage<Kind>::canary}, len{static_cast<LenT>(
                                                   str.length())} {}
  Fchar getFchar() {
    const char *addr{reinterpret_cast<const char *>(data.data())};
    return Fchar{const_cast<char *>(addr), len};
  }

  // UTF-8 dump
  std::ostream &dump(std::ostream &os) const {
    if constexpr (std::is_same_v<CharT, char>) {
      os << data;
    } else {
      std::wstring_convert<std::codecvt_utf8<CharT>, CharT> cvt;
      os << cvt.to_bytes(data);
    }
    return os;
  }
  // Hex dump
  std::ostream &dumpHex(std::ostream &os) const {
    os << std::hex;
    for (auto c : data) {
      if constexpr (std::is_same_v<CharT, char>) {
        os << " 0x" << std::setw(2) << std::setfill('0')
           << (int)((unsigned char)c);
      } else {
        os << " 0x" << std::setw(sizeof(CharT) * 2) << std::setfill('0') << c;
      }
    }
    os << std::dec;
    return os;
  }

  String data;
  LenT len; // may differ from string length for test purposes
};

template <int Kind>
bool Check(const FcharData<Kind> &test, const FcharData<Kind> &ref,
    const std::string &desc) {
  if (test.data != ref.data) {
    std::cout << "Failed: " << desc << std::endl;
    ref.dump(std::cout << "  expected: '") << "'" << std::endl;
    test.dump(std::cout << "  got     : '") << "'" << std::endl;
    return false;
  }
  return true;
}

// Compare against precomputed results.
template <int Kind, typename F, typename... T>
bool TestSubroutine(const std::string &testName, F fooTest,
    const FcharData<Kind> &s1, const FcharData<Kind> &refS1,
    const FcharData<Kind> &s2, const FcharData<Kind> &refS2, T... otherArgs) {
  // Make copies because data may be modified
  FcharData<Kind> testS1{s1}, testS2{s2};
  CallSubroutine(fooTest, testS1.getFchar(), testS2.getFchar(), otherArgs...);
  auto description{testName + " KIND=" + std::to_string(Kind)};
  bool result{Check(testS1, refS1, description + " s1")};
  result &= Check(testS2, refS2, description + " s2");
  return result;
}

// Test driver code (could maybe generated somehow)

// String data to be used as inputs during the tests.
template <int Kind> struct Inputs { static FcharData<Kind> s1, s2, s3; };

template <> FcharData<1> Inputs<1>::s1{"aw*lSe4frliaw"};
template <> FcharData<1> Inputs<1>::s2{"8\n e7t4$%&52Z"};
template <> FcharData<1> Inputs<1>::s3{"quAli64^&$*#$8gl6"};

template <> FcharData<2> Inputs<2>::s1{u"\u4e4dhy7&3o8%\u4e24"};
template <> FcharData<2> Inputs<2>::s2{u"\u4f60\u4e0d\u662f F18 !\uff1f"};
template <>
FcharData<2> Inputs<2>::s3{
    u"\u4f60\u597d\uff0c\u6211\u66df F18 ! \u4f60\u5462\uff1f"};

template <> FcharData<4> Inputs<4>::s1{U"\u4e4dhy7&3o8%\u4e24"};
template <> FcharData<4> Inputs<4>::s2{U"\u4f60\u4e0d\u662f F18 !\uff1f"};
template <>
FcharData<4> Inputs<4>::s3{
    U"\u4f60\u597d\uff0c\u6211\u66df F18 ! \u4f60\u5462\uff1f"};

// Test simple assignment
extern "C" {
// Declare Fortran subroutine to be tested
//
// SUBROUTINE assignK(s1, s2)
//   CHARACTER(*, K) :: s1, s2
//   s1 = s2
// END SUBROUTINE
void _QPassign1(char *, char *, LenT, LenT);
void _QPassign2(char *, char *, LenT, LenT);
void _QPassign4(char *, char *, LenT, LenT);
}

template <int Kind, typename Func>
void TestNormalAssignment(Func testedSub, int &tests, int &passed) {
  auto &s1{Inputs<Kind>::s1};
  auto &s2{Inputs<Kind>::s2};
  auto &s3{Inputs<Kind>::s3};

  assert(s1.len == s2.len && s1.len < s3.len &&
      "Test requires len(s1) = len(s2) < len(s3)");
  const std::string &desc{"normal character assignment"};

  // s1 = s2 ! len(s1) == len(s3)
  tests++;
  if (TestSubroutine(desc, testedSub, s1, /* expect*/ s2, s2, /*expect*/ s2)) {
    passed++;
  }

  // s1 = s3 ! len(s1) < len(s3)
  FcharData<Kind> s3Tos1{s3.data.substr(0, s1.len)};
  tests++;
  if (TestSubroutine(desc, testedSub, s1, /* expect*/ s3Tos1, s3, s3)) {
    passed++;
  }

  // s3 = s1 ! len(s1) < len(s3)
  using ST = typename CharStorage<Kind>::Type;
  FcharData<Kind> s1Tos3{
      s1.data.substr(0, s1.len) + ST(s3.len - s1.len, /* space */ 0x20)};
  tests++;
  if (TestSubroutine(desc, testedSub, s3, /* expect*/ s1Tos3, s1, s1)) {
    passed++;
  }
}

// Test substring assignment
extern "C" {
// SUBROUTINE assign_substringK(s1, s2, lb, ub)
//   CHARACTER(*, K) :: s1, s2
//   INTEGER :: lb, ub
//   s1(lb:ub) = s2
// END SUBROUTINE
void _QPassign_substring1(char *s1, char *s2, int *lb, int *ub, LenT, LenT);
void _QPassign_substring2(char *, char *, int *, int *, LenT, LenT);
void _QPassign_substring4(char *, char *, int *, int *, LenT, LenT);
}

template <int Kind, typename Func>
void TestSubstringAssignment(Func testedSub, int &tests, int &passed) {
  auto &s1{Inputs<Kind>::s3};
  auto &s2{Inputs<Kind>::s1};
  int lb{3};
  int ub{14};
  assert(1 <= lb && lb < ub && ub <= s1.len && "Failed test requirements");
  const std::string &desc{"substring character assignment"};

  // s1(lb:ub) = s2
  auto delta{ub - lb + 1};
  auto s2CpyLen{s2.len < delta ? s2.len : delta};
  auto str{s1.data.substr(0, lb - 1) + s2.data.substr(0, s2CpyLen)};
  if (auto padding{delta - s2.len}; padding >= 0) {
    using ST = typename CharStorage<Kind>::Type;
    str += ST(padding, /* space */ 0x20);
  }
  FcharData<Kind> expected{str + s1.data.substr(ub, s1.len - ub)};
  tests++;
  if (TestSubroutine(desc, testedSub, s1, expected, s2, s2, &lb, &ub)) {
    passed++;
  }
}

// Test when RHS depends on LHS in a way that require a temp to evaluate RHS
extern "C" {
// SUBROUTINE assign_overlapK(s1, s2, lb)
//   CHARACTER(*, K) :: s1, s2
//   INTEGER :: lb
//   s1(lb:) = s2
// END SUBROUTINE
void _QPassign_overlap1(char *s1, char *s2, int *lb, LenT, LenT);
void _QPassign_overlap2(char *, char *, int *, LenT, LenT);
void _QPassign_overlap4(char *, char *, int *, LenT, LenT);
}

template <int Kind, typename Func>
void TestOverlappingAssignment(Func testedSub, int &tests, int &passed) {
  auto &s1{Inputs<Kind>::s1};
  auto &s2{Inputs<Kind>::s2};
  int lb{2};
  assert(lb >= 2 && "Test requires lb>=2");
  assert(s1.len >= lb && "Test requires len(s1)>=lb");
  const std::string &desc{"overlapping character assignment"};

  // s1(lb:) = s1 ! len(s1) >= lb
  auto delta{lb - 1};
  FcharData<Kind> expected{
      s1.data.substr(0, delta) + s1.data.substr(0, s1.len - delta)};
  tests++;
  if (TestSubroutine(desc, testedSub, s1, expected, s2, s2, &lb)) {
    passed++;
  }
}

// Test assignment of character whose length is specified in specification
// expression.
extern "C" {
// SUBROUTINE assign_spec_expr_lenK(s1, s2, l1, l2)
//   INTEGER :: l1, l2
//   CHARACTER(l1, K) :: s1
//   CHARACTER(l2, K) :: s2
//   s1 = s2
// END SUBROUTINE
void _QPassign_spec_expr_len1(char *s1, char *s2, int *l1, int *l2, LenT, LenT);
void _QPassign_spec_expr_len2(char *s1, char *s2, int *l1, int *l2, LenT, LenT);
void _QPassign_spec_expr_len4(char *s1, char *s2, int *l1, int *l2, LenT, LenT);
}

template <int Kind, typename Func>
void TestSpecExprLenAssignment(Func testedSub, int &tests, int &passed) {
  auto &s1{Inputs<Kind>::s1};
  auto &s2{Inputs<Kind>::s2};
  auto &s3{Inputs<Kind>::s3};

  int l1{static_cast<int>(s1.len / 2)};
  int l2{static_cast<int>(s2.len / 2)};
  int l3{static_cast<int>(s3.len / 2)};
  assert(l1 == l2 && l1 < l3 && "Test requires l1 = l2 < l3");
  const std::string &desc{"assignment of character with specified expr length"};

  // s1 = s2 ! l1 == l3
  tests++;
  FcharData<Kind> expect1{
      s2.data.substr(0, l1) + s1.data.substr(l1, s1.len - l1)};
  if (TestSubroutine(desc, testedSub, s1, expect1, s2, s2, &l1, &l2)) {
    passed++;
  }

  // s1 = s3 ! l1 < l3
  FcharData<Kind> expect2{
      s3.data.substr(0, l1) + s1.data.substr(l1, s1.len - l1)};
  tests++;
  if (TestSubroutine(desc, testedSub, s1, expect2, s3, s3, &l1, &l3)) {
    passed++;
  }

  // s3 = s1 ! l1 < l3
  using ST = typename CharStorage<Kind>::Type;
  FcharData<Kind> expect3{s1.data.substr(0, l1) +
      ST(l3 - l1, /* space */ 0x20) + s3.data.substr(l3, s3.len - l3)};
  tests++;
  if (TestSubroutine(desc, testedSub, s3, expect3, s1, s1, &l3, &l1)) {
    passed++;
  }
}

// Test concatenation
extern "C" {
// SUBROUTINE concat1(s1, s2)
//  CHARACTER(*) :: s1, s2
//  s2 = s1 // " another piece of string"
// END SUBROUTINE
void _QPconcat1(char *s1, char *s2, LenT, LenT);
}

template <int Kind, typename Func>
void TestConcat(Func testedSub, int &tests, int &passed) {
  auto &s1{Inputs<Kind>::s1};
  using ST = typename CharStorage<Kind>::Type;
  ST appended = " another piece of string";
  FcharData<Kind> output{ST(s1.len + appended.length(), ' ')};

  const std::string &desc{"concatenation"};
  FcharData<Kind> expected{s1.data.substr(0, s1.len) + appended};
  tests++;
  if (TestSubroutine(desc, testedSub, s1, s1, output, /* expect*/ expected)) {
    passed++;
  }
}

int main(int, char **) {
  int tests{0}, passed{0};

  TestNormalAssignment<1>(_QPassign1, tests, passed);
  TestNormalAssignment<2>(_QPassign2, tests, passed);
  TestNormalAssignment<4>(_QPassign4, tests, passed);

  TestSubstringAssignment<1>(_QPassign_substring1, tests, passed);
  TestSubstringAssignment<2>(_QPassign_substring2, tests, passed);
  TestSubstringAssignment<4>(_QPassign_substring4, tests, passed);

  TestOverlappingAssignment<1>(_QPassign_overlap1, tests, passed);
  TestOverlappingAssignment<2>(_QPassign_overlap2, tests, passed);
  TestOverlappingAssignment<4>(_QPassign_overlap4, tests, passed);

  TestSpecExprLenAssignment<1>(_QPassign_spec_expr_len1, tests, passed);
  TestSpecExprLenAssignment<2>(_QPassign_spec_expr_len2, tests, passed);
  TestSpecExprLenAssignment<4>(_QPassign_spec_expr_len4, tests, passed);

  TestConcat<1>(_QPconcat1, tests, passed);

  std::cout << passed << " tests passed out of " << tests << std::endl;
  return tests == passed ? 0 : -1;
}
