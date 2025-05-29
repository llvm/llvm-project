//===-- flang/unittests/Common/FastIntSetTest.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Common/enum-class.h"
#include "flang/Support/Fortran-features.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace Fortran::common {

// Not currently exported from Fortran-features.h
llvm::SmallVector<llvm::StringRef> splitCamelCase(llvm::StringRef input);
llvm::SmallVector<llvm::StringRef> splitHyphenated(llvm::StringRef input);
bool equalLowerCaseWithCamelCaseWord(llvm::StringRef l, llvm::StringRef r);

ENUM_CLASS(TestEnumExtra, TwentyOne, FortyTwo, SevenSevenSeven)
ENUM_CLASS_EXTRA(TestEnumExtra)

TEST(EnumClassTest, SplitCamelCase) {

  auto parts = splitCamelCase("oP");
  ASSERT_EQ(parts.size(), (size_t)2);

  if (parts[0].compare(llvm::StringRef("o", 1))) {
    ADD_FAILURE() << "First part is not OP";
  }
  if (parts[1].compare(llvm::StringRef("P", 1))) {
    ADD_FAILURE() << "Second part is not Name";
  }

  parts = splitCamelCase("OPName");
  ASSERT_EQ(parts.size(), (size_t)2);

  if (parts[0].compare(llvm::StringRef("OP", 2))) {
    ADD_FAILURE() << "First part is not OP";
  }
  if (parts[1].compare(llvm::StringRef("Name", 4))) {
    ADD_FAILURE() << "Second part is not Name";
  }

  parts = splitCamelCase("OpName");
  ASSERT_EQ(parts.size(), (size_t)2);
  if (parts[0].compare(llvm::StringRef("Op", 2))) {
    ADD_FAILURE() << "First part is not Op";
  }
  if (parts[1].compare(llvm::StringRef("Name", 4))) {
    ADD_FAILURE() << "Second part is not Name";
  }

  parts = splitCamelCase("opName");
  ASSERT_EQ(parts.size(), (size_t)2);
  if (parts[0].compare(llvm::StringRef("op", 2))) {
    ADD_FAILURE() << "First part is not op";
  }
  if (parts[1].compare(llvm::StringRef("Name", 4))) {
    ADD_FAILURE() << "Second part is not Name";
  }

  parts = splitCamelCase("FlangTestProgram123");
  ASSERT_EQ(parts.size(), (size_t)3);
  if (parts[0].compare(llvm::StringRef("Flang", 5))) {
    ADD_FAILURE() << "First part is not Flang";
  }
  if (parts[1].compare(llvm::StringRef("Test", 4))) {
    ADD_FAILURE() << "Second part is not Test";
  }
  if (parts[2].compare(llvm::StringRef("Program123", 10))) {
    ADD_FAILURE() << "Third part is not Program123";
  }
  for (auto p : parts) {
    llvm::errs() << p << " " << p.size() << "\n";
  }
}

TEST(EnumClassTest, SplitHyphenated) {
  auto parts = splitHyphenated("no-twenty-one");
  ASSERT_EQ(parts.size(), (size_t)3);
  if (parts[0].compare(llvm::StringRef("no", 2))) {
    ADD_FAILURE() << "First part is not twenty";
  }
  if (parts[1].compare(llvm::StringRef("twenty", 6))) {
    ADD_FAILURE() << "Second part is not one";
  }
  if (parts[2].compare(llvm::StringRef("one", 3))) {
    ADD_FAILURE() << "Third part is not one";
  }
  for (auto p : parts) {
    llvm::errs() << p << " " << p.size() << "\n";
  }
}

TEST(EnumClassTest, equalLowerCaseWithCamelCaseWord) {
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("O", "O"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("o", "p"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("o", "P"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("1", "2"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("Op", "op"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("op", "Oplss"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("oplss", "OplSS"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("OPLSS", "oplss"));
  EXPECT_FALSE(equalLowerCaseWithCamelCaseWord("OPLSS", "OPLSS"));

  EXPECT_TRUE(equalLowerCaseWithCamelCaseWord("o", "O"));
  EXPECT_TRUE(equalLowerCaseWithCamelCaseWord("oplss", "OPLSS"));
  EXPECT_TRUE(equalLowerCaseWithCamelCaseWord("oplss", "oplss"));
  EXPECT_TRUE(equalLowerCaseWithCamelCaseWord("op555", "OP555"));
  EXPECT_TRUE(equalLowerCaseWithCamelCaseWord("op555", "op555"));
}

std::optional<std::pair<bool, TestEnumExtra>> parseCLITestEnumExtraOption(
    llvm::StringRef input) {
  return parseCLIEnum<TestEnumExtra>(input, FindTestEnumExtraIndex);
}

TEST(EnumClassTest, parseCLIEnumOption) {
  auto result = parseCLITestEnumExtraOption("no-twenty-one");
  auto expected =
      std::pair<bool, TestEnumExtra>(false, TestEnumExtra::TwentyOne);
  ASSERT_EQ(result, std::optional{expected});
  result = parseCLITestEnumExtraOption("twenty-one");
  expected = std::pair<bool, TestEnumExtra>(true, TestEnumExtra::TwentyOne);
  ASSERT_EQ(result, std::optional{expected});
  result = parseCLITestEnumExtraOption("no-forty-two");
  expected = std::pair<bool, TestEnumExtra>(false, TestEnumExtra::FortyTwo);
  ASSERT_EQ(result, std::optional{expected});
  result = parseCLITestEnumExtraOption("forty-two");
  expected = std::pair<bool, TestEnumExtra>(true, TestEnumExtra::FortyTwo);
  ASSERT_EQ(result, std::optional{expected});
  result = parseCLITestEnumExtraOption("no-seven-seven-seven");
  expected =
      std::pair<bool, TestEnumExtra>(false, TestEnumExtra::SevenSevenSeven);
  ASSERT_EQ(result, std::optional{expected});
  result = parseCLITestEnumExtraOption("seven-seven-seven");
  expected =
      std::pair<bool, TestEnumExtra>(true, TestEnumExtra::SevenSevenSeven);
  ASSERT_EQ(result, std::optional{expected});
}

} // namespace Fortran::common
