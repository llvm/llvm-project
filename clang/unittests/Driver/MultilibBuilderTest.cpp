//===- unittests/Driver/MultilibBuilderTest.cpp --- MultilibBuilder tests
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for MultilibBuilder and MultilibSetBuilder
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/MultilibBuilder.h"
#include "../../lib/Driver/ToolChains/CommonArgs.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "gtest/gtest.h"

using llvm::is_contained;
using namespace clang;
using namespace driver;

TEST(MultilibBuilderTest, MultilibValidity) {

  ASSERT_TRUE(MultilibBuilder().isValid()) << "Empty multilib is not valid";

  ASSERT_TRUE(MultilibBuilder().flag("-foo").isValid())
      << "Single indicative flag is not valid";

  ASSERT_TRUE(MultilibBuilder().flag("-foo", /*Disallow=*/true).isValid())
      << "Single contraindicative flag is not valid";

  ASSERT_FALSE(
      MultilibBuilder().flag("-foo").flag("-foo", /*Disallow=*/true).isValid())
      << "Conflicting flags should invalidate the Multilib";

  ASSERT_TRUE(MultilibBuilder().flag("-foo").flag("-foo").isValid())
      << "Multilib should be valid even if it has the same flag "
         "twice";

  ASSERT_TRUE(MultilibBuilder()
                  .flag("-foo")
                  .flag("-foobar", /*Disallow=*/true)
                  .isValid())
      << "Seemingly conflicting prefixes shouldn't actually conflict";
}

TEST(MultilibBuilderTest, Construction1) {
  MultilibBuilder M("gcc64", "os64", "inc64");
  ASSERT_TRUE(M.gccSuffix() == "/gcc64");
  ASSERT_TRUE(M.osSuffix() == "/os64");
  ASSERT_TRUE(M.includeSuffix() == "/inc64");
}

TEST(MultilibBuilderTest, Construction3) {
  MultilibBuilder M =
      MultilibBuilder().flag("-f1").flag("-f2").flag("-f3", /*Disallow=*/true);
  for (const std::string &A : M.flags()) {
    ASSERT_TRUE(llvm::StringSwitch<bool>(A)
                    .Cases("-f1", "-f2", "!f3", true)
                    .Default(false));
  }
}

TEST(MultilibBuilderTest, SetConstruction1) {
  // Single maybe
  MultilibSet MS = MultilibSetBuilder()
                       .Maybe(MultilibBuilder("64").flag("-m64"))
                       .makeMultilibSet();
  ASSERT_TRUE(MS.size() == 2);
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    if (I->gccSuffix() == "/64")
      ASSERT_TRUE(*I->flags().begin() == "-m64");
    else if (I->gccSuffix() == "")
      ASSERT_TRUE(*I->flags().begin() == "!m64");
    else
      FAIL() << "Unrecognized gccSufix: " << I->gccSuffix();
  }
}

TEST(MultilibBuilderTest, SetConstruction2) {
  // Double maybe
  MultilibSet MS = MultilibSetBuilder()
                       .Maybe(MultilibBuilder("sof").flag("-sof"))
                       .Maybe(MultilibBuilder("el").flag("-EL"))
                       .makeMultilibSet();
  ASSERT_TRUE(MS.size() == 4);
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Cases("", "/sof", "/el", "/sof/el", true)
                    .Default(false))
        << "Multilib " << *I << " wasn't expected";
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Case("", is_contained(I->flags(), "!sof"))
                    .Case("/sof", is_contained(I->flags(), "-sof"))
                    .Case("/el", is_contained(I->flags(), "!sof"))
                    .Case("/sof/el", is_contained(I->flags(), "-sof"))
                    .Default(false))
        << "Multilib " << *I << " didn't have the appropriate {-,!}sof flag";
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Case("", is_contained(I->flags(), "!EL"))
                    .Case("/sof", is_contained(I->flags(), "!EL"))
                    .Case("/el", is_contained(I->flags(), "-EL"))
                    .Case("/sof/el", is_contained(I->flags(), "-EL"))
                    .Default(false))
        << "Multilib " << *I << " didn't have the appropriate {-,!}EL flag";
  }
}

TEST(MultilibBuilderTest, SetRegexFilter) {
  MultilibSetBuilder MB;
  MB.Maybe(MultilibBuilder("one"))
      .Maybe(MultilibBuilder("two"))
      .Maybe(MultilibBuilder("three"))
      .makeMultilibSet();
  MultilibSet MS = MB.makeMultilibSet();
  ASSERT_EQ(MS.size(), (unsigned)2 * 2 * 2)
      << "Size before filter was incorrect. Contents:\n"
      << MS;
  MB.FilterOut("/one/two/three");
  MS = MB.makeMultilibSet();
  ASSERT_EQ(MS.size(), (unsigned)2 * 2 * 2 - 1)
      << "Size after filter was incorrect. Contents:\n"
      << MS;
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_TRUE(I->gccSuffix() != "/one/two/three")
        << "The filter should have removed " << *I;
  }
}

TEST(MultilibBuilderTest, SetFilterObject) {
  MultilibSet MS = MultilibSetBuilder()
                       .Maybe(MultilibBuilder("orange"))
                       .Maybe(MultilibBuilder("pear"))
                       .Maybe(MultilibBuilder("plum"))
                       .makeMultilibSet();
  ASSERT_EQ((int)MS.size(), 1 /* Default */ + 1 /* pear */ + 1 /* plum */ +
                                1 /* pear/plum */ + 1 /* orange */ +
                                1 /* orange/pear */ + 1 /* orange/plum */ +
                                1 /* orange/pear/plum */)
      << "Size before filter was incorrect. Contents:\n"
      << MS;
  MS.FilterOut([](const Multilib &M) {
    return StringRef(M.gccSuffix()).starts_with("/p");
  });
  ASSERT_EQ((int)MS.size(), 1 /* Default */ + 1 /* orange */ +
                                1 /* orange/pear */ + 1 /* orange/plum */ +
                                1 /* orange/pear/plum */)
      << "Size after filter was incorrect. Contents:\n"
      << MS;
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_FALSE(StringRef(I->gccSuffix()).starts_with("/p"))
        << "The filter should have removed " << *I;
  }
}

TEST(MultilibBuilderTest, SetSelection1) {
  MultilibSet MS1 = MultilibSetBuilder()
                        .Maybe(MultilibBuilder("64").flag("-m64"))
                        .makeMultilibSet();

  Multilib::flags_list FlagM64 = {"-m64"};
  llvm::SmallVector<Multilib> SelectionM64;
  ASSERT_TRUE(MS1.select(FlagM64, SelectionM64))
      << "Flag set was {\"-m64\"}, but selection not found";
  ASSERT_TRUE(SelectionM64.back().gccSuffix() == "/64")
      << "Selection picked " << SelectionM64.back()
      << " which was not expected";

  Multilib::flags_list FlagNoM64 = {"!m64"};
  llvm::SmallVector<Multilib> SelectionNoM64;
  ASSERT_TRUE(MS1.select(FlagNoM64, SelectionNoM64))
      << "Flag set was {\"!m64\"}, but selection not found";
  ASSERT_TRUE(SelectionNoM64.back().gccSuffix() == "")
      << "Selection picked " << SelectionNoM64.back()
      << " which was not expected";
}

TEST(MultilibBuilderTest, SetSelection2) {
  MultilibSet MS2 = MultilibSetBuilder()
                        .Maybe(MultilibBuilder("el").flag("-EL"))
                        .Maybe(MultilibBuilder("sf").flag("-SF"))
                        .makeMultilibSet();

  for (unsigned I = 0; I < 4; ++I) {
    bool IsEL = I & 0x1;
    bool IsSF = I & 0x2;
    Multilib::flags_list Flags;
    if (IsEL)
      Flags.push_back("-EL");
    else
      Flags.push_back("!EL");

    if (IsSF)
      Flags.push_back("-SF");
    else
      Flags.push_back("!SF");

    llvm::SmallVector<Multilib> Selection;
    ASSERT_TRUE(MS2.select(Flags, Selection))
        << "Selection failed for " << (IsEL ? "-EL" : "!EL") << " "
        << (IsSF ? "-SF" : "!SF");

    std::string Suffix;
    if (IsEL)
      Suffix += "/el";
    if (IsSF)
      Suffix += "/sf";

    ASSERT_EQ(Selection.back().gccSuffix(), Suffix)
        << "Selection picked " << Selection.back()
        << " which was not expected ";
  }
}
