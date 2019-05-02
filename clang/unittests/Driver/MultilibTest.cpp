//===- unittests/Driver/MultilibTest.cpp --- Multilib tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Multilib and MultilibSet
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Multilib.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "gtest/gtest.h"

using namespace clang::driver;
using namespace clang;

TEST(MultilibTest, MultilibValidity) {

  ASSERT_TRUE(Multilib().isValid()) << "Empty multilib is not valid";

  ASSERT_TRUE(Multilib().flag("+foo").isValid())
      << "Single indicative flag is not valid";

  ASSERT_TRUE(Multilib().flag("-foo").isValid())
      << "Single contraindicative flag is not valid";

  ASSERT_FALSE(Multilib().flag("+foo").flag("-foo").isValid())
      << "Conflicting flags should invalidate the Multilib";

  ASSERT_TRUE(Multilib().flag("+foo").flag("+foo").isValid())
      << "Multilib should be valid even if it has the same flag twice";

  ASSERT_TRUE(Multilib().flag("+foo").flag("-foobar").isValid())
      << "Seemingly conflicting prefixes shouldn't actually conflict";
}

TEST(MultilibTest, OpEqReflexivity1) {
  Multilib M;
  ASSERT_TRUE(M == M) << "Multilib::operator==() is not reflexive";
}

TEST(MultilibTest, OpEqReflexivity2) {
  ASSERT_TRUE(Multilib() == Multilib())
      << "Separately constructed default multilibs are not equal";
}

TEST(MultilibTest, OpEqReflexivity3) {
  Multilib M1, M2;
  M1.flag("+foo");
  M2.flag("+foo");
  ASSERT_TRUE(M1 == M2) << "Multilibs with the same flag should be the same";
}

TEST(MultilibTest, OpEqInequivalence1) {
  Multilib M1, M2;
  M1.flag("+foo");
  M2.flag("-foo");
  ASSERT_FALSE(M1 == M2) << "Multilibs with conflicting flags are not the same";
  ASSERT_FALSE(M2 == M1)
      << "Multilibs with conflicting flags are not the same (commuted)";
}

TEST(MultilibTest, OpEqInequivalence2) {
  Multilib M1, M2;
  M2.flag("+foo");
  ASSERT_FALSE(M1 == M2) << "Flags make Multilibs different";
}

TEST(MultilibTest, OpEqEquivalence1) {
  Multilib M1, M2;
  M1.flag("+foo");
  M2.flag("+foo").flag("+foo");
  ASSERT_TRUE(M1 == M2) << "Flag duplication shouldn't affect equivalence";
  ASSERT_TRUE(M2 == M1)
      << "Flag duplication shouldn't affect equivalence (commuted)";
}

TEST(MultilibTest, OpEqEquivalence2) {
  Multilib M1("64");
  Multilib M2;
  M2.gccSuffix("/64");
  ASSERT_TRUE(M1 == M2)
      << "Constructor argument must match Multilib::gccSuffix()";
  ASSERT_TRUE(M2 == M1)
      << "Constructor argument must match Multilib::gccSuffix() (commuted)";
}

TEST(MultilibTest, OpEqEquivalence3) {
  Multilib M1("", "32");
  Multilib M2;
  M2.osSuffix("/32");
  ASSERT_TRUE(M1 == M2)
      << "Constructor argument must match Multilib::osSuffix()";
  ASSERT_TRUE(M2 == M1)
      << "Constructor argument must match Multilib::osSuffix() (commuted)";
}

TEST(MultilibTest, OpEqEquivalence4) {
  Multilib M1("", "", "16");
  Multilib M2;
  M2.includeSuffix("/16");
  ASSERT_TRUE(M1 == M2)
      << "Constructor argument must match Multilib::includeSuffix()";
  ASSERT_TRUE(M2 == M1)
      << "Constructor argument must match Multilib::includeSuffix() (commuted)";
}

TEST(MultilibTest, OpEqInequivalence3) {
  Multilib M1("foo");
  Multilib M2("bar");
  ASSERT_FALSE(M1 == M2) << "Differing gccSuffixes should be different";
  ASSERT_FALSE(M2 == M1)
      << "Differing gccSuffixes should be different (commuted)";
}

TEST(MultilibTest, OpEqInequivalence4) {
  Multilib M1("", "foo");
  Multilib M2("", "bar");
  ASSERT_FALSE(M1 == M2) << "Differing osSuffixes should be different";
  ASSERT_FALSE(M2 == M1)
      << "Differing osSuffixes should be different (commuted)";
}

TEST(MultilibTest, OpEqInequivalence5) {
  Multilib M1("", "", "foo");
  Multilib M2("", "", "bar");
  ASSERT_FALSE(M1 == M2) << "Differing includeSuffixes should be different";
  ASSERT_FALSE(M2 == M1)
      << "Differing includeSuffixes should be different (commuted)";
}

TEST(MultilibTest, Construction1) {
  Multilib M("gcc64", "os64", "inc64");
  ASSERT_TRUE(M.gccSuffix() == "/gcc64");
  ASSERT_TRUE(M.osSuffix() == "/os64");
  ASSERT_TRUE(M.includeSuffix() == "/inc64");
}

TEST(MultilibTest, Construction2) {
  Multilib M1;
  Multilib M2("");
  Multilib M3("", "");
  Multilib M4("", "", "");
  ASSERT_TRUE(M1 == M2)
      << "Default arguments to Multilib constructor broken (first argument)";
  ASSERT_TRUE(M1 == M3)
      << "Default arguments to Multilib constructor broken (second argument)";
  ASSERT_TRUE(M1 == M4)
      << "Default arguments to Multilib constructor broken (third argument)";
}

TEST(MultilibTest, Construction3) {
  Multilib M = Multilib().flag("+f1").flag("+f2").flag("-f3");
  for (Multilib::flags_list::const_iterator I = M.flags().begin(),
                                            E = M.flags().end();
       I != E; ++I) {
    ASSERT_TRUE(llvm::StringSwitch<bool>(*I)
                    .Cases("+f1", "+f2", "-f3", true)
                    .Default(false));
  }
}

static bool hasFlag(const Multilib &M, StringRef Flag) {
  for (Multilib::flags_list::const_iterator I = M.flags().begin(),
                                            E = M.flags().end();
       I != E; ++I) {
    if (*I == Flag)
      return true;
    else if (StringRef(*I).substr(1) == Flag.substr(1))
      return false;
  }
  return false;
}

TEST(MultilibTest, SetConstruction1) {
  // Single maybe
  MultilibSet MS;
  ASSERT_TRUE(MS.size() == 0);
  MS.Maybe(Multilib("64").flag("+m64"));
  ASSERT_TRUE(MS.size() == 2);
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    if (I->gccSuffix() == "/64")
      ASSERT_TRUE(I->flags()[0] == "+m64");
    else if (I->gccSuffix() == "")
      ASSERT_TRUE(I->flags()[0] == "-m64");
    else
      FAIL() << "Unrecognized gccSufix: " << I->gccSuffix();
  }
}

TEST(MultilibTest, SetConstruction2) {
  // Double maybe
  MultilibSet MS;
  MS.Maybe(Multilib("sof").flag("+sof"));
  MS.Maybe(Multilib("el").flag("+EL"));
  ASSERT_TRUE(MS.size() == 4);
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_TRUE(I->isValid()) << "Multilb " << *I << " should be valid";
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Cases("", "/sof", "/el", "/sof/el", true)
                    .Default(false))
        << "Multilib " << *I << " wasn't expected";
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Case("", hasFlag(*I, "-sof"))
                    .Case("/sof", hasFlag(*I, "+sof"))
                    .Case("/el", hasFlag(*I, "-sof"))
                    .Case("/sof/el", hasFlag(*I, "+sof"))
                    .Default(false))
        << "Multilib " << *I << " didn't have the appropriate {+,-}sof flag";
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Case("", hasFlag(*I, "-EL"))
                    .Case("/sof", hasFlag(*I, "-EL"))
                    .Case("/el", hasFlag(*I, "+EL"))
                    .Case("/sof/el", hasFlag(*I, "+EL"))
                    .Default(false))
        << "Multilib " << *I << " didn't have the appropriate {+,-}EL flag";
  }
}

TEST(MultilibTest, SetPushback) {
  MultilibSet MS;
  MS.push_back(Multilib("one"));
  MS.push_back(Multilib("two"));
  ASSERT_TRUE(MS.size() == 2);
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Cases("/one", "/two", true)
                    .Default(false));
  }
  MS.clear();
  ASSERT_TRUE(MS.size() == 0);
}

TEST(MultilibTest, SetRegexFilter) {
  MultilibSet MS;
  MS.Maybe(Multilib("one"));
  MS.Maybe(Multilib("two"));
  MS.Maybe(Multilib("three"));
  ASSERT_EQ(MS.size(), (unsigned)2 * 2 * 2)
      << "Size before filter was incorrect. Contents:\n" << MS;
  MS.FilterOut("/one/two/three");
  ASSERT_EQ(MS.size(), (unsigned)2 * 2 * 2 - 1)
      << "Size after filter was incorrect. Contents:\n" << MS;
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_TRUE(I->gccSuffix() != "/one/two/three")
        << "The filter should have removed " << *I;
  }
}

TEST(MultilibTest, SetFilterObject) {
  MultilibSet MS;
  MS.Maybe(Multilib("orange"));
  MS.Maybe(Multilib("pear"));
  MS.Maybe(Multilib("plum"));
  ASSERT_EQ((int)MS.size(), 1 /* Default */ +
                            1 /* pear */ +
                            1 /* plum */ +
                            1 /* pear/plum */ +
                            1 /* orange */ +
                            1 /* orange/pear */ +
                            1 /* orange/plum */ +
                            1 /* orange/pear/plum */ )
      << "Size before filter was incorrect. Contents:\n" << MS;
  MS.FilterOut([](const Multilib &M) {
    return StringRef(M.gccSuffix()).startswith("/p");
  });
  ASSERT_EQ((int)MS.size(), 1 /* Default */ +
                            1 /* orange */ +
                            1 /* orange/pear */ +
                            1 /* orange/plum */ + 
                            1 /* orange/pear/plum */ )
      << "Size after filter was incorrect. Contents:\n" << MS;
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_FALSE(StringRef(I->gccSuffix()).startswith("/p"))
        << "The filter should have removed " << *I;
  }
}

TEST(MultilibTest, SetSelection1) {
  MultilibSet MS1 = MultilibSet()
    .Maybe(Multilib("64").flag("+m64"));

  Multilib::flags_list FlagM64;
  FlagM64.push_back("+m64");
  Multilib SelectionM64;
  ASSERT_TRUE(MS1.select(FlagM64, SelectionM64))
      << "Flag set was {\"+m64\"}, but selection not found";
  ASSERT_TRUE(SelectionM64.gccSuffix() == "/64")
      << "Selection picked " << SelectionM64 << " which was not expected";

  Multilib::flags_list FlagNoM64;
  FlagNoM64.push_back("-m64");
  Multilib SelectionNoM64;
  ASSERT_TRUE(MS1.select(FlagNoM64, SelectionNoM64))
      << "Flag set was {\"-m64\"}, but selection not found";
  ASSERT_TRUE(SelectionNoM64.gccSuffix() == "")
      << "Selection picked " << SelectionNoM64 << " which was not expected";
}

TEST(MultilibTest, SetSelection2) {
  MultilibSet MS2 = MultilibSet()
    .Maybe(Multilib("el").flag("+EL"))
    .Maybe(Multilib("sf").flag("+SF"));

  for (unsigned I = 0; I < 4; ++I) {
    bool IsEL = I & 0x1;
    bool IsSF = I & 0x2;
    Multilib::flags_list Flags;
    if (IsEL)
      Flags.push_back("+EL");
    else
      Flags.push_back("-EL");

    if (IsSF)
      Flags.push_back("+SF");
    else
      Flags.push_back("-SF");

    Multilib Selection;
    ASSERT_TRUE(MS2.select(Flags, Selection)) << "Selection failed for "
                                              << (IsEL ? "+EL" : "-EL") << " "
                                              << (IsSF ? "+SF" : "-SF");

    std::string Suffix;
    if (IsEL)
      Suffix += "/el";
    if (IsSF)
      Suffix += "/sf";

    ASSERT_EQ(Selection.gccSuffix(), Suffix) << "Selection picked " << Selection
                                             << " which was not expected ";
  }
}

TEST(MultilibTest, SetCombineWith) {
  MultilibSet Coffee;
  Coffee.push_back(Multilib("coffee"));
  MultilibSet Milk;
  Milk.push_back(Multilib("milk"));
  MultilibSet Latte;
  ASSERT_EQ(Latte.size(), (unsigned)0);
  Latte.combineWith(Coffee);
  ASSERT_EQ(Latte.size(), (unsigned)1);
  Latte.combineWith(Milk);
  ASSERT_EQ(Latte.size(), (unsigned)2);
}

TEST(MultilibTest, SetPriority) {
  MultilibSet MS;
  MS.push_back(Multilib("foo", {}, {}, 1).flag("+foo"));
  MS.push_back(Multilib("bar", {}, {}, 2).flag("+bar"));

  Multilib::flags_list Flags1;
  Flags1.push_back("+foo");
  Flags1.push_back("-bar");
  Multilib Selection1;
  ASSERT_TRUE(MS.select(Flags1, Selection1))
      << "Flag set was {\"+foo\"}, but selection not found";
  ASSERT_TRUE(Selection1.gccSuffix() == "/foo")
      << "Selection picked " << Selection1 << " which was not expected";

  Multilib::flags_list Flags2;
  Flags2.push_back("+foo");
  Flags2.push_back("+bar");
  Multilib Selection2;
  ASSERT_TRUE(MS.select(Flags2, Selection2))
      << "Flag set was {\"+bar\"}, but selection not found";
  ASSERT_TRUE(Selection2.gccSuffix() == "/bar")
      << "Selection picked " << Selection2 << " which was not expected";
}
