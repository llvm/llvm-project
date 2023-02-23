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
#include "../../lib/Driver/ToolChains/CommonArgs.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace clang::driver;
using namespace clang;

TEST(MultilibTest, OpEqReflexivity1) {
  Multilib M;
  ASSERT_TRUE(M == M) << "Multilib::operator==() is not reflexive";
}

TEST(MultilibTest, OpEqReflexivity2) {
  ASSERT_TRUE(Multilib() == Multilib())
      << "Separately constructed default multilibs are not equal";
}

TEST(MultilibTest, OpEqReflexivity3) {
  Multilib M1({}, {}, {}, 0, {"+foo"});
  Multilib M2({}, {}, {}, 0, {"+foo"});
  ASSERT_TRUE(M1 == M2) << "Multilibs with the same flag should be the same";
}

TEST(MultilibTest, OpEqInequivalence1) {
  Multilib M1({}, {}, {}, 0, {"+foo"});
  Multilib M2({}, {}, {}, 0, {"-foo"});
  ASSERT_FALSE(M1 == M2) << "Multilibs with conflicting flags are not the same";
  ASSERT_FALSE(M2 == M1)
      << "Multilibs with conflicting flags are not the same (commuted)";
}

TEST(MultilibTest, OpEqInequivalence2) {
  Multilib M1;
  Multilib M2({}, {}, {}, 0, {"+foo"});
  ASSERT_FALSE(M1 == M2) << "Flags make Multilibs different";
}

TEST(MultilibTest, OpEqEquivalence2) {
  Multilib M1("/64");
  Multilib M2("/64");
  ASSERT_TRUE(M1 == M2)
      << "Constructor argument must match Multilib::gccSuffix()";
  ASSERT_TRUE(M2 == M1)
      << "Constructor argument must match Multilib::gccSuffix() (commuted)";
}

TEST(MultilibTest, OpEqEquivalence3) {
  Multilib M1("", "/32");
  Multilib M2("", "/32");
  ASSERT_TRUE(M1 == M2)
      << "Constructor argument must match Multilib::osSuffix()";
  ASSERT_TRUE(M2 == M1)
      << "Constructor argument must match Multilib::osSuffix() (commuted)";
}

TEST(MultilibTest, OpEqEquivalence4) {
  Multilib M1("", "", "/16");
  Multilib M2("", "", "/16");
  ASSERT_TRUE(M1 == M2)
      << "Constructor argument must match Multilib::includeSuffix()";
  ASSERT_TRUE(M2 == M1)
      << "Constructor argument must match Multilib::includeSuffix() (commuted)";
}

TEST(MultilibTest, OpEqInequivalence3) {
  Multilib M1("/foo");
  Multilib M2("/bar");
  ASSERT_FALSE(M1 == M2) << "Differing gccSuffixes should be different";
  ASSERT_FALSE(M2 == M1)
      << "Differing gccSuffixes should be different (commuted)";
}

TEST(MultilibTest, OpEqInequivalence4) {
  Multilib M1("", "/foo");
  Multilib M2("", "/bar");
  ASSERT_FALSE(M1 == M2) << "Differing osSuffixes should be different";
  ASSERT_FALSE(M2 == M1)
      << "Differing osSuffixes should be different (commuted)";
}

TEST(MultilibTest, OpEqInequivalence5) {
  Multilib M1("", "", "/foo");
  Multilib M2("", "", "/bar");
  ASSERT_FALSE(M1 == M2) << "Differing includeSuffixes should be different";
  ASSERT_FALSE(M2 == M1)
      << "Differing includeSuffixes should be different (commuted)";
}

TEST(MultilibTest, Construction1) {
  Multilib M("/gcc64", "/os64", "/inc64");
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
  Multilib M({}, {}, {}, 0, {"+f1", "+f2", "-f3"});
  for (Multilib::flags_list::const_iterator I = M.flags().begin(),
                                            E = M.flags().end();
       I != E; ++I) {
    ASSERT_TRUE(llvm::StringSwitch<bool>(*I)
                    .Cases("+f1", "+f2", "-f3", true)
                    .Default(false));
  }
}

TEST(MultilibTest, SetPushback) {
  MultilibSet MS({
      Multilib("/one"),
      Multilib("/two"),
  });
  ASSERT_TRUE(MS.size() == 2);
  for (MultilibSet::const_iterator I = MS.begin(), E = MS.end(); I != E; ++I) {
    ASSERT_TRUE(llvm::StringSwitch<bool>(I->gccSuffix())
                    .Cases("/one", "/two", true)
                    .Default(false));
  }
}

TEST(MultilibTest, SetPriority) {
  MultilibSet MS({
      Multilib("/foo", {}, {}, 1, {"+foo"}),
      Multilib("/bar", {}, {}, 2, {"+bar"}),
  });
  Multilib::flags_list Flags1 = {"+foo", "-bar"};
  Multilib Selection1;
  ASSERT_TRUE(MS.select(Flags1, Selection1))
      << "Flag set was {\"+foo\"}, but selection not found";
  ASSERT_TRUE(Selection1.gccSuffix() == "/foo")
      << "Selection picked " << Selection1 << " which was not expected";

  Multilib::flags_list Flags2 = {"+foo", "+bar"};
  Multilib Selection2;
  ASSERT_TRUE(MS.select(Flags2, Selection2))
      << "Flag set was {\"+bar\"}, but selection not found";
  ASSERT_TRUE(Selection2.gccSuffix() == "/bar")
      << "Selection picked " << Selection2 << " which was not expected";
}
