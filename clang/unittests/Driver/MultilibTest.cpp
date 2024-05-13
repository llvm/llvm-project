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
#include "clang/Basic/Version.h"
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
  Multilib M1({}, {}, {}, {"+foo"});
  Multilib M2({}, {}, {}, {"+foo"});
  ASSERT_TRUE(M1 == M2) << "Multilibs with the same flag should be the same";
}

TEST(MultilibTest, OpEqInequivalence1) {
  Multilib M1({}, {}, {}, {"+foo"});
  Multilib M2({}, {}, {}, {"-foo"});
  ASSERT_FALSE(M1 == M2) << "Multilibs with conflicting flags are not the same";
  ASSERT_FALSE(M2 == M1)
      << "Multilibs with conflicting flags are not the same (commuted)";
}

TEST(MultilibTest, OpEqInequivalence2) {
  Multilib M1;
  Multilib M2({}, {}, {}, {"+foo"});
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
  Multilib M({}, {}, {}, {"+f1", "+f2", "-f3"});
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
      Multilib("/foo", {}, {}, {"+foo"}),
      Multilib("/bar", {}, {}, {"+bar"}),
  });
  Multilib::flags_list Flags1 = {"+foo", "-bar"};
  llvm::SmallVector<Multilib> Selection1;
  ASSERT_TRUE(MS.select(Flags1, Selection1))
      << "Flag set was {\"+foo\"}, but selection not found";
  ASSERT_TRUE(Selection1.back().gccSuffix() == "/foo")
      << "Selection picked " << Selection1.back() << " which was not expected";

  Multilib::flags_list Flags2 = {"+foo", "+bar"};
  llvm::SmallVector<Multilib> Selection2;
  ASSERT_TRUE(MS.select(Flags2, Selection2))
      << "Flag set was {\"+bar\"}, but selection not found";
  ASSERT_TRUE(Selection2.back().gccSuffix() == "/bar")
      << "Selection picked " << Selection2.back() << " which was not expected";
}

TEST(MultilibTest, SelectMultiple) {
  MultilibSet MS({
      Multilib("/a", {}, {}, {"x"}),
      Multilib("/b", {}, {}, {"y"}),
  });
  llvm::SmallVector<Multilib> Selection;

  ASSERT_TRUE(MS.select({"x"}, Selection));
  ASSERT_EQ(1u, Selection.size());
  EXPECT_EQ("/a", Selection[0].gccSuffix());

  ASSERT_TRUE(MS.select({"y"}, Selection));
  ASSERT_EQ(1u, Selection.size());
  EXPECT_EQ("/b", Selection[0].gccSuffix());

  ASSERT_TRUE(MS.select({"y", "x"}, Selection));
  ASSERT_EQ(2u, Selection.size());
  EXPECT_EQ("/a", Selection[0].gccSuffix());
  EXPECT_EQ("/b", Selection[1].gccSuffix());
}

static void diagnosticCallback(const llvm::SMDiagnostic &D, void *Out) {
  *reinterpret_cast<std::string *>(Out) = D.getMessage();
}

static bool parseYaml(MultilibSet &MS, std::string &Diagnostic,
                      const char *Data) {
  auto ErrorOrMS = MultilibSet::parseYaml(llvm::MemoryBufferRef(Data, "TEST"),
                                          diagnosticCallback, &Diagnostic);
  if (ErrorOrMS.getError())
    return false;
  MS = std::move(ErrorOrMS.get());
  return true;
}

static bool parseYaml(MultilibSet &MS, const char *Data) {
  auto ErrorOrMS = MultilibSet::parseYaml(llvm::MemoryBufferRef(Data, "TEST"));
  if (ErrorOrMS.getError())
    return false;
  MS = std::move(ErrorOrMS.get());
  return true;
}

// When updating this version also update MultilibVersionCurrent in Multilib.cpp
#define YAML_PREAMBLE "MultilibVersion: 1.0\n"

TEST(MultilibTest, ParseInvalid) {
  std::string Diagnostic;

  MultilibSet MS;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, R"(
Variants: []
)"));
  EXPECT_TRUE(
      StringRef(Diagnostic).contains("missing required key 'MultilibVersion'"))
      << Diagnostic;

  // Reject files with a different major version
  EXPECT_FALSE(parseYaml(MS, Diagnostic,
                         R"(
MultilibVersion: 2.0
Variants: []
)"));
  EXPECT_TRUE(
      StringRef(Diagnostic).contains("multilib version 2.0 is unsupported"))
      << Diagnostic;
  EXPECT_FALSE(parseYaml(MS, Diagnostic,
                         R"(
MultilibVersion: 0.1
Variants: []
)"));
  EXPECT_TRUE(
      StringRef(Diagnostic).contains("multilib version 0.1 is unsupported"))
      << Diagnostic;

  // Reject files with a later minor version
  EXPECT_FALSE(parseYaml(MS, Diagnostic,
                         R"(
MultilibVersion: 1.9
Variants: []
)"));
  EXPECT_TRUE(
      StringRef(Diagnostic).contains("multilib version 1.9 is unsupported"))
      << Diagnostic;

  // Accept files with the same major version and the same or earlier minor
  // version
  EXPECT_TRUE(parseYaml(MS, Diagnostic, R"(
MultilibVersion: 1.0
Variants: []
)")) << Diagnostic;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, YAML_PREAMBLE));
  EXPECT_TRUE(StringRef(Diagnostic).contains("missing required key 'Variants'"))
      << Diagnostic;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, YAML_PREAMBLE R"(
Variants:
- Dir: /abc
  Flags: []
)"));
  EXPECT_TRUE(StringRef(Diagnostic).contains("paths must be relative"))
      << Diagnostic;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, YAML_PREAMBLE R"(
Variants:
- Flags: []
)"));
  EXPECT_TRUE(StringRef(Diagnostic).contains("missing required key 'Dir'"))
      << Diagnostic;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, YAML_PREAMBLE R"(
Variants:
- Dir: .
)"));
  EXPECT_TRUE(StringRef(Diagnostic).contains("missing required key 'Flags'"))
      << Diagnostic;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, YAML_PREAMBLE R"(
Variants: []
Mappings:
- Match: abc
)"));
  EXPECT_TRUE(StringRef(Diagnostic).contains("value required for 'Flags'"))
      << Diagnostic;

  EXPECT_FALSE(parseYaml(MS, Diagnostic, YAML_PREAMBLE R"(
Variants: []
Mappings:
- Dir: .
  Match: '('
  Flags: []
)"));
  EXPECT_TRUE(StringRef(Diagnostic).contains("parentheses not balanced"))
      << Diagnostic;
}

TEST(MultilibTest, Parse) {
  MultilibSet MS;
  EXPECT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: .
  Flags: []
)"));
  EXPECT_EQ(1U, MS.size());
  EXPECT_EQ("", MS.begin()->gccSuffix());

  EXPECT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: abc
  Flags: []
)"));
  EXPECT_EQ(1U, MS.size());
  EXPECT_EQ("/abc", MS.begin()->gccSuffix());

  EXPECT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: pqr
  Flags: [-mfloat-abi=soft]
)"));
  EXPECT_EQ(1U, MS.size());
  EXPECT_EQ("/pqr", MS.begin()->gccSuffix());
  EXPECT_EQ(std::vector<std::string>({"-mfloat-abi=soft"}),
            MS.begin()->flags());

  EXPECT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: pqr
  Flags: [-mfloat-abi=soft, -fno-exceptions]
)"));
  EXPECT_EQ(1U, MS.size());
  EXPECT_EQ(std::vector<std::string>({"-mfloat-abi=soft", "-fno-exceptions"}),
            MS.begin()->flags());

  EXPECT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: a
  Flags: []
- Dir: b
  Flags: []
)"));
  EXPECT_EQ(2U, MS.size());
}

TEST(MultilibTest, SelectSoft) {
  MultilibSet MS;
  llvm::SmallVector<Multilib> Selected;
  ASSERT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: s
  Flags: [-mfloat-abi=soft]
Mappings:
- Match: -mfloat-abi=softfp
  Flags: [-mfloat-abi=soft]
)"));
  EXPECT_TRUE(MS.select({"-mfloat-abi=soft"}, Selected));
  EXPECT_TRUE(MS.select({"-mfloat-abi=softfp"}, Selected));
  EXPECT_FALSE(MS.select({"-mfloat-abi=hard"}, Selected));
}

TEST(MultilibTest, SelectSoftFP) {
  MultilibSet MS;
  llvm::SmallVector<Multilib> Selected;
  ASSERT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: f
  Flags: [-mfloat-abi=softfp]
)"));
  EXPECT_FALSE(MS.select({"-mfloat-abi=soft"}, Selected));
  EXPECT_TRUE(MS.select({"-mfloat-abi=softfp"}, Selected));
  EXPECT_FALSE(MS.select({"-mfloat-abi=hard"}, Selected));
}

TEST(MultilibTest, SelectHard) {
  // If hard float is all that's available then select that only if compiling
  // with hard float.
  MultilibSet MS;
  llvm::SmallVector<Multilib> Selected;
  ASSERT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: h
  Flags: [-mfloat-abi=hard]
)"));
  EXPECT_FALSE(MS.select({"-mfloat-abi=soft"}, Selected));
  EXPECT_FALSE(MS.select({"-mfloat-abi=softfp"}, Selected));
  EXPECT_TRUE(MS.select({"-mfloat-abi=hard"}, Selected));
}

TEST(MultilibTest, SelectFloatABI) {
  MultilibSet MS;
  llvm::SmallVector<Multilib> Selected;
  ASSERT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: s
  Flags: [-mfloat-abi=soft]
- Dir: f
  Flags: [-mfloat-abi=softfp]
- Dir: h
  Flags: [-mfloat-abi=hard]
Mappings:
- Match: -mfloat-abi=softfp
  Flags: [-mfloat-abi=soft]
)"));
  MS.select({"-mfloat-abi=soft"}, Selected);
  EXPECT_EQ("/s", Selected.back().gccSuffix());
  MS.select({"-mfloat-abi=softfp"}, Selected);
  EXPECT_EQ("/f", Selected.back().gccSuffix());
  MS.select({"-mfloat-abi=hard"}, Selected);
  EXPECT_EQ("/h", Selected.back().gccSuffix());
}

TEST(MultilibTest, SelectFloatABIReversed) {
  // If soft is specified after softfp then softfp will never be
  // selected because soft is compatible with softfp and last wins.
  MultilibSet MS;
  llvm::SmallVector<Multilib> Selected;
  ASSERT_TRUE(parseYaml(MS, YAML_PREAMBLE R"(
Variants:
- Dir: h
  Flags: [-mfloat-abi=hard]
- Dir: f
  Flags: [-mfloat-abi=softfp]
- Dir: s
  Flags: [-mfloat-abi=soft]
Mappings:
- Match: -mfloat-abi=softfp
  Flags: [-mfloat-abi=soft]
)"));
  MS.select({"-mfloat-abi=soft"}, Selected);
  EXPECT_EQ("/s", Selected.back().gccSuffix());
  MS.select({"-mfloat-abi=softfp"}, Selected);
  EXPECT_EQ("/s", Selected.back().gccSuffix());
  MS.select({"-mfloat-abi=hard"}, Selected);
  EXPECT_EQ("/h", Selected.back().gccSuffix());
}

TEST(MultilibTest, SelectMClass) {
  const char *MultilibSpec = YAML_PREAMBLE R"(
Variants:
- Dir: thumb/v6-m/nofp
  Flags: [--target=thumbv6m-none-unknown-eabi, -mfpu=none]

- Dir: thumb/v7-m/nofp
  Flags: [--target=thumbv7m-none-unknown-eabi, -mfpu=none]

- Dir: thumb/v7e-m/nofp
  Flags: [--target=thumbv7em-none-unknown-eabi, -mfpu=none]

- Dir: thumb/v8-m.main/nofp
  Flags: [--target=thumbv8m.main-none-unknown-eabi, -mfpu=none]

- Dir: thumb/v8.1-m.main/nofp/nomve
  Flags: [--target=thumbv8.1m.main-none-unknown-eabi, -mfpu=none]

- Dir: thumb/v7e-m/fpv4_sp_d16
  Flags: [--target=thumbv7em-none-unknown-eabihf, -mfpu=fpv4-sp-d16]

- Dir: thumb/v7e-m/fpv5_d16
  Flags: [--target=thumbv7em-none-unknown-eabihf, -mfpu=fpv5-d16]

- Dir: thumb/v8-m.main/fp
  Flags: [--target=thumbv8m.main-none-unknown-eabihf]

- Dir: thumb/v8.1-m.main/fp
  Flags: [--target=thumbv8.1m.main-none-unknown-eabihf]

- Dir: thumb/v8.1-m.main/nofp/mve
  Flags: [--target=thumbv8.1m.main-none-unknown-eabihf, -march=thumbv8.1m.main+mve]

Mappings:
- Match: --target=thumbv8(\.[0-9]+)?m\.base-none-unknown-eabi
  Flags: [--target=thumbv6m-none-unknown-eabi]
- Match: -target=thumbv8\.[1-9]m\.main-none-unknown-eabi
  Flags: [--target=thumbv8.1m.main-none-unknown-eabi]
- Match: -target=thumbv8\.[1-9]m\.main-none-unknown-eabihf
  Flags: [--target=thumbv8.1m.main-none-unknown-eabihf]
- Match: -march=thumbv8\.[1-9]m\.main.*\+mve($|\+).*
  Flags: [-march=thumbv8.1m.main+mve]
)";

  MultilibSet MS;
  llvm::SmallVector<Multilib> Selected;
  ASSERT_TRUE(parseYaml(MS, MultilibSpec));

  ASSERT_TRUE(MS.select({"--target=thumbv6m-none-unknown-eabi", "-mfpu=none"},
                        Selected));
  EXPECT_EQ("/thumb/v6-m/nofp", Selected.back().gccSuffix());

  ASSERT_TRUE(MS.select({"--target=thumbv7m-none-unknown-eabi", "-mfpu=none"},
                        Selected));
  EXPECT_EQ("/thumb/v7-m/nofp", Selected.back().gccSuffix());

  ASSERT_TRUE(MS.select({"--target=thumbv7em-none-unknown-eabi", "-mfpu=none"},
                        Selected));
  EXPECT_EQ("/thumb/v7e-m/nofp", Selected.back().gccSuffix());

  ASSERT_TRUE(MS.select(
      {"--target=thumbv8m.main-none-unknown-eabi", "-mfpu=none"}, Selected));
  EXPECT_EQ("/thumb/v8-m.main/nofp", Selected.back().gccSuffix());

  ASSERT_TRUE(MS.select(
      {"--target=thumbv8.1m.main-none-unknown-eabi", "-mfpu=none"}, Selected));
  EXPECT_EQ("/thumb/v8.1-m.main/nofp/nomve", Selected.back().gccSuffix());

  ASSERT_TRUE(
      MS.select({"--target=thumbv7em-none-unknown-eabihf", "-mfpu=fpv4-sp-d16"},
                Selected));
  EXPECT_EQ("/thumb/v7e-m/fpv4_sp_d16", Selected.back().gccSuffix());

  ASSERT_TRUE(MS.select(
      {"--target=thumbv7em-none-unknown-eabihf", "-mfpu=fpv5-d16"}, Selected));
  EXPECT_EQ("/thumb/v7e-m/fpv5_d16", Selected.back().gccSuffix());

  ASSERT_TRUE(
      MS.select({"--target=thumbv8m.main-none-unknown-eabihf"}, Selected));
  EXPECT_EQ("/thumb/v8-m.main/fp", Selected.back().gccSuffix());

  ASSERT_TRUE(
      MS.select({"--target=thumbv8.1m.main-none-unknown-eabihf"}, Selected));
  EXPECT_EQ("/thumb/v8.1-m.main/fp", Selected.back().gccSuffix());

  ASSERT_TRUE(MS.select({"--target=thumbv8.1m.main-none-unknown-eabihf",
                         "-mfpu=none", "-march=thumbv8.1m.main+dsp+mve"},
                        Selected));
  EXPECT_EQ("/thumb/v8.1-m.main/nofp/mve", Selected.back().gccSuffix());
}
