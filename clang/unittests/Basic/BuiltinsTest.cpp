//===- unittests/Basic/BuiltinsTest.cpp - Builtin::Context tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "gtest/gtest.h"
#include <thread>
#include <vector>

using namespace clang;

namespace {

class BuiltinsTest : public ::testing::Test {
protected:
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags{DiagnosticIDs::create(), DiagOpts,
                          new IgnoringDiagConsumer()};
  LangOptions LangOpts;
  TargetOptions TargetOpts, AuxTargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target, AuxTarget;
  Builtin::Context Builtins;
  std::unique_ptr<IdentifierTable> Idents;

  void initializeTarget(StringRef Triple, StringRef AuxTriple = "") {
    TargetOpts.Triple = Triple.str();
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
    ASSERT_TRUE(Target);
    if (!AuxTriple.empty()) {
      AuxTargetOpts.Triple = AuxTriple.str();
      AuxTarget = TargetInfo::CreateTargetInfo(Diags, AuxTargetOpts);
      ASSERT_TRUE(AuxTarget);
    }
    Builtins = Builtin::Context(*Target, AuxTarget.get());
    Idents = std::make_unique<IdentifierTable>(LangOpts);
  }

  void initialize(StringRef Triple, StringRef AuxTriple = "") {
    initializeTarget(Triple, AuxTriple);
    Idents->setBuiltinInfo(Builtins, LangOpts);
  }

  unsigned getBuiltinID(StringRef Name) {
    return Idents->get(Name).getBuiltinID();
  }

  bool contains(StringRef Name) const {
    return Idents->find(Name) != Idents->end();
  }

  unsigned getNumBuiltins() const {
    unsigned Num = Builtin::FirstTSBuiltin;
    for (const TargetInfo *T : {Target.get(), AuxTarget.get()})
      if (T)
        for (const Builtin::InfosShard &Shard : T->getTargetBuiltins())
          Num += Shard.Infos.size();
    return Num;
  }
};

TEST_F(BuiltinsTest, IdentifiersAreCreatedOnDemand) {
  initialize("arm64-apple-macosx");
  // Registering every builtin upfront would create ~12,000 identifiers.
  EXPECT_LT(Idents->size(), 2000u);
  EXPECT_FALSE(contains("__builtin_abs"));
  EXPECT_FALSE(contains("__builtin_neon_vaddq_v"));

  EXPECT_EQ(getBuiltinID("__builtin_abs"), Builtin::BI__builtin_abs);
  EXPECT_EQ(Idents->getOwn("__builtin_expect").getBuiltinID(),
            Builtin::BI__builtin_expect);
  EXPECT_EQ(getBuiltinID("not_a_builtin"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("__builtin_"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("__builtin_neon_"), Builtin::NotBuiltin);
}

TEST_F(BuiltinsTest, ExistingIdentifiersAreResolved) {
  initializeTarget("x86_64-linux-gnu");
  EXPECT_EQ(getBuiltinID("__builtin_abs"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("__builtin_ia32_pause"), Builtin::NotBuiltin);

  Idents->setBuiltinInfo(Builtins, LangOpts);
  EXPECT_EQ(getBuiltinID("__builtin_abs"), Builtin::BI__builtin_abs);
  unsigned ID = getBuiltinID("__builtin_ia32_pause");
  EXPECT_TRUE(Builtins.isTSBuiltin(ID));
  EXPECT_EQ(Builtins.getName(ID), "__builtin_ia32_pause");
}

TEST_F(BuiltinsTest, TargetBuiltins) {
  initialize("arm64-apple-macosx");
  // Builtins in a shard without and with a name prefix.
  for (StringRef Name : {"__builtin_arm_rbit", "__builtin_neon_vaddq_v",
                         "__builtin_sve_svadd_s8_x"}) {
    unsigned ID = getBuiltinID(Name);
    EXPECT_TRUE(Builtins.isTSBuiltin(ID)) << Name;
    EXPECT_FALSE(Builtins.isAuxBuiltinID(ID)) << Name;
    EXPECT_EQ(Builtins.getName(ID), Name);
  }
  EXPECT_EQ(getBuiltinID("__builtin_ia32_pause"), Builtin::NotBuiltin);
}

TEST_F(BuiltinsTest, AuxTargetBuiltins) {
  LangOpts.CUDA = true;
  LangOpts.CPlusPlus = true;
  initialize("nvptx64-nvidia-cuda", "x86_64-linux-gnu");
  unsigned ID = getBuiltinID("__builtin_ia32_pause");
  EXPECT_TRUE(Builtins.isAuxBuiltinID(ID));
  EXPECT_EQ(Builtins.getName(ID), "__builtin_ia32_pause");

  ID = getBuiltinID("__nvvm_read_ptx_sreg_tid_x");
  EXPECT_TRUE(Builtins.isTSBuiltin(ID));
  EXPECT_FALSE(Builtins.isAuxBuiltinID(ID));
}

TEST_F(BuiltinsTest, UnsupportedBuiltins) {
  initialize("x86_64-linux-gnu");
  EXPECT_EQ(getBuiltinID("__noop"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("objc_msgSend"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("move"), Builtin::NotBuiltin);
}

TEST_F(BuiltinsTest, LanguageSpecificBuiltins) {
  LangOpts.MicrosoftExt = true;
  LangOpts.ObjC = true;
  LangOpts.CPlusPlus = true;
  initialize("x86_64-linux-gnu");
  EXPECT_EQ(getBuiltinID("__noop"), Builtin::BI__noop);
  EXPECT_EQ(getBuiltinID("objc_msgSend"), Builtin::BIobjc_msgSend);
  EXPECT_EQ(getBuiltinID("move"), Builtin::BImove);
}

TEST_F(BuiltinsTest, NoBuiltinFuncs) {
  LangOpts.NoBuiltinFuncs = {"malloc", "std-forward", "move"};
  LangOpts.CPlusPlus = true;
  initialize("x86_64-linux-gnu");
  EXPECT_EQ(getBuiltinID("malloc"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("__builtin_malloc"), Builtin::BI__builtin_malloc);
  EXPECT_EQ(getBuiltinID("free"), Builtin::BIfree);
  // -fno-builtin-std-foo only applies to builtins in namespace std, and
  // -fno-builtin-foo only to those outside of it.
  EXPECT_EQ(getBuiltinID("forward"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("move"), Builtin::BImove);
}

TEST_F(BuiltinsTest, NoBuiltin) {
  LangOpts.NoBuiltin = true;
  initialize("x86_64-linux-gnu");
  EXPECT_EQ(getBuiltinID("malloc"), Builtin::NotBuiltin);
  EXPECT_EQ(getBuiltinID("__builtin_malloc"), Builtin::BI__builtin_malloc);
}

TEST_F(BuiltinsTest, AuxTargetBuiltinsWinOverTargetBuiltins) {
  // __mulh is a builtin of both x86-64 and AArch64.
  LangOpts.MicrosoftExt = true;
  initialize("x86_64-windows-msvc", "aarch64-windows-msvc");
  unsigned ID = getBuiltinID("__mulh");
  EXPECT_TRUE(Builtins.isAuxBuiltinID(ID));
  EXPECT_EQ(Builtins.getName(ID), "__mulh");
}

TEST_F(BuiltinsTest, ConcurrentContexts) {
  // Contexts for different targets on several threads at once, each building
  // its own index.
  std::vector<std::thread> Threads;
  std::vector<unsigned> IDs(8);
  for (unsigned I = 0; I != IDs.size(); ++I) {
    Threads.emplace_back([&, I] {
      DiagnosticOptions DiagOpts;
      DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                              new IgnoringDiagConsumer());
      TargetOptions Opts;
      Opts.Triple = I % 2 ? "riscv32-unknown-elf" : "mipsel-linux-gnu";
      IntrusiveRefCntPtr<TargetInfo> T =
          TargetInfo::CreateTargetInfo(Diags, Opts);
      LangOptions LO;
      Builtin::Context Ctx(*T, nullptr);
      IdentifierTable Table(LO);
      Table.setBuiltinInfo(Ctx, LO);
      IDs[I] = Table.get("__builtin_abs").getBuiltinID();
    });
  }
  for (std::thread &Thread : Threads)
    Thread.join();
  for (unsigned ID : IDs)
    EXPECT_EQ(ID, Builtin::BI__builtin_abs);
}

TEST_F(BuiltinsTest, SharedTargetBuiltins) {
  initializeTarget("arm64-apple-macosx");
  // Contexts with different language options can share the builtins, which
  // stay valid as long as any context uses them.
  auto Shared =
      std::make_shared<const Builtin::TargetBuiltins>(*Target, nullptr);
  auto MSContext = std::make_unique<Builtin::Context>(Shared);
  Builtin::Context Ctx(std::move(Shared));
  LangOptions MS;
  MS.MicrosoftExt = true;
  EXPECT_EQ(MSContext->lookupBuiltinID("__noop", MS), Builtin::BI__noop);
  MSContext.reset();
  EXPECT_EQ(Ctx.lookupBuiltinID("__noop", LangOpts), Builtin::NotBuiltin);
  EXPECT_EQ(Ctx.lookupBuiltinID("__builtin_abs", LangOpts),
            Builtin::BI__builtin_abs);
  EXPECT_TRUE(
      Ctx.isTSBuiltin(Ctx.lookupBuiltinID("__builtin_neon_vaddq_v", LangOpts)));
}

TEST_F(BuiltinsTest, ContextWithoutTarget) {
  Builtin::Context Ctx;
  EXPECT_EQ(Ctx.lookupBuiltinID("__builtin_abs", LangOpts),
            Builtin::NotBuiltin);
  EXPECT_EQ(Ctx.getName(Builtin::BI__builtin_abs), "__builtin_abs");
  unsigned NumNames = 0;
  Ctx.forEachBuiltinName(LangOpts, [&](StringRef) { ++NumNames; });
  EXPECT_EQ(NumNames, 0u);
}

TEST_F(BuiltinsTest, TargetBuiltinsLookup) {
  // __mulh is a builtin of both x86-64 and AArch64.
  initializeTarget("x86_64-windows-msvc", "aarch64-windows-msvc");
  Builtin::TargetBuiltins Index(*Target, AuxTarget.get());
  EXPECT_EQ(Index.lookup("__builtin_abs"),
            SmallVector<unsigned>({Builtin::BI__builtin_abs}));
  SmallVector<unsigned, 2> Mulh = Index.lookup("__mulh");
  ASSERT_EQ(Mulh.size(), 2u);
  EXPECT_NE(Builtins.isAuxBuiltinID(Mulh[0]), Builtins.isAuxBuiltinID(Mulh[1]));
  for (unsigned ID : Mulh)
    EXPECT_EQ(Builtins.getName(ID), "__mulh");
  EXPECT_TRUE(Index.lookup("not_a_builtin").empty());
  EXPECT_TRUE(Index.lookup("").empty());
  EXPECT_TRUE(Index.lookup("__builtin_neon_").empty());
}

struct BuiltinsConfig {
  const char *Triple;
  const char *AuxTriple;
  void (*Configure)(LangOptions &);
};

class BuiltinsConsistencyTest
    : public BuiltinsTest,
      public ::testing::WithParamInterface<BuiltinsConfig> {};

/// The lookup through the name indices agrees with the enumeration of all
/// supported builtins, which checks every builtin directly: every enumerated
/// name resolves to a builtin with that name, and no other builtin name
/// resolves.
TEST_P(BuiltinsConsistencyTest, LookupMatchesEnumeration) {
  GetParam().Configure(LangOpts);
  initialize(GetParam().Triple, GetParam().AuxTriple);
  llvm::StringSet<> Supported;
  Builtins.forEachBuiltinName(LangOpts,
                              [&](StringRef Name) { Supported.insert(Name); });
  ASSERT_FALSE(Supported.empty());

  llvm::StringMap<unsigned> LastAuxID;
  for (unsigned ID = 1, E = getNumBuiltins(); ID != E; ++ID)
    if (Builtins.isAuxBuiltinID(ID))
      LastAuxID[Builtins.getName(ID)] = ID;

  unsigned NumResolved = 0;
  for (unsigned ID = 1, E = getNumBuiltins(); ID != E; ++ID) {
    std::string Name = Builtins.getName(ID);
    unsigned Resolved = getBuiltinID(Name);
    if (Supported.contains(Name)) {
      ASSERT_NE(Resolved, Builtin::NotBuiltin) << Name;
      EXPECT_EQ(Builtins.getName(Resolved), Name);
      ++NumResolved;
    } else {
      EXPECT_EQ(Resolved, Builtin::NotBuiltin) << Name;
    }
    if (auto It = LastAuxID.find(Name); It != LastAuxID.end())
      EXPECT_EQ(Resolved, It->second) << Name;
  }
  EXPECT_GT(NumResolved, 1000u);
}

static void c17(LangOptions &) {}
static void cxx20(LangOptions &LO) {
  LO.CPlusPlus = LO.CPlusPlus11 = LO.CPlusPlus14 = LO.CPlusPlus17 =
      LO.CPlusPlus20 = true;
}
static void objcMS(LangOptions &LO) {
  LO.ObjC = LO.Blocks = LO.MicrosoftExt = LO.GNUMode = true;
}
static void noBuiltin(LangOptions &LO) {
  LO.NoBuiltin = LO.NoMathBuiltin = true;
}
static void cuda(LangOptions &LO) {
  cxx20(LO);
  LO.CUDA = true;
}
static void openCL(LangOptions &LO) {
  LO.OpenCL = true;
  LO.OpenCLVersion = 200;
}

INSTANTIATE_TEST_SUITE_P(
    Targets, BuiltinsConsistencyTest,
    ::testing::Values(
        BuiltinsConfig{"arm64-apple-macosx", "", c17},
        BuiltinsConfig{"arm64-apple-macosx", "", cxx20},
        BuiltinsConfig{"arm64-apple-macosx", "", objcMS},
        BuiltinsConfig{"arm64-apple-macosx", "", noBuiltin},
        BuiltinsConfig{"x86_64-windows-msvc", "aarch64-windows-msvc", objcMS},
        BuiltinsConfig{"armv7-linux-gnueabihf", "", c17},
        BuiltinsConfig{"riscv64-linux-gnu", "", c17},
        BuiltinsConfig{"ppc64le-linux-gnu", "", cxx20},
        BuiltinsConfig{"mips64-linux-gnu", "", c17},
        BuiltinsConfig{"wasm32", "", c17},
        BuiltinsConfig{"loongarch64-linux-gnu", "", c17},
        BuiltinsConfig{"hexagon-unknown-elf", "", c17},
        BuiltinsConfig{"systemz-linux-gnu", "", c17},
        BuiltinsConfig{"amdgcn-amd-amdhsa", "x86_64-linux-gnu", cuda},
        BuiltinsConfig{"nvptx64-nvidia-cuda", "aarch64-linux-gnu", cuda},
        BuiltinsConfig{"x86_64-linux-gnu", "nvptx64-nvidia-cuda", cuda},
        BuiltinsConfig{"spir64", "", openCL}));

TEST_F(BuiltinsTest, ForEachBuiltinName) {
  initialize("arm64-apple-macosx");
  llvm::StringSet<> Names;
  Builtins.forEachBuiltinName(LangOpts,
                              [&](StringRef Name) { Names.insert(Name); });
  EXPECT_TRUE(Names.contains("__builtin_abs"));
  EXPECT_TRUE(Names.contains("__builtin_neon_vaddq_v"));
  EXPECT_TRUE(Names.contains("__builtin_arm_rbit"));
  EXPECT_FALSE(Names.contains("__noop"));
  EXPECT_FALSE(Names.contains("__builtin_ia32_pause"));
  EXPECT_FALSE(Names.contains(""));
  // Enumerating names doesn't create identifiers.
  EXPECT_FALSE(contains("__builtin_abs"));
}

} // namespace
