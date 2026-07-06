//===--- FeatureModulesRegistryTests.cpp  ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FeatureModule.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::ElementsAre;

namespace llvm {
raw_ostream &operator<<(raw_ostream &OS,
                        const clang::clangd::FeatureModuleRegistry::entry &E) {
  OS << "(name = " << E.getName() << ", description = '" << E.getDesc() << "')";
  return OS;
}

raw_ostream &operator<<(
    raw_ostream &OS,
    const iterator_range<Registry<clang::clangd::FeatureModule>::iterator>
        &Rng) {
  OS << "{ ";
  bool First = true;
  for (clang::clangd::FeatureModuleRegistry::entry E : Rng) {
    if (First)
      First = false;
    else
      OS << ", ";
    OS << E;
  }
  OS << " }";
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const clang::clangd::Tweak &T) {
  OS << "(id = " << T.id() << ", "
     << "title = " << T.title() << ")";
  return OS;
}
} // namespace llvm

namespace clang::clangd {
namespace {

class Dummy final : public FeatureModule {
  static constexpr const char *TweakID = "DummyTweak";
  struct DummyTweak final : public Tweak {
    const char *id() const override { return TweakID; }
    bool prepare(const Selection &) override { return true; }
    Expected<Effect> apply(const Selection &) override {
      return error("not implemented");
    }
    std::string title() const override { return id(); }
    llvm::StringLiteral kind() const override {
      return llvm::StringLiteral("");
    };
  };

  void contributeTweaks(std::vector<std::unique_ptr<Tweak>> &Out) override {
    Out.emplace_back(new DummyTweak);
  }
};

static FeatureModuleRegistry::Add<Dummy>
    X("dummy", "Dummy feature module with dummy tweak");

MATCHER_P(moduleName, Name, "") { return arg.getName() == Name; }
MATCHER_P(tweakID, ID, "") { return arg->id() == llvm::StringRef(ID); }

// In this test, it is assumed that for unittests executable, all feature
// modules are added to the registry only here (in this file). To implement
// modules for clangd tool, one need to link them directly to the clangd
// executable in clangd/tool/CMakeLists.txt.
TEST(FeatureModulesRegistryTest, DummyModule) {
  EXPECT_THAT(FeatureModuleRegistry::entries(),
              ElementsAre(moduleName("dummy")));
  FeatureModuleSet Set = FeatureModuleSet::fromRegistry();
  ASSERT_EQ(Set.end() - Set.begin(), 1u);
  std::vector<std::unique_ptr<Tweak>> Tweaks;
  Set.begin()->contributeTweaks(Tweaks);
  EXPECT_THAT(Tweaks, ElementsAre(tweakID("DummyTweak")));
}

} // namespace
} // namespace clang::clangd
