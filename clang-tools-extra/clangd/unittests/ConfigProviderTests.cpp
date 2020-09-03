//===-- ConfigProviderTests.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "ConfigProvider.h"
#include "ConfigTesting.h"
#include "TestFS.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>

namespace clang {
namespace clangd {
namespace config {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Provider that appends an arg to compile flags.
// The arg is prefix<N>, where N is the times getFragments() was called.
// It also yields a diagnostic each time it's called.
class FakeProvider : public Provider {
  std::string Prefix;
  mutable std::atomic<unsigned> Index = {0};

  std::vector<CompiledFragment>
  getFragments(const Params &, DiagnosticCallback DC) const override {
    DC(llvm::SMDiagnostic("", llvm::SourceMgr::DK_Error, Prefix));
    CompiledFragment F =
        [Arg(Prefix + std::to_string(++Index))](const Params &P, Config &C) {
          C.CompileFlags.Edits.push_back(
              [Arg](std::vector<std::string> &Argv) { Argv.push_back(Arg); });
          return true;
        };
    return {F};
  }

public:
  FakeProvider(llvm::StringRef Prefix) : Prefix(Prefix) {}
};

std::vector<std::string> getAddedArgs(Config &C) {
  std::vector<std::string> Argv;
  for (auto &Edit : C.CompileFlags.Edits)
    Edit(Argv);
  return Argv;
}

// The provider from combine() should invoke its providers in order, and not
// cache their results.
TEST(ProviderTest, Combine) {
  CapturedDiags Diags;
  FakeProvider Foo("foo");
  FakeProvider Bar("bar");
  auto Combined = Provider::combine({&Foo, &Bar});
  Config Cfg = Combined->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(DiagMessage("foo"), DiagMessage("bar")));
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo1", "bar1"));
  Diags.Diagnostics.clear();

  Cfg = Combined->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(DiagMessage("foo"), DiagMessage("bar")));
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo2", "bar2"));
}

const char *AddFooWithErr = R"yaml(
CompileFlags:
  Add: foo
  Unknown: 42
)yaml";

const char *AddBarBaz = R"yaml(
CompileFlags:
  Add: bar
---
CompileFlags:
  Add: baz
)yaml";

TEST(ProviderTest, FromYAMLFile) {
  MockFS FS;
  FS.Files["foo.yaml"] = AddFooWithErr;

  CapturedDiags Diags;
  auto P = Provider::fromYAMLFile(testPath("foo.yaml"), FS);
  auto Cfg = P->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(DiagMessage("Unknown CompileFlags key Unknown")));
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo"));
  Diags.Diagnostics.clear();

  Cfg = P->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty()) << "Cached, not re-parsed";
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo"));

  FS.Files["foo.yaml"] = AddBarBaz;
  Cfg = P->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty()) << "New config, no errors";
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("bar", "baz"));

  FS.Files.erase("foo.yaml");
  Cfg = P->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty()) << "Missing file is not an error";
  EXPECT_THAT(getAddedArgs(Cfg), IsEmpty());
}

TEST(ProviderTest, FromAncestorRelativeYAMLFiles) {
  MockFS FS;
  FS.Files["a/b/c/foo.yaml"] = AddBarBaz;
  FS.Files["a/foo.yaml"] = AddFooWithErr;

  std::string ABCPath =
      testPath("a/b/c/d/test.cc", llvm::sys::path::Style::posix);
  Params ABCParams;
  ABCParams.Path = ABCPath;
  std::string APath =
      testPath("a/b/e/f/test.cc", llvm::sys::path::Style::posix);
  Params AParams;
  AParams.Path = APath;

  CapturedDiags Diags;
  auto P = Provider::fromAncestorRelativeYAMLFiles("foo.yaml", FS);

  auto Cfg = P->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(getAddedArgs(Cfg), IsEmpty());

  Cfg = P->getConfig(ABCParams, Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(DiagMessage("Unknown CompileFlags key Unknown")));
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo", "bar", "baz"));
  Diags.Diagnostics.clear();

  Cfg = P->getConfig(AParams, Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty()) << "Cached config";
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo"));

  FS.Files.erase("a/foo.yaml");
  Cfg = P->getConfig(ABCParams, Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("bar", "baz"));
}

TEST(ProviderTest, Staleness) {
  MockFS FS;

  auto StartTime = std::chrono::steady_clock::now();
  Params StaleOK;
  StaleOK.FreshTime = StartTime;
  Params MustBeFresh;
  MustBeFresh.FreshTime = StartTime + std::chrono::hours(1);
  CapturedDiags Diags;
  auto P = Provider::fromYAMLFile(testPath("foo.yaml"), FS);

  // Initial query always reads, regardless of policy.
  FS.Files["foo.yaml"] = AddFooWithErr;
  auto Cfg = P->getConfig(StaleOK, Diags.callback());
  EXPECT_THAT(Diags.Diagnostics,
              ElementsAre(DiagMessage("Unknown CompileFlags key Unknown")));
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo"));
  Diags.Diagnostics.clear();

  // Stale value reused by policy.
  FS.Files["foo.yaml"] = AddBarBaz;
  Cfg = P->getConfig(StaleOK, Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty()) << "Cached, not re-parsed";
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("foo"));

  // Cache revalidated by policy.
  Cfg = P->getConfig(MustBeFresh, Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty()) << "New config, no errors";
  EXPECT_THAT(getAddedArgs(Cfg), ElementsAre("bar", "baz"));

  // Cache revalidated by (default) policy.
  FS.Files.erase("foo.yaml");
  Cfg = P->getConfig(Params(), Diags.callback());
  EXPECT_THAT(Diags.Diagnostics, IsEmpty());
  EXPECT_THAT(getAddedArgs(Cfg), IsEmpty());
}

} // namespace
} // namespace config
} // namespace clangd
} // namespace clang
