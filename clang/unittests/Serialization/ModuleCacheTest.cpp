//===- unittests/Serialization/ModuleCacheTest.cpp - CI tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class ModuleCacheTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("modulecache-test", TestDir));

    ModuleCachePath = SmallString<256>(TestDir);
    sys::path::append(ModuleCachePath, "mcp");
    ASSERT_FALSE(sys::fs::create_directories(ModuleCachePath));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

public:
  SmallString<256> TestDir;
  SmallString<256> ModuleCachePath;

  void addFile(StringRef Path, StringRef Contents) {
    ASSERT_FALSE(sys::path::is_absolute(Path));

    SmallString<256> AbsPath(TestDir);
    sys::path::append(AbsPath, Path);

    std::error_code EC;
    ASSERT_FALSE(
        sys::fs::create_directories(llvm::sys::path::parent_path(AbsPath)));
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  void addDuplicateFrameworks() {
    addFile("test.m", R"cpp(
        @import Top;
    )cpp");

    addFile("frameworks/Top.framework/Headers/top.h", R"cpp(
        @import M;
    )cpp");
    addFile("frameworks/Top.framework/Modules/module.modulemap", R"cpp(
        framework module Top [system] {
          header "top.h"
          export *
        }
    )cpp");

    addFile("frameworks/M.framework/Headers/m.h", R"cpp(
        void foo();
    )cpp");
    addFile("frameworks/M.framework/Modules/module.modulemap", R"cpp(
        framework module M [system] {
          header "m.h"
          export *
        }
    )cpp");

    addFile("frameworks2/M.framework/Headers/m.h", R"cpp(
        void foo();
    )cpp");
    addFile("frameworks2/M.framework/Modules/module.modulemap", R"cpp(
        framework module M [system] {
          header "m.h"
          export *
        }
    )cpp");
  }
};

TEST_F(ModuleCacheTest, CachedModuleNewPath) {
  addDuplicateFrameworks();

  SmallString<256> MCPArg("-fmodules-cache-path=");
  MCPArg.append(ModuleCachePath);
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());
  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;

  // First run should pass with no errors
  const char *Args[] = {"clang",        "-fmodules",          "-Fframeworks",
                        MCPArg.c_str(), "-working-directory", TestDir.c_str(),
                        "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  CompilerInstance Instance;
  Instance.setDiagnostics(Diags.get());
  Instance.setInvocation(Invocation);
  SyntaxOnlyAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  // Now add `frameworks2` to the search path. `Top.pcm` will have a reference
  // to the `M` from `frameworks`, but a search will find the `M` from
  // `frameworks2` - causing a mismatch and it to be considered out of date.
  //
  // Normally this would be fine - `M` and the modules it depends on would be
  // rebuilt. However, since we have a shared module cache and thus an already
  // finalized `Top`, recompiling `Top` will cause the existing module to be
  // removed from the cache, causing possible crashed if it is ever used.
  //
  // Make sure that an error occurs instead.
  const char *Args2[] = {"clang",         "-fmodules",    "-Fframeworks2",
                         "-Fframeworks",  MCPArg.c_str(), "-working-directory",
                         TestDir.c_str(), "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation2 =
      createInvocation(Args2, CIOpts);
  ASSERT_TRUE(Invocation2);
  CompilerInstance Instance2(Instance.getPCHContainerOperations(),
                             &Instance.getModuleCache());
  Instance2.setDiagnostics(Diags.get());
  Instance2.setInvocation(Invocation2);
  SyntaxOnlyAction Action2;
  ASSERT_FALSE(Instance2.ExecuteAction(Action2));
  ASSERT_TRUE(Diags->hasErrorOccurred());
}

TEST_F(ModuleCacheTest, CachedModuleNewPathAllowErrors) {
  addDuplicateFrameworks();

  SmallString<256> MCPArg("-fmodules-cache-path=");
  MCPArg.append(ModuleCachePath);
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());
  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;

  // First run should pass with no errors
  const char *Args[] = {"clang",        "-fmodules",          "-Fframeworks",
                        MCPArg.c_str(), "-working-directory", TestDir.c_str(),
                        "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  CompilerInstance Instance;
  Instance.setDiagnostics(Diags.get());
  Instance.setInvocation(Invocation);
  SyntaxOnlyAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  // Same as `CachedModuleNewPath` but while allowing errors. This is a hard
  // failure where the module wasn't created, so it should still fail.
  const char *Args2[] = {
      "clang",         "-fmodules",    "-Fframeworks2",
      "-Fframeworks",  MCPArg.c_str(), "-working-directory",
      TestDir.c_str(), "-Xclang",      "-fallow-pcm-with-compiler-errors",
      "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation2 =
      createInvocation(Args2, CIOpts);
  ASSERT_TRUE(Invocation2);
  CompilerInstance Instance2(Instance.getPCHContainerOperations(),
                             &Instance.getModuleCache());
  Instance2.setDiagnostics(Diags.get());
  Instance2.setInvocation(Invocation2);
  SyntaxOnlyAction Action2;
  ASSERT_FALSE(Instance2.ExecuteAction(Action2));
  ASSERT_TRUE(Diags->hasErrorOccurred());
}

} // anonymous namespace
