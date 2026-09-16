//===- unittests/Lex/ModuleMapTest.cpp - PPCallbacks tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//
//

#include "clang/Lex/ModuleMap.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gtest/gtest.h"

#include <vector>

namespace clang {
namespace {

struct InterceptorFS : llvm::vfs::ProxyFileSystem {
  // Record the paths looked up by a ModuleMap.
  std::vector<std::string> StatPaths;

  InterceptorFS(IntrusiveRefCntPtr<llvm::vfs::FileSystem> UnderlyingFS)
      : ProxyFileSystem(UnderlyingFS) {}

  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
    StatPaths.emplace_back(Path.str());
    return ProxyFileSystem::status(Path);
  }
};

class ModuleMapTest : public ::testing::Test {
protected:
  ModuleMapTest()
      : InMemFS(llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>()),
        VFS(llvm::makeIntrusiveRefCnt<InterceptorFS>(InMemFS)),
        FileMgr(FileMgrOpts, VFS),
        Diagnostics(DiagnosticIDs::create(), DiagnosticOpts,
                    new IgnoringDiagConsumer()),
        SrcMgr(Diagnostics, FileMgr), TargetOpts(new TargetOptions),
        HdrSearch(HdrSearchOpts, SrcMgr, Diagnostics, LangOpts,
                  /* Target = */ nullptr),
        Map(SrcMgr, Diagnostics, LangOpts, /* Target= */ nullptr, HdrSearch) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diagnostics, *TargetOpts);
    Map.setTarget(*Target);
  }

  void addFile(llvm::StringRef Path, llvm::StringRef Content) {
    InMemFS->addFile(Path, 0, llvm::MemoryBuffer::getMemBufferCopy(Content));
  }

  bool loadRoot(llvm::StringRef Path) {
    auto File = FileMgr.getOptionalFileRef(Path);
    if (!File) {
      // parseAndLoadModuleMapFile returns false on success, true on error.
      return true;
    }
    return Map.parseAndLoadModuleMapFile(*File, /* IsSystem = */ false,
                                         /* ImplicitlyDiscovered = */ false,
                                         File->getDir());
  }

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemFS;
  IntrusiveRefCntPtr<InterceptorFS> VFS;
  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  DiagnosticOptions DiagnosticOpts;
  DiagnosticsEngine Diagnostics;
  SourceManager SrcMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  llvm::IntrusiveRefCntPtr<TargetInfo> Target;
  HeaderSearchOptions HdrSearchOpts;
  HeaderSearch HdrSearch;

  ModuleMap Map;
};

// Regression test for
// https://github.com/llvm/llvm-project/issues/147220.
//
// These tests specifically aim to validate that ModuleMap paths do not grow
// in an unbounded fashion in the presence of chained relative paths to extern
// modules.
//
TEST_F(ModuleMapTest, ExternModuleRelativeLookupPathIsNormalized) {
  // Root module as entry point, chained extern module references:
  // A -> B -> C
  addFile("/root/A.cppmap", R"(
module A {}
extern module B "../root/B.cppmap"
  )");
  addFile("/root/B.cppmap", R"(
module B {}
extern module C "../root/C.cppmap"
  )");
  // Leaf module
  addFile("/root/C.cppmap", "module C {}");

  ASSERT_FALSE(loadRoot("/root/A.cppmap"));

  // Now check paths used are normalised.
  llvm::SmallSet<llvm::StringRef, 4> Seen;
  for (const std::string &Path : VFS->StatPaths) {
    llvm::StringRef BaseName;
    for (llvm::StringRef Component : llvm::make_range(
             llvm::sys::path::begin(Path), llvm::sys::path::end(Path))) {
      // As `/root/A.cppmap` is absolute, there should be no relative paths at
      // lookup time.
      EXPECT_NE(Component, "..");
      // Last component is basename
      BaseName = Component;
    }
    Seen.insert(BaseName);
  }

  // Ensure the path check covered all the modules we expected.
  ASSERT_TRUE(Seen.contains("A.cppmap"));
  ASSERT_TRUE(Seen.contains("B.cppmap"));
  ASSERT_TRUE(Seen.contains("C.cppmap"));
}

} // namespace
} // namespace clang
