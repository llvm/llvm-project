//====-- unittests/Frontend/SearchPathTest.cpp - FrontendAction tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace clang {
namespace {

class SearchPathTest : public ::testing::Test {
protected:
  SearchPathTest()
      : Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        VFS(new llvm::vfs::InMemoryFileSystem),
        FileMgr(FileSystemOptions(), VFS), SourceMgr(Diags, FileMgr),
        Invocation(std::make_unique<CompilerInvocation>()) {}

  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  FileManager FileMgr;
  SourceManager SourceMgr;
  std::unique_ptr<CompilerInvocation> Invocation;
  IntrusiveRefCntPtr<TargetInfo> Target;

  void addDirectories(ArrayRef<StringRef> Dirs) {
    for (StringRef Dir : Dirs) {
      VFS->addFile(Dir, 0, llvm::MemoryBuffer::getMemBuffer(""),
                   /*User=*/std::nullopt, /*Group=*/std::nullopt,
                   llvm::sys::fs::file_type::directory_file);
    }
  }

  std::unique_ptr<HeaderSearch>
  makeHeaderSearchFromCC1Args(llvm::opt::ArgStringList Args) {
    CompilerInvocation::CreateFromArgs(*Invocation, Args, Diags);
    HeaderSearchOptions HSOpts = Invocation->getHeaderSearchOpts();
    LangOptions LangOpts = Invocation->getLangOpts();
    Target = TargetInfo::CreateTargetInfo(Diags, Invocation->getTargetOpts());
    auto HeaderInfo = std::make_unique<HeaderSearch>(HSOpts, SourceMgr, Diags,
                                                     LangOpts, Target.get());
    ApplyHeaderSearchOptions(*HeaderInfo, HSOpts, LangOpts,
                             Target->getTriple());
    return HeaderInfo;
  }
};

TEST_F(SearchPathTest, SearchPathOrder) {
  addDirectories({"One", "Two", "Three", "Four", "Five", "Six", "Seven",
                  "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
                  "Fourteen", "Fifteen", "Sixteen", "Seventeen"});
  llvm::opt::ArgStringList Args = {
      // Make sure to use a triple and language that don't automatically add any
      // search paths.
      "-triple", "arm64-apple-darwin24.4.0", "-x", "c",

      // clang-format off
      "-internal-isystem", "One",
      "-iwithsysroot", "Two",
      "-c-isystem", "Three",
      "-IFour",
      "-idirafter", "Five",
      "-internal-externc-isystem", "Six",
      "-iwithprefix", "Seven",
      "-FEight",
      "-idirafter", "Nine",
      "-iframeworkwithsysroot", "Ten",
      "-internal-iframework", "Eleven",
      "-iframework", "Twelve",
      "-iwithprefixbefore", "Thirteen",
      "-internal-isystem", "Fourteen",
      "-isystem", "Fifteen",
      "-ISixteen",
      "-iwithsysroot", "Seventeen",
      // clang-format on
  };

  // The search path arguments get categorized by IncludeDirGroup, but
  // ultimately are sorted with some groups mixed together and some flags sorted
  // very specifically within their group. The conceptual groups below don't
  // exactly correspond to IncludeDirGroup.
  const std::vector<StringRef> expected = {
      // User paths: -I and -F mixed together, -iwithprefixbefore.
      /*-I*/ "Four",
      /*-F*/ "Eight",
      /*-I*/ "Sixteen",
      /*-iwithprefixbefore*/ "Thirteen",

      // System paths: -isystem and -iwithsysroot, -iframework,
      // -iframeworkwithsysroot, one of {-c-isystem, -cxx-isystem,
      // -objc-isystem, -objcxx-isystem}
      /*-iwithsysroot*/ "Two",
      /*-isystem*/ "Fifteen",
      /*-iwithsysroot*/ "Seventeen",
      /*-iframework*/ "Twelve",
      /*-iframeworkwithsysroot*/ "Ten",
      /*-c-isystem*/ "Three",

      // Internal paths: -internal-isystem and -internal-externc-isystem,
      // -internal-iframework
      /*-internal-isystem*/ "One",
      /*-internal-externc-isystem*/ "Six",
      /*-internal-isystem*/ "Fourteen",
      /*-internal-iframework*/ "Eleven",

      // After paths: -iwithprefix, -idirafter
      /*-iwithprefix*/ "Seven",
      /*-idirafter*/ "Five",
      /*-idirafter*/ "Nine",
  };

  auto HeaderInfo = makeHeaderSearchFromCC1Args(Args);
  ConstSearchDirRange SearchDirs(HeaderInfo->angled_dir_begin(),
                                 HeaderInfo->search_dir_end());
  for (auto SearchPaths : zip_longest(SearchDirs, expected)) {
    auto ActualDirectory = std::get<0>(SearchPaths);
    EXPECT_TRUE(ActualDirectory.has_value());
    auto ExpectedPath = std::get<1>(SearchPaths);
    EXPECT_TRUE(ExpectedPath.has_value());
    if (ActualDirectory && ExpectedPath) {
      EXPECT_EQ(ActualDirectory->getName(), *ExpectedPath);
    }
  }
}

} // namespace
} // namespace clang
