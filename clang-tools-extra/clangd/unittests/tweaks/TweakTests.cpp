//===-- TweakTests.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "refactor/Tweak.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

TEST(FileEdits, AbsolutePath) {
  auto RelPaths = {"a.h", "foo.cpp", "test/test.cpp"};

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> MemFS(
      new llvm::vfs::InMemoryFileSystem);
  MemFS->setCurrentWorkingDirectory(testRoot());
  for (const auto *Path : RelPaths)
    MemFS->addFile(Path, 0, llvm::MemoryBuffer::getMemBuffer("", Path));
  FileManager FM(FileSystemOptions(), MemFS);
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine DE(new DiagnosticIDs, DiagOpts);
  SourceManager SM(DE, FM);

  for (const auto *Path : RelPaths) {
    auto FID = SM.createFileID(*FM.getOptionalFileRef(Path), SourceLocation(),
                               clang::SrcMgr::C_User);
    auto Res = Tweak::Effect::fileEdit(SM, FID, tooling::Replacements());
    ASSERT_THAT_EXPECTED(Res, llvm::Succeeded());
    EXPECT_EQ(Res->first, testPath(Path));
  }
}

} // namespace
} // namespace clangd
} // namespace clang
