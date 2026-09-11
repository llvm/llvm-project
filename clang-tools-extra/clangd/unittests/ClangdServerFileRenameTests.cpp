//===-- ClangdServerFileRenameTests.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdServer.h"
#include "GlobalCompilationDatabase.h"
#include "TestFS.h"
#include "support/Path.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(ClangdServerFileRename, MovesOpenDraftAfterDirectoryRename) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());
  Path Old = testPath("old/main.cpp");
  Path New = testPath("new/main.cpp");
  llvm::StringLiteral Contents = "int value;\n";
  FS.Files[Old] = Contents.str();
  Server.addDocument(Old, Contents, "7");
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  FS.Files[New] = FS.Files[Old];
  FS.Files.erase(Old);
  Server.didRenameFiles({{testPath("old"), testPath("new")}});
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  EXPECT_FALSE(Server.getDraft(Old));
  auto Moved = Server.getDraft(New);
  ASSERT_TRUE(Moved);
  EXPECT_EQ(*Moved, Contents);
}

} // namespace
} // namespace clangd
} // namespace clang
