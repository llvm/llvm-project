//===-- HeadersTests.cpp - Include headers unit tests -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

class HeadersTest : public ::testing::Test {
public:
  HeadersTest() {
    CDB.ExtraClangFlags = {SearchDirArg.c_str()};
    FS.Files[MainFile] = "";
  }

protected:
  // Calculates the include path for \p Header, or returns "" on error.
  std::string calculate(PathRef Header) {
    auto VFS = FS.getTaggedFileSystem(MainFile).Value;
    auto Cmd = CDB.getCompileCommand(MainFile);
    assert(static_cast<bool>(Cmd));
    VFS->setCurrentWorkingDirectory(Cmd->Directory);
    auto Path =
        calculateIncludePath(MainFile, FS.Files[MainFile], Header, *Cmd, VFS);
    if (!Path) {
      llvm::consumeError(Path.takeError());
      return std::string();
    }
    return std::move(*Path);
  }
  MockFSProvider FS;
  MockCompilationDatabase CDB;
  std::string MainFile = testPath("main.cpp");
  std::string Subdir = testPath("sub");
  std::string SearchDirArg = (llvm::Twine("-I") + Subdir).str();
};

TEST_F(HeadersTest, InsertInclude) {
  std::string Path = testPath("sub/bar.h");
  FS.Files[Path] = "";
  EXPECT_EQ(calculate(Path), "\"bar.h\"");
}

TEST_F(HeadersTest, DontInsertDuplicateSameName) {
  FS.Files[MainFile] = R"cpp(
#include "bar.h"
)cpp";
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";
  EXPECT_EQ(calculate(BarHeader), "");
}

TEST_F(HeadersTest, DontInsertDuplicateDifferentName) {
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";
  FS.Files[MainFile] = R"cpp(
#include "sub/bar.h"  // not shortest
)cpp";
  EXPECT_EQ(calculate(BarHeader), "");
}

TEST_F(HeadersTest, StillInsertIfTrasitivelyIncluded) {
  std::string BazHeader = testPath("sub/baz.h");
  FS.Files[BazHeader] = "";
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = R"cpp(
#include "baz.h"
)cpp";
  FS.Files[MainFile] = R"cpp(
#include "bar.h"
)cpp";
  EXPECT_EQ(calculate(BazHeader), "\"baz.h\"");
}

TEST_F(HeadersTest, DoNotInsertIfInSameFile) {
  MainFile = testPath("main.h");
  FS.Files[MainFile] = "";
  EXPECT_EQ(calculate(MainFile), "");
}

} // namespace
} // namespace clangd
} // namespace clang

