//===- unittests/Lex/HeaderSearchTest.cpp ------ HeaderSearch tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/HeaderSearch.h"
#include "HeaderMapTestUtils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Serialization/InMemoryModuleCache.h"
#include "gtest/gtest.h"

namespace clang {
namespace {

// The test fixture.
class HeaderSearchTest : public ::testing::Test {
protected:
  HeaderSearchTest()
      : VFS(new llvm::vfs::InMemoryFileSystem), FileMgr(FileMgrOpts, VFS),
        DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions),
        Search(std::make_shared<HeaderSearchOptions>(), SourceMgr, Diags,
               LangOpts, Target.get()) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  void addSearchDir(llvm::StringRef Dir) {
    VFS->addFile(Dir, 0, llvm::MemoryBuffer::getMemBuffer(""), /*User=*/None,
                 /*Group=*/None, llvm::sys::fs::file_type::directory_file);
    auto DE = FileMgr.getOptionalDirectoryRef(Dir);
    assert(DE);
    auto DL = DirectoryLookup(*DE, SrcMgr::C_User, /*isFramework=*/false);
    Search.AddSearchPath(DL, /*isAngled=*/false);
  }

  void addHeaderMap(llvm::StringRef Filename,
                    std::unique_ptr<llvm::MemoryBuffer> Buf) {
    VFS->addFile(Filename, 0, std::move(Buf), /*User=*/None, /*Group=*/None,
                 llvm::sys::fs::file_type::regular_file);
    auto FE = FileMgr.getFile(Filename, true);
    assert(FE);

    // Test class supports only one HMap at a time.
    assert(!HMap);
    HMap = HeaderMap::Create(*FE, FileMgr);
    auto DL =
        DirectoryLookup(HMap.get(), SrcMgr::C_User, /*isFramework=*/false);
    Search.AddSearchPath(DL, /*isAngled=*/false);
  }

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
  HeaderSearch Search;
  std::unique_ptr<HeaderMap> HMap;
};

TEST_F(HeaderSearchTest, NoSearchDir) {
  EXPECT_EQ(Search.search_dir_size(), 0u);
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/x/y/z", /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "/x/y/z");
}

TEST_F(HeaderSearchTest, SimpleShorten) {
  addSearchDir("/x");
  addSearchDir("/x/y");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/x/y/z", /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "z");
  addSearchDir("/a/b/");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/a/b/c", /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "c");
}

TEST_F(HeaderSearchTest, ShortenWithWorkingDir) {
  addSearchDir("x/y");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/a/b/c/x/y/z",
                                                   /*WorkingDir=*/"/a/b/c",
                                                   /*MainFile=*/""),
            "z");
}

TEST_F(HeaderSearchTest, Dots) {
  addSearchDir("/x/./y/");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/x/y/./z",
                                                   /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "z");
  addSearchDir("a/.././c/");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/m/n/./c/z",
                                                   /*WorkingDir=*/"/m/n/",
                                                   /*MainFile=*/""),
            "z");
}

#ifdef _WIN32
TEST_F(HeaderSearchTest, BackSlash) {
  addSearchDir("C:\\x\\y\\");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("C:\\x\\y\\z\\t",
                                                   /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "z/t");
}

TEST_F(HeaderSearchTest, BackSlashWithDotDot) {
  addSearchDir("..\\y");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("C:\\x\\y\\z\\t",
                                                   /*WorkingDir=*/"C:/x/y/",
                                                   /*MainFile=*/""),
            "z/t");
}
#endif

TEST_F(HeaderSearchTest, DotDotsWithAbsPath) {
  addSearchDir("/x/../y/");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/y/z",
                                                   /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "z");
}

TEST_F(HeaderSearchTest, IncludeFromSameDirectory) {
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/y/z/t.h",
                                                   /*WorkingDir=*/"",
                                                   /*MainFile=*/"/y/a.cc"),
            "z/t.h");

  addSearchDir("/");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/y/z/t.h",
                                                   /*WorkingDir=*/"",
                                                   /*MainFile=*/"/y/a.cc"),
            "y/z/t.h");
}

// Helper struct with null terminator character to make MemoryBuffer happy.
template <class FileTy, class PaddingTy>
struct NullTerminatedFile : public FileTy {
  PaddingTy Padding = 0;
};

TEST_F(HeaderSearchTest, HeaderMapReverseLookup) {
  typedef NullTerminatedFile<test::HMapFileMock<2, 32>, char> FileTy;
  FileTy File;
  File.init();

  test::HMapFileMockMaker<FileTy> Maker(File);
  auto a = Maker.addString("d.h");
  auto b = Maker.addString("b/");
  auto c = Maker.addString("c.h");
  Maker.addBucket("d.h", a, b, c);

  addHeaderMap("/x/y/z.hmap", File.getBuffer());
  addSearchDir("/a");

  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/a/b/c.h",
                                                   /*WorkingDir=*/"",
                                                   /*MainFile=*/""),
            "d.h");
}

} // namespace
} // namespace clang
