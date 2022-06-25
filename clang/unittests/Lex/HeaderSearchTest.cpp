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
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/Support/MemoryBuffer.h"
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

  void addSystemFrameworkSearchDir(llvm::StringRef Dir) {
    VFS->addFile(Dir, 0, llvm::MemoryBuffer::getMemBuffer(""), /*User=*/None,
                 /*Group=*/None, llvm::sys::fs::file_type::directory_file);
    auto DE = FileMgr.getOptionalDirectoryRef(Dir);
    assert(DE);
    auto DL = DirectoryLookup(*DE, SrcMgr::C_System, /*isFramework=*/true);
    Search.AddSystemSearchPath(DL);
  }

  void addHeaderMap(llvm::StringRef Filename,
                    std::unique_ptr<llvm::MemoryBuffer> Buf,
                    bool isAngled = false) {
    VFS->addFile(Filename, 0, std::move(Buf), /*User=*/None, /*Group=*/None,
                 llvm::sys::fs::file_type::regular_file);
    auto FE = FileMgr.getFile(Filename, true);
    assert(FE);

    // Test class supports only one HMap at a time.
    assert(!HMap);
    HMap = HeaderMap::Create(*FE, FileMgr);
    auto DL =
        DirectoryLookup(HMap.get(), SrcMgr::C_User, /*isFramework=*/false);
    Search.AddSearchPath(DL, isAngled);
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

TEST_F(HeaderSearchTest, SdkFramework) {
  addSystemFrameworkSearchDir(
      "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.3.sdk/Frameworks/");
  bool IsSystem = false;
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics(
                "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/"
                "Frameworks/AppKit.framework/Headers/NSView.h",
                /*WorkingDir=*/"",
                /*MainFile=*/"", &IsSystem),
            "AppKit/NSView.h");
  EXPECT_TRUE(IsSystem);
}

TEST_F(HeaderSearchTest, NestedFramework) {
  addSystemFrameworkSearchDir("/Platforms/MacOSX/Frameworks");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics(
                "/Platforms/MacOSX/Frameworks/AppKit.framework/Frameworks/"
                "Sub.framework/Headers/Sub.h",
                /*WorkingDir=*/"",
                /*MainFile=*/""),
            "Sub/Sub.h");
}

TEST_F(HeaderSearchTest, HeaderFrameworkLookup) {
  std::string HeaderPath = "/tmp/Frameworks/Foo.framework/Headers/Foo.h";
  addSystemFrameworkSearchDir("/tmp/Frameworks");
  VFS->addFile(
      HeaderPath, 0, llvm::MemoryBuffer::getMemBufferCopy("", HeaderPath),
      /*User=*/None, /*Group=*/None, llvm::sys::fs::file_type::regular_file);

  bool IsFrameworkFound = false;
  auto FoundFile = Search.LookupFile(
      "Foo/Foo.h", SourceLocation(), /*isAngled=*/true, /*FromDir=*/nullptr,
      /*CurDir=*/nullptr, /*Includers=*/{}, /*SearchPath=*/nullptr,
      /*RelativePath=*/nullptr, /*RequestingModule=*/nullptr,
      /*SuggestedModule=*/nullptr, /*IsMapped=*/nullptr, &IsFrameworkFound);

  EXPECT_TRUE(FoundFile.has_value());
  EXPECT_TRUE(IsFrameworkFound);
  auto &FE = *FoundFile;
  auto FI = Search.getExistingFileInfo(FE);
  EXPECT_TRUE(FI);
  EXPECT_TRUE(FI->IsValid);
  EXPECT_EQ(FI->Framework.str(), "Foo");
  EXPECT_EQ(Search.getIncludeNameForHeader(FE), "Foo/Foo.h");
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

TEST_F(HeaderSearchTest, HeaderMapFrameworkLookup) {
  typedef NullTerminatedFile<test::HMapFileMock<4, 128>, char> FileTy;
  FileTy File;
  File.init();

  std::string HeaderDirName = "/tmp/Sources/Foo/Headers/";
  std::string HeaderName = "Foo.h";
  if (is_style_windows(llvm::sys::path::Style::native)) {
    // Force header path to be absolute on windows.
    // As headermap content should represent absolute locations.
    HeaderDirName = "C:" + HeaderDirName;
  }

  test::HMapFileMockMaker<FileTy> Maker(File);
  auto a = Maker.addString("Foo/Foo.h");
  auto b = Maker.addString(HeaderDirName);
  auto c = Maker.addString(HeaderName);
  Maker.addBucket("Foo/Foo.h", a, b, c);
  addHeaderMap("product-headers.hmap", File.getBuffer(), /*isAngled=*/true);

  VFS->addFile(
      HeaderDirName + HeaderName, 0,
      llvm::MemoryBuffer::getMemBufferCopy("", HeaderDirName + HeaderName),
      /*User=*/None, /*Group=*/None, llvm::sys::fs::file_type::regular_file);

  bool IsMapped = false;
  auto FoundFile = Search.LookupFile(
      "Foo/Foo.h", SourceLocation(), /*isAngled=*/true, /*FromDir=*/nullptr,
      /*CurDir=*/nullptr, /*Includers=*/{}, /*SearchPath=*/nullptr,
      /*RelativePath=*/nullptr, /*RequestingModule=*/nullptr,
      /*SuggestedModule=*/nullptr, &IsMapped,
      /*IsFrameworkFound=*/nullptr);

  EXPECT_TRUE(FoundFile.has_value());
  EXPECT_TRUE(IsMapped);
  auto &FE = *FoundFile;
  auto FI = Search.getExistingFileInfo(FE);
  EXPECT_TRUE(FI);
  EXPECT_TRUE(FI->IsValid);
  EXPECT_EQ(FI->Framework.str(), "Foo");
  EXPECT_EQ(Search.getIncludeNameForHeader(FE), "Foo/Foo.h");
}

} // namespace
} // namespace clang
