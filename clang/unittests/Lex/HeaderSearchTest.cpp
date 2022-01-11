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

static std::shared_ptr<TargetOptions> createTargetOptions() {
  auto TargetOpts = std::make_shared<TargetOptions>();
  TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
  return TargetOpts;
}

// The test fixture.
class HeaderSearchTest : public ::testing::Test {
protected:
  HeaderSearchTest()
      : VFS(new llvm::vfs::InMemoryFileSystem), FileMgr(FileMgrOpts, VFS),
        DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(createTargetOptions()),
        Target(TargetInfo::CreateTargetInfo(Diags, TargetOpts)),
        Search(std::make_shared<HeaderSearchOptions>(), SourceMgr, Diags,
               LangOpts, Target.get()) {}

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

  void setSearchDirs(llvm::ArrayRef<llvm::StringRef> QuotedDirs,
                     llvm::ArrayRef<llvm::StringRef> AngledDirs) {
    auto AddPath = [&](StringRef Dir, bool IsAngled) {
      VFS->addFile(Dir, 0, llvm::MemoryBuffer::getMemBuffer(""), /*User=*/None,
                   /*Group=*/None, llvm::sys::fs::file_type::directory_file);
      auto Group = IsAngled ? frontend::IncludeDirGroup::Angled
                            : frontend::IncludeDirGroup::Quoted;
      Search.getHeaderSearchOpts().AddPath(Dir, Group,
                                           /*IsFramework=*/false,
                                           /*IgnoreSysRoot=*/true);
    };

    for (llvm::StringRef Dir : QuotedDirs)
      AddPath(Dir, /*IsAngled=*/false);
    for (llvm::StringRef Dir : AngledDirs)
      AddPath(Dir, /*IsAngled=*/true);

    clang::ApplyHeaderSearchOptions(Search, Search.getHeaderSearchOpts(),
                                    LangOpts, Target->getTriple());
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

  void createModule(StringRef Mod) {
    std::string ModDir = ("/" + Mod).str();
    std::string ModHeader = (Mod + ".h").str();
    VFS->addFile(
        ModDir + "/module.modulemap", 0,
        llvm::MemoryBuffer::getMemBufferCopy(
            ("module " + Mod + " { header \"" + ModHeader + "\" }").str()));
    VFS->addFile(ModDir + "/" + ModHeader, 0,
                 llvm::MemoryBuffer::getMemBuffer(""));
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
  const DirectoryLookup *CurDir = nullptr;
  auto FoundFile = Search.LookupFile(
      "Foo/Foo.h", SourceLocation(), /*isAngled=*/true, /*FromDir=*/nullptr,
      CurDir, /*Includers=*/{}, /*SearchPath=*/nullptr,
      /*RelativePath=*/nullptr, /*RequestingModule=*/nullptr,
      /*SuggestedModule=*/nullptr, &IsMapped,
      /*IsFrameworkFound=*/nullptr);

  EXPECT_TRUE(FoundFile.hasValue());
  EXPECT_TRUE(IsMapped);
  auto &FE = FoundFile.getValue();
  auto FI = Search.getExistingFileInfo(FE);
  EXPECT_TRUE(FI);
  EXPECT_TRUE(FI->IsValid);
  EXPECT_EQ(FI->Framework.str(), "Foo");
}

TEST_F(HeaderSearchTest, SearchPathUsage) {
  Search.getHeaderSearchOpts().ImplicitModuleMaps = true;

  setSearchDirs(/*QuotedDirs=*/{"/M0"}, /*AngledDirs=*/{"/M2", "/M3"});
  createModule("M0");
  createModule("M2");
  createModule("M3");

  {
    Module *M2 = Search.lookupModule("M2");
    EXPECT_NE(M2, nullptr);
    EXPECT_EQ(Search.getSearchDirUsage(), (std::vector<bool>{0, 1, 0}));
    EXPECT_EQ(Search.computeUserEntryUsage(), (std::vector<bool>{0, 1, 0}));
  }

  addSearchDir("/M1");
  createModule("M1");

  {
    Module *M1 = Search.lookupModule("M1");
    EXPECT_NE(M1, nullptr);
    EXPECT_EQ(Search.getSearchDirUsage(), (std::vector<bool>{0, 1, 1, 0}));
    EXPECT_EQ(Search.computeUserEntryUsage(), (std::vector<bool>{0, 1, 0}));
  }
}

} // namespace
} // namespace clang
