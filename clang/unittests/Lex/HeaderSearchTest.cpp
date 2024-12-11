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
#include <memory>
#include <string>

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
    VFS->addFile(
        Dir, 0, llvm::MemoryBuffer::getMemBuffer(""), /*User=*/std::nullopt,
        /*Group=*/std::nullopt, llvm::sys::fs::file_type::directory_file);
    auto DE = FileMgr.getOptionalDirectoryRef(Dir);
    assert(DE);
    auto DL = DirectoryLookup(*DE, SrcMgr::C_User, /*isFramework=*/false);
    Search.AddSearchPath(DL, /*isAngled=*/false);
  }

  void addFrameworkSearchDir(llvm::StringRef Dir, bool IsSystem = true) {
    VFS->addFile(
        Dir, 0, llvm::MemoryBuffer::getMemBuffer(""), /*User=*/std::nullopt,
        /*Group=*/std::nullopt, llvm::sys::fs::file_type::directory_file);
    auto DE = FileMgr.getOptionalDirectoryRef(Dir);
    assert(DE);
    auto DL = DirectoryLookup(*DE, IsSystem ? SrcMgr::C_System : SrcMgr::C_User,
                              /*isFramework=*/true);
    if (IsSystem)
      Search.AddSystemSearchPath(DL);
    else
      Search.AddSearchPath(DL, /*isAngled=*/true);
  }

  void addHeaderMap(llvm::StringRef Filename,
                    std::unique_ptr<llvm::MemoryBuffer> Buf,
                    bool isAngled = false) {
    VFS->addFile(Filename, 0, std::move(Buf), /*User=*/std::nullopt,
                 /*Group=*/std::nullopt,
                 llvm::sys::fs::file_type::regular_file);
    auto FE = FileMgr.getOptionalFileRef(Filename, true);
    assert(FE);

    // Test class supports only one HMap at a time.
    assert(!HMap);
    HMap = HeaderMap::Create(*FE, FileMgr);
    auto DL = DirectoryLookup(HMap.get(), SrcMgr::C_User);
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

TEST_F(HeaderSearchTest, RelativeDirs) {
  ASSERT_FALSE(VFS->setCurrentWorkingDirectory("/root/some/dir"));
  addSearchDir("..");
  EXPECT_EQ(
      Search.suggestPathToFileForDiagnostics("/root/some/foo.h",
                                             /*WorkingDir=*/"/root/some/dir",
                                             /*MainFile=*/""),
      "foo.h");
  EXPECT_EQ(
      Search.suggestPathToFileForDiagnostics("../foo.h",
                                             /*WorkingDir=*/"/root/some/dir",
                                             /*MainFile=*/""),
      "foo.h");
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

TEST_F(HeaderSearchTest, BothDotDots) {
  addSearchDir("/x/../y/");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics("/x/../y/z",
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
  addFrameworkSearchDir(
      "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.3.sdk/Frameworks/");
  bool IsAngled = false;
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics(
                "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/"
                "Frameworks/AppKit.framework/Headers/NSView.h",
                /*WorkingDir=*/"",
                /*MainFile=*/"", &IsAngled),
            "AppKit/NSView.h");
  EXPECT_TRUE(IsAngled);

  addFrameworkSearchDir("/System/Developer/Library/Framworks/",
                        /*IsSystem*/ false);
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics(
                "/System/Developer/Library/Framworks/"
                "Foo.framework/Headers/Foo.h",
                /*WorkingDir=*/"",
                /*MainFile=*/"", &IsAngled),
            "Foo/Foo.h");
  // Expect to be true even though we passed false to IsSystem earlier since
  // all frameworks should be treated as <>.
  EXPECT_TRUE(IsAngled);
}

TEST_F(HeaderSearchTest, NestedFramework) {
  addFrameworkSearchDir("/Platforms/MacOSX/Frameworks");
  EXPECT_EQ(Search.suggestPathToFileForDiagnostics(
                "/Platforms/MacOSX/Frameworks/AppKit.framework/Frameworks/"
                "Sub.framework/Headers/Sub.h",
                /*WorkingDir=*/"",
                /*MainFile=*/""),
            "Sub/Sub.h");
}

TEST_F(HeaderSearchTest, HeaderFrameworkLookup) {
  std::string HeaderPath = "/tmp/Frameworks/Foo.framework/Headers/Foo.h";
  addFrameworkSearchDir("/tmp/Frameworks");
  VFS->addFile(HeaderPath, 0,
               llvm::MemoryBuffer::getMemBufferCopy("", HeaderPath),
               /*User=*/std::nullopt, /*Group=*/std::nullopt,
               llvm::sys::fs::file_type::regular_file);

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
      /*User=*/std::nullopt, /*Group=*/std::nullopt,
      llvm::sys::fs::file_type::regular_file);

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
  EXPECT_EQ(Search.getIncludeNameForHeader(FE), "Foo/Foo.h");
}

TEST_F(HeaderSearchTest, HeaderFileInfoMerge) {
  auto AddHeader = [&](std::string HeaderPath) -> FileEntryRef {
    VFS->addFile(HeaderPath, 0,
                 llvm::MemoryBuffer::getMemBufferCopy("", HeaderPath),
                 /*User=*/std::nullopt, /*Group=*/std::nullopt,
                 llvm::sys::fs::file_type::regular_file);
    return *FileMgr.getOptionalFileRef(HeaderPath);
  };

  class MockExternalHeaderFileInfoSource : public ExternalHeaderFileInfoSource {
    HeaderFileInfo GetHeaderFileInfo(FileEntryRef FE) {
      HeaderFileInfo HFI;
      auto FileName = FE.getName();
      if (FileName == ModularPath)
        HFI.mergeModuleMembership(ModuleMap::NormalHeader);
      else if (FileName == TextualPath)
        HFI.mergeModuleMembership(ModuleMap::TextualHeader);
      HFI.External = true;
      HFI.IsValid = true;
      return HFI;
    }

  public:
    std::string ModularPath = "/modular.h";
    std::string TextualPath = "/textual.h";
  };

  auto ExternalSource = std::make_unique<MockExternalHeaderFileInfoSource>();
  Search.SetExternalSource(ExternalSource.get());

  // Everything should start out external.
  auto ModularFE = AddHeader(ExternalSource->ModularPath);
  auto TextualFE = AddHeader(ExternalSource->TextualPath);
  EXPECT_TRUE(Search.getExistingFileInfo(ModularFE)->External);
  EXPECT_TRUE(Search.getExistingFileInfo(TextualFE)->External);

  // Marking the same role should keep it external
  Search.MarkFileModuleHeader(ModularFE, ModuleMap::NormalHeader,
                              /*isCompilingModuleHeader=*/false);
  Search.MarkFileModuleHeader(TextualFE, ModuleMap::TextualHeader,
                              /*isCompilingModuleHeader=*/false);
  EXPECT_TRUE(Search.getExistingFileInfo(ModularFE)->External);
  EXPECT_TRUE(Search.getExistingFileInfo(TextualFE)->External);

  // textual -> modular should update the HFI, but modular -> textual should be
  // a no-op.
  Search.MarkFileModuleHeader(ModularFE, ModuleMap::TextualHeader,
                              /*isCompilingModuleHeader=*/false);
  Search.MarkFileModuleHeader(TextualFE, ModuleMap::NormalHeader,
                              /*isCompilingModuleHeader=*/false);
  auto ModularFI = Search.getExistingFileInfo(ModularFE);
  auto TextualFI = Search.getExistingFileInfo(TextualFE);
  EXPECT_TRUE(ModularFI->External);
  EXPECT_TRUE(ModularFI->isModuleHeader);
  EXPECT_FALSE(ModularFI->isTextualModuleHeader);
  EXPECT_FALSE(TextualFI->External);
  EXPECT_TRUE(TextualFI->isModuleHeader);
  EXPECT_FALSE(TextualFI->isTextualModuleHeader);

  // Compiling the module should make the HFI local.
  Search.MarkFileModuleHeader(ModularFE, ModuleMap::NormalHeader,
                              /*isCompilingModuleHeader=*/true);
  Search.MarkFileModuleHeader(TextualFE, ModuleMap::NormalHeader,
                              /*isCompilingModuleHeader=*/true);
  EXPECT_FALSE(Search.getExistingFileInfo(ModularFE)->External);
  EXPECT_FALSE(Search.getExistingFileInfo(TextualFE)->External);
}

} // namespace
} // namespace clang
