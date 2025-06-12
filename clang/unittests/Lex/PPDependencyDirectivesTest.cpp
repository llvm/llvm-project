//===- unittests/Lex/PPDependencyDirectivesTest.cpp -------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/DependencyDirectivesScanner.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

using namespace clang;

namespace {

// The test fixture.
class PPDependencyDirectivesTest : public ::testing::Test {
protected:
  PPDependencyDirectivesTest()
      : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
        Diags(DiagID, DiagOpts, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions) {
    TargetOpts->Triple = "x86_64-apple-macos12";
    Target = TargetInfo::CreateTargetInfo(Diags, *TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

class IncludeCollector : public PPCallbacks {
public:
  Preprocessor &PP;
  SmallVectorImpl<StringRef> &IncludedFiles;

  IncludeCollector(Preprocessor &PP, SmallVectorImpl<StringRef> &IncludedFiles)
      : PP(PP), IncludedFiles(IncludedFiles) {}

  void LexedFileChanged(FileID FID, LexedFileChangeReason Reason,
                        SrcMgr::CharacteristicKind FileType, FileID PrevFID,
                        SourceLocation Loc) override {
    if (Reason != LexedFileChangeReason::EnterFile)
      return;
    if (FID == PP.getPredefinesFileID())
      return;
    StringRef Filename =
        PP.getSourceManager().getSLocEntry(FID).getFile().getName();
    IncludedFiles.push_back(Filename);
  }
};

TEST_F(PPDependencyDirectivesTest, MacroGuard) {
  // "head1.h" has a macro guard and should only be included once.
  // "head2.h" and "head3.h" have tokens following the macro check, they should
  // be included multiple times.

  auto VFS = new llvm::vfs::InMemoryFileSystem();
  VFS->addFile(
      "head1.h", 0,
      llvm::MemoryBuffer::getMemBuffer("#ifndef H1_H\n#define H1_H\n#endif\n"));
  VFS->addFile(
      "head2.h", 0,
      llvm::MemoryBuffer::getMemBuffer("#ifndef H2_H\n#define H2_H\n#endif\n\n"
                                       "extern int foo;\n"));
  VFS->addFile("head3.h", 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#ifndef H3_H\n#define H3_H\n#endif\n\n"
                   "#ifdef SOMEMAC\nextern int foo;\n#endif\n"));
  VFS->addFile("main.c", 0,
               llvm::MemoryBuffer::getMemBuffer(
                   "#include \"head1.h\"\n#include \"head1.h\"\n"
                   "#include \"head2.h\"\n#include \"head2.h\"\n"
                   "#include \"head3.h\"\n#include \"head3.h\"\n"));
  FileMgr.setVirtualFileSystem(VFS);

  OptionalFileEntryRef FE;
  ASSERT_THAT_ERROR(FileMgr.getFileRef("main.c").moveInto(FE),
                    llvm::Succeeded());
  SourceMgr.setMainFileID(
      SourceMgr.createFileID(*FE, SourceLocation(), SrcMgr::C_User));

  struct DepDirectives {
    SmallVector<dependency_directives_scan::Token> Tokens;
    SmallVector<dependency_directives_scan::Directive> Directives;
  };

  class TestDependencyDirectivesGetter : public DependencyDirectivesGetter {
    FileManager &FileMgr;
    SmallVector<std::unique_ptr<DepDirectives>> DepDirectivesObjects;

  public:
    TestDependencyDirectivesGetter(FileManager &FileMgr) : FileMgr(FileMgr) {}

    std::unique_ptr<DependencyDirectivesGetter>
    cloneFor(FileManager &FileMgr) override {
      return std::make_unique<TestDependencyDirectivesGetter>(FileMgr);
    }

    std::optional<ArrayRef<dependency_directives_scan::Directive>>
    operator()(FileEntryRef File) override {
      DepDirectivesObjects.push_back(std::make_unique<DepDirectives>());
      StringRef Input = (*FileMgr.getBufferForFile(File))->getBuffer();
      bool Err = scanSourceForDependencyDirectives(
          Input, DepDirectivesObjects.back()->Tokens,
          DepDirectivesObjects.back()->Directives);
      EXPECT_FALSE(Err);
      return DepDirectivesObjects.back()->Directives;
    }
  };
  TestDependencyDirectivesGetter GetDependencyDirectives(FileMgr);

  PreprocessorOptions PPOpts;
  HeaderSearchOptions HSOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, Target.get());
  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);

  PP.setDependencyDirectivesGetter(GetDependencyDirectives);

  SmallVector<StringRef> IncludedFiles;
  PP.addPPCallbacks(std::make_unique<IncludeCollector>(PP, IncludedFiles));
  PP.EnterMainSourceFile();
  PP.LexTokensUntilEOF();

  SmallVector<std::string> IncludedFilesSlash;
  for (StringRef IncludedFile : IncludedFiles)
    IncludedFilesSlash.push_back(
        llvm::sys::path::convert_to_slash(IncludedFile));
  SmallVector<std::string> ExpectedIncludes{
      "main.c", "./head1.h", "./head2.h", "./head2.h", "./head3.h", "./head3.h",
  };
  EXPECT_EQ(IncludedFilesSlash, ExpectedIncludes);
}

} // anonymous namespace
