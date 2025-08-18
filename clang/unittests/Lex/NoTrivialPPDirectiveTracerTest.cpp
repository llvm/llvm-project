//===- unittests/Lex/NoTrivialPPDirectiveTracerTest.cpp -------------------===//
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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <initializer_list>

using namespace clang;

namespace {
class NoTrivialPPDirectiveTracerTest : public ::testing::Test {
protected:
  NoTrivialPPDirectiveTracerTest()
      : VFS(llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>()),
        FileMgr(FileMgrOpts, VFS),
        Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions) {
    TargetOpts->Triple = "x86_64-unknown-linux-gnu";
    Target = TargetInfo::CreateTargetInfo(Diags, *TargetOpts);
  }

  void addFile(const char *source, StringRef Filename) {
    VFS->addFile(Filename, 0, llvm::MemoryBuffer::getMemBuffer(source),
                 /*User=*/std::nullopt,
                 /*Group=*/std::nullopt,
                 llvm::sys::fs::file_type::regular_file);
  }

  std::unique_ptr<Preprocessor> getPreprocessor(const char *source,
                                                Language Lang) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(source);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    std::vector<std::string> Includes;
    LangOptions::setLangDefaults(LangOpts, Lang, Target->getTriple(), Includes,
                                 LangStandard::lang_cxx20);
    LangOpts.CPlusPlusModules = true;
    if (Lang != Language::CXX) {
      LangOpts.Modules = true;
      LangOpts.ImplicitModules = true;
    }

    HeaderInfo.emplace(HSOpts, SourceMgr, Diags, LangOpts, Target.get());

    auto DE = FileMgr.getOptionalDirectoryRef(".");
    assert(DE);
    auto DL = DirectoryLookup(*DE, SrcMgr::C_User, /*isFramework=*/false);
    HeaderInfo->AddSearchPath(DL, /*isAngled=*/false);

    return std::make_unique<Preprocessor>(PPOpts, Diags, LangOpts, SourceMgr,
                                          *HeaderInfo, ModLoader,
                                          /*IILookup=*/nullptr,
                                          /*OwnsHeaderSearch=*/false);
  }

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
  LangOptions LangOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearchOptions HSOpts;
  std::optional<HeaderSearch> HeaderInfo;
  PreprocessorOptions PPOpts;
};

TEST_F(NoTrivialPPDirectiveTracerTest, TrivialDirective) {
  const char *source = R"(
    #line 7
    # 1 __FILE__ 1 3
    #ident "$Header:$"
    #pragma comment(lib, "msvcrt.lib")
    #pragma mark LLVM's world
    #pragma detect_mismatch("test", "1")
    #pragma clang __debug dump Test
    #pragma message "test"
    #pragma GCC warning "Foo"
    #pragma GCC error "Foo"
    #pragma gcc diagnostic push
    #pragma gcc diagnostic pop
    #pragma GCC diagnostic ignored "-Wframe-larger-than"
    #pragma OPENCL EXTENSION __cl_clang_variadic_functions : enable
    #pragma warning(push)
    #pragma warning(pop)
    #pragma execution_character_set(push, "UTF-8")
    #pragma execution_character_set(pop)
    #pragma clang assume_nonnull begin
    #pragma clang assume_nonnull end
    int foo;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, Language::CXX);
  PP->Initialize(*Target);
  PP->EnterMainSourceFile();
  Token Tok;
  PP->Lex(Tok);
  EXPECT_FALSE(PP->hasSeenNoTrivialPPDirective());
}

TEST_F(NoTrivialPPDirectiveTracerTest, IncludeDirective) {
  const char *source = R"(
    #include "header.h"
    int foo;
  )";
  const char *header = R"(
    #ifndef HEADER_H
    #define HEADER_H
    #endif // HEADER_H
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, Language::CXX);
  addFile(header, "header.h");
  PP->Initialize(*Target);
  PP->EnterMainSourceFile();
  Token Tok;
  PP->Lex(Tok);
  EXPECT_TRUE(PP->hasSeenNoTrivialPPDirective());
}

TEST_F(NoTrivialPPDirectiveTracerTest, DefineDirective) {
  const char *source = R"(
    #define FOO
    int foo;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, Language::CXX);
  PP->Initialize(*Target);
  PP->EnterMainSourceFile();
  Token Tok;
  PP->Lex(Tok);
  EXPECT_TRUE(PP->hasSeenNoTrivialPPDirective());
}

TEST_F(NoTrivialPPDirectiveTracerTest, UnDefineDirective) {
  const char *source = R"(
    #undef FOO
    int foo;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, Language::CXX);
  PP->Initialize(*Target);
  PP->setPredefines("#define FOO");
  PP->EnterMainSourceFile();
  Token Tok;
  PP->Lex(Tok);
  EXPECT_TRUE(PP->hasSeenNoTrivialPPDirective());
}

TEST_F(NoTrivialPPDirectiveTracerTest, IfDefinedDirective) {
  const char *source = R"(
    #if defined(FOO)
    #endif
    int foo;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, Language::CXX);
  PP->Initialize(*Target);
  PP->setPredefines("#define FOO");
  PP->EnterMainSourceFile();
  Token Tok;
  PP->Lex(Tok);
  EXPECT_TRUE(PP->hasSeenNoTrivialPPDirective());
}

} // namespace
