//===- unittests/Lex/ModuleDeclStateTest.cpp - PPCallbacks tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInvocation.h"
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

class CheckNamedModuleImportingCB : public PPCallbacks {
  Preprocessor &PP;
  std::vector<bool> IsImportingNamedModulesAssertions;
  std::size_t NextCheckingIndex;

public:
  CheckNamedModuleImportingCB(Preprocessor &PP,
                              std::initializer_list<bool> lists)
      : PP(PP), IsImportingNamedModulesAssertions(lists), NextCheckingIndex(0) {
  }

  void moduleImport(SourceLocation ImportLoc, ModuleIdPath Path,
                    const Module *Imported) override {
    ASSERT_TRUE(NextCheckingIndex < IsImportingNamedModulesAssertions.size());
    EXPECT_EQ(PP.isInImportingCXXNamedModules(),
              IsImportingNamedModulesAssertions[NextCheckingIndex]);
    NextCheckingIndex++;

    ASSERT_EQ(Imported, nullptr);
  }

  // Currently, only the named module will be handled by `moduleImport`
  // callback.
  std::size_t importNamedModuleNum() { return NextCheckingIndex; }
};
class ModuleDeclStateTest : public ::testing::Test {
protected:
  ModuleDeclStateTest()
      : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
        Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions), Invocation() {
    TargetOpts->Triple = "x86_64-unknown-linux-gnu";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  LangOptions &getLangOpts(ArrayRef<const char *> CommandLineArgs) {
    CompilerInvocation::CreateFromArgs(Invocation, CommandLineArgs, Diags);
    return *Invocation.getLangOpts();
  }

  std::unique_ptr<Preprocessor>
  getPreprocessor(const char *source, ArrayRef<const char *> CommandLineArgs) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(source);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    LangOptions &LangOpts = getLangOpts(CommandLineArgs);
    HeaderInfo.emplace(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                       Diags, LangOpts, Target.get());

    return std::make_unique<Preprocessor>(
        std::make_shared<PreprocessorOptions>(), Diags, LangOpts, SourceMgr,
        *HeaderInfo, ModLoader,
        /*IILookup =*/nullptr,
        /*OwnsHeaderSearch =*/false);
  }

  void preprocess(Preprocessor &PP, std::unique_ptr<PPCallbacks> C) {
    PP.Initialize(*Target);
    PP.addPPCallbacks(std::move(C));
    PP.EnterMainSourceFile();

    while (1) {
      Token tok;
      PP.Lex(tok);
      if (tok.is(tok::eof))
        break;
    }
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
  CompilerInvocation Invocation;
  TrivialModuleLoader ModLoader;
  std::optional<HeaderSearch> HeaderInfo;
};

TEST_F(ModuleDeclStateTest, NamedModuleInterface) {
  const char *source = R"(
export module foo;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)0);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_TRUE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo");
}

TEST_F(ModuleDeclStateTest, NamedModuleImplementation) {
  const char *source = R"(
module foo;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)0);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_FALSE(PP->isInNamedInterfaceUnit());
  EXPECT_TRUE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo");
}

TEST_F(ModuleDeclStateTest, ModuleImplementationPartition) {
  const char *source = R"(
module foo:part;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)0);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_FALSE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo:part");
}

TEST_F(ModuleDeclStateTest, ModuleInterfacePartition) {
  const char *source = R"(
export module foo:part;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)0);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_TRUE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo:part");
}

TEST_F(ModuleDeclStateTest, ModuleNameWithDot) {
  const char *source = R"(
export module foo.dot:part.dot;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)0);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_TRUE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo.dot:part.dot");
}

TEST_F(ModuleDeclStateTest, NotModule) {
  const char *source = R"(
// export module foo:part;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)0);
  EXPECT_FALSE(PP->isInNamedModule());
  EXPECT_FALSE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
}

TEST_F(ModuleDeclStateTest, ModuleWithGMF) {
  const char *source = R"(
module;
#include "bar.h"
#include <zoo.h>
import "bar";
import <zoo>;
export module foo:part;
import "HU";
import M;
import :another;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {true, true};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)2);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_TRUE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo:part");
}

TEST_F(ModuleDeclStateTest, ModuleWithGMFWithClangNamedModule) {
  const char *source = R"(
module;
#include "bar.h"
#include <zoo.h>
import "bar";
import <zoo>;
export module foo:part;
import "HU";
import M;
import :another;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {true, true};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)2);
  EXPECT_TRUE(PP->isInNamedModule());
  EXPECT_TRUE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
  EXPECT_EQ(PP->getNamedModuleName(), "foo:part");
}

TEST_F(ModuleDeclStateTest, ImportsInNormalTU) {
  const char *source = R"(
#include "bar.h"
#include <zoo.h>
import "bar";
import <zoo>;
import "HU";
import M;
// We can't import a partition in non-module TU.
import :another;
  )";
  std::unique_ptr<Preprocessor> PP = getPreprocessor(source, "-std=c++20");

  std::initializer_list<bool> ImportKinds = {true};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)1);
  EXPECT_FALSE(PP->isInNamedModule());
  EXPECT_FALSE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
}

TEST_F(ModuleDeclStateTest, ImportAClangNamedModule) {
  const char *source = R"(
@import anything;
  )";
  std::unique_ptr<Preprocessor> PP =
      getPreprocessor(source, {"-fmodules", "-fimplicit-module-maps", "-x",
                               "objective-c++", "-std=c++20"});

  std::initializer_list<bool> ImportKinds = {false};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)1);
  EXPECT_FALSE(PP->isInNamedModule());
  EXPECT_FALSE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
}

TEST_F(ModuleDeclStateTest, ImportWixedForm) {
  const char *source = R"(
import "HU";
@import anything;
import M;
@import another;
import M2;
  )";
  std::unique_ptr<Preprocessor> PP =
      getPreprocessor(source, {"-fmodules", "-fimplicit-module-maps", "-x",
                               "objective-c++", "-std=c++20"});

  std::initializer_list<bool> ImportKinds = {false, true, false, true};
  preprocess(*PP,
             std::make_unique<CheckNamedModuleImportingCB>(*PP, ImportKinds));

  auto *Callback =
      static_cast<CheckNamedModuleImportingCB *>(PP->getPPCallbacks());
  EXPECT_EQ(Callback->importNamedModuleNum(), (size_t)4);
  EXPECT_FALSE(PP->isInNamedModule());
  EXPECT_FALSE(PP->isInNamedInterfaceUnit());
  EXPECT_FALSE(PP->isInImplementationUnit());
}

} // namespace
