//===--- clang-tidy/cir-tidy/CIRTidy.cpp - CIR tidy tool ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements a cir-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "CIRTidy.h"
#include "CIRASTConsumer.h"
#include "ClangTidyModuleRegistry.h"
#include "ClangTidyProfiling.h"
#include "clang-tidy-config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/DiagnosticsYaml.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "clang/Tooling/Tooling.h"

using namespace clang::tooling;
using namespace llvm;

namespace cir {
namespace tidy {

CIRTidyASTConsumerFactory::CIRTidyASTConsumerFactory(
    ClangTidyContext &Context,
    IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS)
    : Context(Context), OverlayFS(std::move(OverlayFS)) {}

std::unique_ptr<clang::ASTConsumer>
CIRTidyASTConsumerFactory::createASTConsumer(clang::CompilerInstance &Compiler,
                                             StringRef File) {
  // FIXME(clang-tidy): Move this to a separate method, so that
  // CreateASTConsumer doesn't modify Compiler.
  SourceManager *SM = &Compiler.getSourceManager();
  Context.setSourceManager(SM);
  Context.setCurrentFile(File);
  Context.setASTContext(&Compiler.getASTContext());

  auto WorkingDir = Compiler.getSourceManager()
                        .getFileManager()
                        .getVirtualFileSystem()
                        .getCurrentWorkingDirectory();
  if (WorkingDir)
    Context.setCurrentBuildDirectory(WorkingDir.get());
  return std::make_unique<CIRASTConsumer>(Compiler, File, Context);
}

std::vector<std::string> CIRTidyASTConsumerFactory::getCheckNames() {
  std::vector<std::string> CheckNames;
  for (const auto &CIRCheckName : this->CIRChecks) {
    if (Context.isCheckEnabled(CIRCheckName))
      CheckNames.emplace_back(CIRCheckName);
  }

  llvm::sort(CheckNames);
  return CheckNames;
}

void exportReplacements(const llvm::StringRef MainFilePath,
                        const std::vector<ClangTidyError> &Errors,
                        raw_ostream &OS) {
  TranslationUnitDiagnostics TUD;
  TUD.MainSourceFile = std::string(MainFilePath);
  for (const auto &Error : Errors) {
    tooling::Diagnostic Diag = Error;
    TUD.Diagnostics.insert(TUD.Diagnostics.end(), Diag);
  }

  yaml::Output YAML(OS);
  YAML << TUD;
}

std::vector<ClangTidyError>
runCIRTidy(ClangTidyContext &Context, const CompilationDatabase &Compilations,
           ArrayRef<std::string> InputFiles,
           llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> BaseFS,
           bool ApplyAnyFix, bool EnableCheckProfile,
           llvm::StringRef StoreCheckProfile) {
  ClangTool Tool(Compilations, InputFiles,
                 std::make_shared<PCHContainerOperations>(), BaseFS);

  Context.setEnableProfiling(EnableCheckProfile);
  Context.setProfileStoragePrefix(StoreCheckProfile);

  ClangTidyDiagnosticConsumer DiagConsumer(Context, nullptr, true, ApplyAnyFix);
  DiagnosticsEngine DE(new DiagnosticIDs(), new DiagnosticOptions(),
                       &DiagConsumer, /*ShouldOwnClient=*/false);
  Context.setDiagnosticsEngine(&DE);
  Tool.setDiagnosticConsumer(&DiagConsumer);

  class ActionFactory : public FrontendActionFactory {
  public:
    ActionFactory(ClangTidyContext &Context,
                  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> BaseFS)
        : ConsumerFactory(Context, std::move(BaseFS)) {}
    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<Action>(&ConsumerFactory);
    }

    bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                       FileManager *Files,
                       std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                       DiagnosticConsumer *DiagConsumer) override {
      // Explicitly ask to define __clang_analyzer__ macro.
      Invocation->getPreprocessorOpts().SetUpStaticAnalyzer = true;
      return FrontendActionFactory::runInvocation(
          Invocation, Files, PCHContainerOps, DiagConsumer);
    }

  private:
    class Action : public ASTFrontendAction {
    public:
      Action(CIRTidyASTConsumerFactory *Factory) : Factory(Factory) {}
      std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                     StringRef File) override {
        return Factory->createASTConsumer(Compiler, File);
      }

    private:
      CIRTidyASTConsumerFactory *Factory;
    };

    CIRTidyASTConsumerFactory ConsumerFactory;
  };

  ActionFactory Factory(Context, std::move(BaseFS));
  Tool.run(&Factory);
  return DiagConsumer.take();
}

} // namespace tidy
} // namespace cir

// Now that clang-tidy is integrated with the lifetime checker, CIR changes to
// ClangTidyForceLinker.h are forcing CIRModuleAnchorSource to also be available
// as part of cir-tidy. Since cir-tidy is going to be removed soon, add this so
// that it can still builds in the meantime.
namespace clang::tidy {

// This anchor is used to force the linker to link in the generated object file
// and thus register the CIRModule.
volatile int CIRModuleAnchorSource = 0;

} // namespace clang::tidy
