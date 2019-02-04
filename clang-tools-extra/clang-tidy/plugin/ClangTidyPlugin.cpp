//===- ClangTidyPlugin.cpp - clang-tidy as a clang plugin -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyForceLinker.h"
#include "../ClangTidyModule.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"

namespace clang {
namespace tidy {

/// The core clang tidy plugin action. This just provides the AST consumer and
/// command line flag parsing for using clang-tidy as a clang plugin.
class ClangTidyPluginAction : public PluginASTAction {
  /// Wrapper to grant the context the same lifetime as the action. We use
  /// MultiplexConsumer to avoid writing out all the forwarding methods.
  class WrapConsumer : public MultiplexConsumer {
    std::unique_ptr<ClangTidyContext> Context;

  public:
    WrapConsumer(std::unique_ptr<ClangTidyContext> Context,
                 std::vector<std::unique_ptr<ASTConsumer>> Consumer)
        : MultiplexConsumer(std::move(Consumer)), Context(std::move(Context)) {}
  };

public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    // Insert the current diagnostics engine.
    Context->setDiagnosticsEngine(&Compiler.getDiagnostics());

    // Create the AST consumer.
    ClangTidyASTConsumerFactory Factory(*Context);
    std::vector<std::unique_ptr<ASTConsumer>> Vec;
    Vec.push_back(Factory.CreateASTConsumer(Compiler, File));

    return llvm::make_unique<WrapConsumer>(std::move(Context), std::move(Vec));
  }

  bool ParseArgs(const CompilerInstance &,
                 const std::vector<std::string> &Args) override {
    ClangTidyGlobalOptions GlobalOptions;
    ClangTidyOptions DefaultOptions;
    ClangTidyOptions OverrideOptions;

    // Parse the extra command line args.
    // FIXME: This is very limited at the moment.
    for (StringRef Arg : Args)
      if (Arg.startswith("-checks="))
        OverrideOptions.Checks = Arg.substr(strlen("-checks="));

    auto Options = llvm::make_unique<FileOptionsProvider>(
        GlobalOptions, DefaultOptions, OverrideOptions);
    Context = llvm::make_unique<ClangTidyContext>(std::move(Options));
    return true;
  }

private:
  std::unique_ptr<ClangTidyContext> Context;
};
} // namespace tidy
} // namespace clang

// This anchor is used to force the linker to link in the generated object file
// and thus register the clang-tidy plugin.
volatile int ClangTidyPluginAnchorSource = 0;

static clang::FrontendPluginRegistry::Add<clang::tidy::ClangTidyPluginAction>
    X("clang-tidy", "clang-tidy");
