//===--- ClangTidyTest.h - clang-tidy ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANGTIDYTEST_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANGTIDYTEST_H

#include "ClangTidy.h"
#include "ClangTidyCheck.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"
#include <map>
#include <memory>

namespace clang {
namespace tidy {
namespace test {

template <typename Check, typename... Checks> struct CheckFactory {
  static void
  createChecks(ClangTidyContext *Context,
               SmallVectorImpl<std::unique_ptr<ClangTidyCheck>> &Result) {
    CheckFactory<Check>::createChecks(Context, Result);
    CheckFactory<Checks...>::createChecks(Context, Result);
  }
};

template <typename Check> struct CheckFactory<Check> {
  static void
  createChecks(ClangTidyContext *Context,
               SmallVectorImpl<std::unique_ptr<ClangTidyCheck>> &Result) {
    Result.emplace_back(std::make_unique<Check>(
        "test-check-" + std::to_string(Result.size()), Context));
  }
};

template <typename... CheckTypes>
class TestClangTidyAction : public ASTFrontendAction {
public:
  TestClangTidyAction(SmallVectorImpl<std::unique_ptr<ClangTidyCheck>> &Checks,
                      ast_matchers::MatchFinder &Finder,
                      ClangTidyContext &Context)
      : Checks(Checks), Finder(Finder), Context(Context) {}

private:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    Context.setSourceManager(&Compiler.getSourceManager());
    Context.setCurrentFile(File);
    Context.setASTContext(&Compiler.getASTContext());

    Preprocessor *PP = &Compiler.getPreprocessor();

    // Checks must be created here, _after_ `Context` has been initialized, so
    // that check constructors can access the context (for example, through
    // `getLangOpts()`).
    CheckFactory<CheckTypes...>::createChecks(&Context, Checks);
    assert(!Checks.empty() && "No checks created");
    for (auto &Check : Checks) {
      assert(Check.get() && "Checks can't be null");
      if (!Check->isLanguageVersionSupported(Context.getLangOpts()))
        continue;
      Check->registerMatchers(&Finder);
      Check->registerPPCallbacks(Compiler.getSourceManager(), PP, PP);
    }
    return Finder.newASTConsumer();
  }

  SmallVectorImpl<std::unique_ptr<ClangTidyCheck>> &Checks;
  ast_matchers::MatchFinder &Finder;
  ClangTidyContext &Context;
};

template <typename... CheckTypes>
std::string
runCheckOnCode(StringRef Code, std::vector<ClangTidyError> *Errors = nullptr,
               const Twine &Filename = "input.cc",
               ArrayRef<std::string> ExtraArgs = {},
               const ClangTidyOptions &ExtraOptions = ClangTidyOptions(),
               std::map<StringRef, StringRef> PathsToContent =
                   std::map<StringRef, StringRef>()) {
  static_assert(sizeof...(CheckTypes) > 0, "No checks specified");
  ClangTidyOptions Options = ExtraOptions;
  Options.Checks = "*";
  ClangTidyContext Context(std::make_unique<DefaultOptionsProvider>(
      ClangTidyGlobalOptions(), Options));
  ClangTidyDiagnosticConsumer DiagConsumer(Context);
  auto DiagOpts = std::make_unique<DiagnosticOptions>();
  DiagnosticsEngine DE(DiagnosticIDs::create(), *DiagOpts, &DiagConsumer,
                       false);
  Context.setDiagnosticsEngine(std::move(DiagOpts), &DE);

  std::vector<std::string> Args(1, "clang-tidy");
  Args.push_back("-fsyntax-only");
  Args.push_back("-fno-delayed-template-parsing");
  std::string extension(
      std::string(llvm::sys::path::extension(Filename.str())));
  if (extension == ".m" || extension == ".mm") {
    Args.push_back("-fobjc-abi-version=2");
    Args.push_back("-fobjc-arc");
  }
  if (extension == ".cc" || extension == ".cpp" || extension == ".mm") {
    Args.push_back("-std=c++20");
  }
  Args.push_back("-Iinclude");
  Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
  Args.push_back(Filename.str());

  ast_matchers::MatchFinder Finder;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), InMemoryFileSystem));

  SmallVector<std::unique_ptr<ClangTidyCheck>, sizeof...(CheckTypes)> Checks;
  tooling::ToolInvocation Invocation(
      Args,
      std::make_unique<TestClangTidyAction<CheckTypes...>>(Checks, Finder,
                                                           Context),
      Files.get());
  InMemoryFileSystem->addFile(Filename, 0,
                              llvm::MemoryBuffer::getMemBuffer(Code));
  for (const auto &FileContent : PathsToContent) {
    InMemoryFileSystem->addFile(
        Twine("include/") + FileContent.first, 0,
        llvm::MemoryBuffer::getMemBuffer(FileContent.second));
  }
  Invocation.setDiagnosticConsumer(&DiagConsumer);
  if (!Invocation.run()) {
    std::string ErrorText;
    for (const auto &Error : DiagConsumer.take()) {
      ErrorText += Error.Message.Message + "\n";
    }
    llvm::report_fatal_error(llvm::Twine(ErrorText));
  }

  tooling::Replacements Fixes;
  std::vector<ClangTidyError> Diags = DiagConsumer.take();
  for (const ClangTidyError &Error : Diags) {
    if (const auto *ChosenFix = tooling::selectFirstFix(Error))
      for (const auto &FileAndFixes : *ChosenFix) {
        for (const auto &Fix : FileAndFixes.second) {
          auto Err = Fixes.add(Fix);
          // FIXME: better error handling. Keep the behavior for now.
          if (Err) {
            llvm::errs() << llvm::toString(std::move(Err)) << "\n";
            return "";
          }
        }
      }
  }
  if (Errors)
    *Errors = std::move(Diags);
  auto Result = tooling::applyAllReplacements(Code, Fixes);
  if (!Result) {
    // FIXME: propagate the error.
    llvm::consumeError(Result.takeError());
    return "";
  }
  return *Result;
}

#define EXPECT_NO_CHANGES(Check, Code)                                         \
  EXPECT_EQ(Code, runCheckOnCode<Check>(Code))

} // namespace test
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_TIDY_CLANGTIDYTEST_H
