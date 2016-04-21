//===-- ClangIncludeFixer.cpp - Standalone include fixer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryXrefsDB.h"
#include "IncludeFixer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
using namespace clang;

static llvm::cl::OptionCategory tool_options("Tool options");

int main(int argc, char **argv) {
  clang::tooling::CommonOptionsParser options(argc, (const char **)argv,
                                              tool_options);
  clang::tooling::ClangTool tool(options.getCompilations(),
                                 options.getSourcePathList());
  // Set up the data source.
  std::map<std::string, std::vector<std::string>> XrefsMap = {
      {"std::string", {"<string>"}}};
  auto XrefsDB =
      llvm::make_unique<include_fixer::InMemoryXrefsDB>(std::move(XrefsMap));

  // Now run our tool.
  std::vector<clang::tooling::Replacement> Replacements;
  include_fixer::IncludeFixerActionFactory Factory(*XrefsDB, Replacements);

  tool.run(&Factory); // Always succeeds.

  // Set up a new source manager for applying the resulting replacements.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new clang::DiagnosticOptions);
  clang::DiagnosticsEngine Diagnostics(new clang::DiagnosticIDs, &*DiagOpts);
  clang::TextDiagnosticPrinter DiagnosticPrinter(llvm::outs(), &*DiagOpts);
  clang::SourceManager source_manager(Diagnostics, tool.getFiles());
  Diagnostics.setClient(&DiagnosticPrinter, false);

  // Write replacements to disk.
  clang::Rewriter Rewrites(source_manager, clang::LangOptions());
  clang::tooling::applyAllReplacements(Replacements, Rewrites);
  return Rewrites.overwriteChangedFiles();
}
