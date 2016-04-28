//===--- tools/extra/clang-rename/ClangRename.cpp - Clang rename tool -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a clang-rename tool that automatically finds and
/// renames symbols in C++ code.
///
//===----------------------------------------------------------------------===//

#include "../USRFindingAction.h"
#include "../RenamingAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Host.h"
#include <cstdlib>
#include <string>

using namespace llvm;

cl::OptionCategory ClangRenameCategory("Clang-rename options");

static cl::opt<std::string>
NewName(
    "new-name",
    cl::desc("The new name to change the symbol to."),
    cl::cat(ClangRenameCategory));
static cl::opt<unsigned>
SymbolOffset(
    "offset",
    cl::desc("Locates the symbol by offset as opposed to <line>:<column>."),
    cl::cat(ClangRenameCategory));
static cl::opt<bool>
Inplace(
    "i",
    cl::desc("Overwrite edited <file>s."),
    cl::cat(ClangRenameCategory));
static cl::opt<bool>
PrintName(
    "pn",
    cl::desc("Print the found symbol's name prior to renaming to stderr."),
    cl::cat(ClangRenameCategory));
static cl::opt<bool>
PrintLocations(
    "pl",
    cl::desc("Print the locations affected by renaming to stderr."),
    cl::cat(ClangRenameCategory));

#define CLANG_RENAME_VERSION "0.0.1"

static void PrintVersion() {
  outs() << "clang-rename version " << CLANG_RENAME_VERSION << "\n";
}

using namespace clang;

const char RenameUsage[] = "A tool to rename symbols in C/C++ code.\n\
clang-rename renames every occurrence of a symbol found at <offset> in\n\
<source0>. If -i is specified, the edited files are overwritten to disk.\n\
Otherwise, the results are written to stdout.\n";

int main(int argc, const char **argv) {
  cl::SetVersionPrinter(PrintVersion);
  tooling::CommonOptionsParser OP(argc, argv, ClangRenameCategory, RenameUsage);

  // Check the arguments for correctness.

  if (NewName.empty()) {
    errs() << "clang-rename: no new name provided.\n\n";
    cl::PrintHelpMessage();
    exit(1);
  }

  // Get the USRs.
  auto Files = OP.getSourcePathList();
  tooling::RefactoringTool Tool(OP.getCompilations(), Files);
  rename::USRFindingAction USRAction(SymbolOffset);

  // Find the USRs.
  Tool.run(tooling::newFrontendActionFactory(&USRAction).get());
  const auto &USRs = USRAction.getUSRs();
  const auto &PrevName = USRAction.getUSRSpelling();

  if (PrevName.empty())
    // An error should have already been printed.
    exit(1);

  if (PrintName)
    errs() << "clang-rename: found name: " << PrevName << "\n";

  // Perform the renaming.
  rename::RenamingAction RenameAction(NewName, PrevName, USRs,
                                      Tool.getReplacements(), PrintLocations);
  auto Factory = tooling::newFrontendActionFactory(&RenameAction);
  int res;

  if (Inplace) {
    res = Tool.runAndSave(Factory.get());
  } else {
    res = Tool.run(Factory.get());

    // Write every file to stdout. Right now we just barf the files without any
    // indication of which files start where, other than that we print the files
    // in the same order we see them.
    LangOptions DefaultLangOptions;
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
        new DiagnosticOptions();
    TextDiagnosticPrinter DiagnosticPrinter(errs(), &*DiagOpts);
    DiagnosticsEngine Diagnostics(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
        &*DiagOpts, &DiagnosticPrinter, false);
    auto &FileMgr = Tool.getFiles();
    SourceManager Sources(Diagnostics, FileMgr);
    Rewriter Rewrite(Sources, DefaultLangOptions);

    Tool.applyAllReplacements(Rewrite);
    for (const auto &File : Files) {
      const auto *Entry = FileMgr.getFile(File);
      auto ID = Sources.translateFile(Entry);
      Rewrite.getEditBuffer(ID).write(outs());
    }
  }

  exit(res);
}
