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
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdlib>
#include <string>
#include <system_error>

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
static cl::opt<std::string>
OldName(
    "old-name",
    cl::desc("The fully qualified name of the symbol, if -offset is not used."),
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
static cl::opt<std::string>
ExportFixes(
    "export-fixes",
    cl::desc("YAML file to store suggested fixes in."),
    cl::value_desc("filename"),
    cl::cat(ClangRenameCategory));

using namespace clang;

const char RenameUsage[] = "A tool to rename symbols in C/C++ code.\n\
clang-rename renames every occurrence of a symbol found at <offset> in\n\
<source0>. If -i is specified, the edited files are overwritten to disk.\n\
Otherwise, the results are written to stdout.\n";

int main(int argc, const char **argv) {
  tooling::CommonOptionsParser OP(argc, argv, ClangRenameCategory, RenameUsage);

  // Check the arguments for correctness.

  if (NewName.empty()) {
    errs() << "ERROR: no new name provided.\n\n";
    exit(1);
  }

  // Check if NewName is a valid identifier in C++17.
  LangOptions Options;
  Options.CPlusPlus = true;
  Options.CPlusPlus1z = true;
  IdentifierTable Table(Options);
  auto NewNameTokKind = Table.get(NewName).getTokenID();
  if (!tok::isAnyIdentifier(NewNameTokKind)) {
    errs() << "ERROR: new name is not a valid identifier in C++17.\n\n";
    exit(1);
  }

  // Get the USRs.
  auto Files = OP.getSourcePathList();
  tooling::RefactoringTool Tool(OP.getCompilations(), Files);
  rename::USRFindingAction USRAction(SymbolOffset, OldName);

  // Find the USRs.
  Tool.run(tooling::newFrontendActionFactory(&USRAction).get());
  const auto &USRs = USRAction.getUSRs();
  const auto &PrevName = USRAction.getUSRSpelling();

  if (PrevName.empty()) {
    // An error should have already been printed.
    exit(1);
  }

  if (PrintName) {
    errs() << "clang-rename: found name: " << PrevName << '\n';
  }

  // Perform the renaming.
  rename::RenamingAction RenameAction(NewName, PrevName, USRs,
                                      Tool.getReplacements(), PrintLocations);
  auto Factory = tooling::newFrontendActionFactory(&RenameAction);
  int ExitCode;

  if (Inplace) {
    ExitCode = Tool.runAndSave(Factory.get());
  } else {
    ExitCode = Tool.run(Factory.get());

    if (!ExportFixes.empty()) {
      std::error_code EC;
      llvm::raw_fd_ostream OS(ExportFixes, EC, llvm::sys::fs::F_None);
      if (EC) {
        llvm::errs() << "Error opening output file: " << EC.message() << '\n';
        exit(1);
      }

      // Export replacements.
      tooling::TranslationUnitReplacements TUR;
      const auto &FileToReplacements = Tool.getReplacements();
      for (const auto &Entry : FileToReplacements)
        TUR.Replacements.insert(TUR.Replacements.end(), Entry.second.begin(),
                                Entry.second.end());

      yaml::Output YAML(OS);
      YAML << TUR;
      OS.close();
      exit(0);
    }

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

  exit(ExitCode);
}
