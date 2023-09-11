//===-- ClangApplyReplacementsMain.cpp - Main file for the tool -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the main function for the
/// clang-apply-replacements tool.
///
//===----------------------------------------------------------------------===//

#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/Format/Format.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace clang;
using namespace clang::replace;

static cl::opt<std::string> Directory(cl::Positional, cl::Required,
                                      cl::desc("<Search Root Directory>"));

static cl::OptionCategory ReplacementCategory("Replacement Options");
static cl::OptionCategory FormattingCategory("Formatting Options");

const cl::OptionCategory *VisibleCategories[] = {&ReplacementCategory,
                                                 &FormattingCategory};

static cl::opt<bool> RemoveTUReplacementFiles(
    "remove-change-desc-files",
    cl::desc("Remove the change description files regardless of successful\n"
             "merging/replacing."),
    cl::init(false), cl::cat(ReplacementCategory));

static cl::opt<bool> IgnoreInsertConflict(
    "ignore-insert-conflict",
    cl::desc("Ignore insert conflict and keep running to fix."),
    cl::init(false), cl::cat(ReplacementCategory));

static cl::opt<bool> DoFormat(
    "format",
    cl::desc("Enable formatting of code changed by applying replacements.\n"
             "Use -style to choose formatting style.\n"),
    cl::cat(FormattingCategory));

// FIXME: Consider making the default behaviour for finding a style
// configuration file to start the search anew for every file being changed to
// handle situations where the style is different for different parts of a
// project.

static cl::opt<std::string> FormatStyleConfig(
    "style-config",
    cl::desc("Path to a directory containing a .clang-format file\n"
             "describing a formatting style to use for formatting\n"
             "code when -style=file.\n"),
    cl::init(""), cl::cat(FormattingCategory));

static cl::opt<std::string>
    FormatStyleOpt("style", cl::desc(format::StyleOptionHelpDescription),
                   cl::init("LLVM"), cl::cat(FormattingCategory));

namespace {
// Helper object to remove the TUReplacement and TUDiagnostic (triggered by
// "remove-change-desc-files" command line option) when exiting current scope.
class ScopedFileRemover {
public:
  ScopedFileRemover(const TUReplacementFiles &Files,
                    clang::DiagnosticsEngine &Diagnostics)
      : TURFiles(Files), Diag(Diagnostics) {}

  ~ScopedFileRemover() { deleteReplacementFiles(TURFiles, Diag); }

private:
  const TUReplacementFiles &TURFiles;
  clang::DiagnosticsEngine &Diag;
};
} // namespace

static void printVersion(raw_ostream &OS) {
  OS << "clang-apply-replacements version " CLANG_VERSION_STRING << "\n";
}

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(ArrayRef(VisibleCategories));

  cl::SetVersionPrinter(printVersion);
  cl::ParseCommandLineOptions(argc, argv);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), DiagOpts.get());

  // Determine a formatting style from options.
  auto FormatStyleOrError = format::getStyle(FormatStyleOpt, FormatStyleConfig,
                                             format::DefaultFallbackStyle);
  if (!FormatStyleOrError) {
    llvm::errs() << llvm::toString(FormatStyleOrError.takeError()) << "\n";
    return 1;
  }
  format::FormatStyle FormatStyle = std::move(*FormatStyleOrError);

  TUReplacements TURs;
  TUReplacementFiles TUFiles;

  std::error_code ErrorCode =
      collectReplacementsFromDirectory(Directory, TURs, TUFiles, Diagnostics);

  TUDiagnostics TUDs;
  TUFiles.clear();
  ErrorCode =
      collectReplacementsFromDirectory(Directory, TUDs, TUFiles, Diagnostics);

  if (ErrorCode) {
    errs() << "Trouble iterating over directory '" << Directory
           << "': " << ErrorCode.message() << "\n";
    return 1;
  }

  // Remove the TUReplacementFiles (triggered by "remove-change-desc-files"
  // command line option) when exiting main().
  std::unique_ptr<ScopedFileRemover> Remover;
  if (RemoveTUReplacementFiles)
    Remover.reset(new ScopedFileRemover(TUFiles, Diagnostics));

  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  FileToChangesMap Changes;
  if (!mergeAndDeduplicate(TURs, TUDs, Changes, SM, IgnoreInsertConflict))
    return 1;

  tooling::ApplyChangesSpec Spec;
  Spec.Cleanup = true;
  Spec.Style = FormatStyle;
  Spec.Format = DoFormat ? tooling::ApplyChangesSpec::kAll
                         : tooling::ApplyChangesSpec::kNone;

  for (const auto &FileChange : Changes) {
    FileEntryRef Entry = FileChange.first;
    StringRef FileName = Entry.getName();
    llvm::Expected<std::string> NewFileData =
        applyChanges(FileName, FileChange.second, Spec, Diagnostics);
    if (!NewFileData) {
      errs() << llvm::toString(NewFileData.takeError()) << "\n";
      continue;
    }

    // Write new file to disk
    std::error_code EC;
    llvm::raw_fd_ostream FileStream(FileName, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Could not open " << FileName << " for writing\n";
      continue;
    }
    FileStream << *NewFileData;
  }

  return 0;
}
