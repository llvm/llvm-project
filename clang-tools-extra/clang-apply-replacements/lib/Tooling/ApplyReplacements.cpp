//===-- ApplyReplacements.cpp - Apply and deduplicate replacements --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation for deduplicating, detecting
/// conflicts in, and applying collections of Replacements.
///
/// FIXME: Use Diagnostics for output instead of llvm::errs().
///
//===----------------------------------------------------------------------===//
#include "clang-apply-replacements/Tooling/ApplyReplacements.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace clang;


static void eatDiagnostics(const SMDiagnostic &, void *) {}

namespace clang {
namespace replace {

std::error_code
collectReplacementsFromDirectory(const llvm::StringRef Directory,
                                 TUReplacements &TUs,
                                 TUReplacementFiles & TURFiles,
                                 clang::DiagnosticsEngine &Diagnostics) {
  using namespace llvm::sys::fs;
  using namespace llvm::sys::path;

  std::error_code ErrorCode;

  for (recursive_directory_iterator I(Directory, ErrorCode), E;
       I != E && !ErrorCode; I.increment(ErrorCode)) {
    if (filename(I->path())[0] == '.') {
      // Indicate not to descend into directories beginning with '.'
      I.no_push();
      continue;
    }

    if (extension(I->path()) != ".yaml")
      continue;

    TURFiles.push_back(I->path());

    ErrorOr<std::unique_ptr<MemoryBuffer>> Out =
        MemoryBuffer::getFile(I->path());
    if (std::error_code BufferError = Out.getError()) {
      errs() << "Error reading " << I->path() << ": " << BufferError.message()
             << "\n";
      continue;
    }

    yaml::Input YIn(Out.get()->getBuffer(), nullptr, &eatDiagnostics);
    tooling::TranslationUnitReplacements TU;
    YIn >> TU;
    if (YIn.error()) {
      // File doesn't appear to be a header change description. Ignore it.
      continue;
    }

    // Only keep files that properly parse.
    TUs.push_back(TU);
  }

  return ErrorCode;
}

/// \brief Dumps information for a sequence of conflicting Replacements.
///
/// \param[in] File FileEntry for the file the conflicting Replacements are
/// for.
/// \param[in] ConflictingReplacements List of conflicting Replacements.
/// \param[in] SM SourceManager used for reporting.
static void reportConflict(
    const FileEntry *File,
    const llvm::ArrayRef<clang::tooling::Replacement> ConflictingReplacements,
    SourceManager &SM) {
  FileID FID = SM.translateFile(File);
  if (FID.isInvalid())
    FID = SM.createFileID(File, SourceLocation(), SrcMgr::C_User);

  // FIXME: Output something a little more user-friendly (e.g. unified diff?)
  errs() << "The following changes conflict:\n";
  for (const tooling::Replacement &R : ConflictingReplacements) {
    if (R.getLength() == 0) {
      errs() << "  Insert at " << SM.getLineNumber(FID, R.getOffset()) << ":"
             << SM.getColumnNumber(FID, R.getOffset()) << " "
             << R.getReplacementText() << "\n";
    } else {
      if (R.getReplacementText().empty())
        errs() << "  Remove ";
      else
        errs() << "  Replace ";

      errs() << SM.getLineNumber(FID, R.getOffset()) << ":"
             << SM.getColumnNumber(FID, R.getOffset()) << "-"
             << SM.getLineNumber(FID, R.getOffset() + R.getLength() - 1) << ":"
             << SM.getColumnNumber(FID, R.getOffset() + R.getLength() - 1);

      if (R.getReplacementText().empty())
        errs() << "\n";
      else
        errs() << " with \"" << R.getReplacementText() << "\"\n";
    }
  }
}

/// \brief Deduplicates and tests for conflicts among the replacements for each
/// file in \c Replacements. Any conflicts found are reported.
///
/// \post Replacements[i].getOffset() <= Replacements[i+1].getOffset().
///
/// \param[in,out] Replacements Container of all replacements grouped by file
/// to be deduplicated and checked for conflicts.
/// \param[in] SM SourceManager required for conflict reporting.
///
/// \returns \parblock
///          \li true if conflicts were detected
///          \li false if no conflicts were detected
static bool deduplicateAndDetectConflicts(FileToReplacementsMap &Replacements,
                                          SourceManager &SM) {
  bool conflictsFound = false;

  for (auto &FileAndReplacements : Replacements) {
    const FileEntry *Entry = FileAndReplacements.first;
    auto &Replacements = FileAndReplacements.second;
    assert(Entry != nullptr && "No file entry!");

    std::vector<tooling::Range> Conflicts;
    tooling::deduplicate(FileAndReplacements.second, Conflicts);

    if (Conflicts.empty())
      continue;

    conflictsFound = true;

    errs() << "There are conflicting changes to " << Entry->getName() << ":\n";

    for (const tooling::Range &Conflict : Conflicts) {
      auto ConflictingReplacements = llvm::makeArrayRef(
          &Replacements[Conflict.getOffset()], Conflict.getLength());
      reportConflict(Entry, ConflictingReplacements, SM);
    }
  }

  return conflictsFound;
}

bool mergeAndDeduplicate(const TUReplacements &TUs,
                         FileToReplacementsMap &GroupedReplacements,
                         clang::SourceManager &SM) {

  // Group all replacements by target file.
  std::set<StringRef> Warned;
  for (const auto &TU : TUs) {
    for (const tooling::Replacement &R : TU.Replacements) {
      // Use the file manager to deduplicate paths. FileEntries are
      // automatically canonicalized.
      const FileEntry *Entry = SM.getFileManager().getFile(R.getFilePath());
      if (!Entry && Warned.insert(R.getFilePath()).second) {
        errs() << "Described file '" << R.getFilePath()
               << "' doesn't exist. Ignoring...\n";
        continue;
      }
      GroupedReplacements[Entry].push_back(R);
    }
  }

  // Ask clang to deduplicate and report conflicts.
  return !deduplicateAndDetectConflicts(GroupedReplacements, SM);
}

bool applyReplacements(const FileToReplacementsMap &GroupedReplacements,
                       clang::Rewriter &Rewrites) {

  // Apply all changes
  //
  // FIXME: No longer certain GroupedReplacements is really the best kind of
  // data structure for applying replacements. Rewriter certainly doesn't care.
  // However, until we nail down the design of ReplacementGroups, might as well
  // leave this as is.
  for (const auto &FileAndReplacements : GroupedReplacements) {
    if (!tooling::applyAllReplacements(FileAndReplacements.second, Rewrites))
      return false;
  }

  return true;
}

RangeVector calculateChangedRanges(
    const std::vector<clang::tooling::Replacement> &Replaces) {
  RangeVector ChangedRanges;

  // Generate the new ranges from the replacements.
  int Shift = 0;
  for (const tooling::Replacement &R : Replaces) {
    unsigned Offset = R.getOffset() + Shift;
    unsigned Length = R.getReplacementText().size();
    Shift += Length - R.getLength();
    ChangedRanges.push_back(tooling::Range(Offset, Length));
  }

  return ChangedRanges;
}

bool writeFiles(const clang::Rewriter &Rewrites) {

  for (Rewriter::const_buffer_iterator BufferI = Rewrites.buffer_begin(),
                                       BufferE = Rewrites.buffer_end();
       BufferI != BufferE; ++BufferI) {
    const char *FileName =
        Rewrites.getSourceMgr().getFileEntryForID(BufferI->first)->getName();

    std::error_code EC;
    llvm::raw_fd_ostream FileStream(FileName, EC, llvm::sys::fs::F_Text);
    if (EC) {
      errs() << "Warning: Could not write to " << EC.message() << "\n";
      continue;
    }
    BufferI->second.write(FileStream);
  }

  return true;
}

bool deleteReplacementFiles(const TUReplacementFiles &Files,
                            clang::DiagnosticsEngine &Diagnostics) {
  bool Success = true;
  for (const auto &Filename : Files) {
    std::error_code Error = llvm::sys::fs::remove(Filename);
    if (Error) {
      Success = false;
      // FIXME: Use Diagnostics for outputting errors.
      errs() << "Error deleting file: " << Filename << "\n";
      errs() << Error.message() << "\n";
      errs() << "Please delete the file manually\n";
    }
  }
  return Success;
}

} // end namespace replace
} // end namespace clang
