//===-- ApplyReplacements.cpp - Apply and deduplicate replacements --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the implementation for deduplicating, detecting
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
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/DiagnosticsYaml.h"
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

std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TUReplacements &TUs,
    TUReplacementFiles &TUFiles, clang::DiagnosticsEngine &Diagnostics) {
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

    TUFiles.push_back(I->path());

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

std::error_code collectReplacementsFromDirectory(
    const llvm::StringRef Directory, TUDiagnostics &TUs,
    TUReplacementFiles &TUFiles, clang::DiagnosticsEngine &Diagnostics) {
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

    TUFiles.push_back(I->path());

    ErrorOr<std::unique_ptr<MemoryBuffer>> Out =
        MemoryBuffer::getFile(I->path());
    if (std::error_code BufferError = Out.getError()) {
      errs() << "Error reading " << I->path() << ": " << BufferError.message()
             << "\n";
      continue;
    }

    yaml::Input YIn(Out.get()->getBuffer(), nullptr, &eatDiagnostics);
    tooling::TranslationUnitDiagnostics TU;
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

/// Extract replacements from collected TranslationUnitReplacements and
/// TranslationUnitDiagnostics and group them per file. Identical replacements
/// from diagnostics are deduplicated.
///
/// \param[in] TUs Collection of all found and deserialized
/// TranslationUnitReplacements.
/// \param[in] TUDs Collection of all found and deserialized
/// TranslationUnitDiagnostics.
/// \param[in] SM Used to deduplicate paths.
///
/// \returns A map mapping FileEntry to a set of Replacement targeting that
/// file.
static llvm::DenseMap<const FileEntry *, std::vector<tooling::Replacement>>
groupReplacements(const TUReplacements &TUs, const TUDiagnostics &TUDs,
                  const clang::SourceManager &SM) {
  std::set<StringRef> Warned;
  llvm::DenseMap<const FileEntry *, std::vector<tooling::Replacement>>
      GroupedReplacements;

  // Deduplicate identical replacements in diagnostics unless they are from the
  // same TU.
  // FIXME: Find an efficient way to deduplicate on diagnostics level.
  llvm::DenseMap<const FileEntry *,
                 std::map<tooling::Replacement,
                          const tooling::TranslationUnitDiagnostics *>>
      DiagReplacements;

  auto AddToGroup = [&](const tooling::Replacement &R,
                        const tooling::TranslationUnitDiagnostics *SourceTU) {
    // Use the file manager to deduplicate paths. FileEntries are
    // automatically canonicalized.
    if (auto Entry = SM.getFileManager().getFile(R.getFilePath())) {
      if (SourceTU) {
        auto &Replaces = DiagReplacements[*Entry];
        auto It = Replaces.find(R);
        if (It == Replaces.end())
          Replaces.emplace(R, SourceTU);
        else if (It->second != SourceTU)
          // This replacement is a duplicate of one suggested by another TU.
          return;
      }
      GroupedReplacements[*Entry].push_back(R);
    } else if (Warned.insert(R.getFilePath()).second) {
      errs() << "Described file '" << R.getFilePath()
             << "' doesn't exist. Ignoring...\n";
    }
  };

  for (const auto &TU : TUs)
    for (const tooling::Replacement &R : TU.Replacements)
      AddToGroup(R, nullptr);

  for (const auto &TU : TUDs)
    for (const auto &D : TU.Diagnostics)
      if (const auto *ChoosenFix = tooling::selectFirstFix(D)) {
        for (const auto &Fix : *ChoosenFix)
          for (const tooling::Replacement &R : Fix.second)
            AddToGroup(R, &TU);
      }

  // Sort replacements per file to keep consistent behavior when
  // clang-apply-replacements run on differents machine.
  for (auto &FileAndReplacements : GroupedReplacements) {
    llvm::sort(FileAndReplacements.second.begin(),
               FileAndReplacements.second.end());
  }

  return GroupedReplacements;
}

bool mergeAndDeduplicate(const TUReplacements &TUs, const TUDiagnostics &TUDs,
                         FileToChangesMap &FileChanges,
                         clang::SourceManager &SM) {
  auto GroupedReplacements = groupReplacements(TUs, TUDs, SM);
  bool ConflictDetected = false;

  // To report conflicting replacements on corresponding file, all replacements
  // are stored into 1 big AtomicChange.
  for (const auto &FileAndReplacements : GroupedReplacements) {
    const FileEntry *Entry = FileAndReplacements.first;
    const SourceLocation BeginLoc =
        SM.getLocForStartOfFile(SM.getOrCreateFileID(Entry, SrcMgr::C_User));
    tooling::AtomicChange FileChange(Entry->getName(), Entry->getName());
    for (const auto &R : FileAndReplacements.second) {
      llvm::Error Err =
          FileChange.replace(SM, BeginLoc.getLocWithOffset(R.getOffset()),
                             R.getLength(), R.getReplacementText());
      if (Err) {
        // FIXME: This will report conflicts by pair using a file+offset format
        // which is not so much human readable.
        // A first improvement could be to translate offset to line+col. For
        // this and without loosing error message some modifications around
        // `tooling::ReplacementError` are need (access to
        // `getReplacementErrString`).
        // A better strategy could be to add a pretty printer methods for
        // conflict reporting. Methods that could be parameterized to report a
        // conflict in different format, file+offset, file+line+col, or even
        // more human readable using VCS conflict markers.
        // For now, printing directly the error reported by `AtomicChange` is
        // the easiest solution.
        errs() << llvm::toString(std::move(Err)) << "\n";
        ConflictDetected = true;
      }
    }
    FileChanges.try_emplace(Entry,
                            std::vector<tooling::AtomicChange>{FileChange});
  }

  return !ConflictDetected;
}

llvm::Expected<std::string>
applyChanges(StringRef File, const std::vector<tooling::AtomicChange> &Changes,
             const tooling::ApplyChangesSpec &Spec,
             DiagnosticsEngine &Diagnostics) {
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      SM.getFileManager().getBufferForFile(File);
  if (!Buffer)
    return errorCodeToError(Buffer.getError());
  return tooling::applyAtomicChanges(File, Buffer.get()->getBuffer(), Changes,
                                     Spec);
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
