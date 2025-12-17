//===--- StandaloneDiagnostic.h - Serializable Diagnostic -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A serializable diagnostic representation to retain diagnostics after their
// SourceManager has been destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_STANDALONEDIAGNOSTICS_H
#define LLVM_CLANG_FRONTEND_STANDALONEDIAGNOSTICS_H

#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/StringExtras.h"
#include <cassert>
#include <string>
#include <vector>

namespace clang {

/// Represents a StoredDiagnostic in a form that can be retained until after its
/// SourceManager has been destroyed.
///
/// Source locations are stored as a combination of filename and offsets into
/// that file.
/// To report the diagnostic, it must first be translated back into a
/// StoredDiagnostic with a new associated SourceManager.
struct StandaloneDiagnostic {
  /// Represents a CharSourceRange within a StandaloneDiagnostic.
  struct SourceOffsetRange {
    SourceOffsetRange(CharSourceRange Range, const SourceManager &SrcMgr,
                      const LangOptions &LangOpts);

    unsigned Begin = 0;
    unsigned End = 0;
  };

  /// Represents a FixItHint within a StandaloneDiagnostic.
  struct StandaloneFixIt {
    StandaloneFixIt(const SourceManager &SrcMgr, const LangOptions &LangOpts,
                    const FixItHint &FixIt);

    SourceOffsetRange RemoveRange;
    SourceOffsetRange InsertFromRange;
    std::string CodeToInsert;
    bool BeforePreviousInsertions;
  };

  StandaloneDiagnostic(const LangOptions &LangOpts,
                       const StoredDiagnostic &InDiag);

  DiagnosticsEngine::Level Level;
  SrcMgr::CharacteristicKind FileKind;
  unsigned ID = 0;
  unsigned FileOffset = 0;
  std::string Message;
  std::string Filename;
  std::vector<SourceOffsetRange> Ranges;
  std::vector<StandaloneFixIt> FixIts;
};

/// Translates \c StandaloneDiag into a StoredDiagnostic, associating it with
/// the provided FileManager and SourceManager.
///
/// This allows the diagnostic to be emitted using the diagnostics engine, since
/// StandaloneDiagnostics themselfs cannot be emitted directly.
StoredDiagnostic
translateStandaloneDiag(FileManager &FileMgr, SourceManager &SrcMgr,
                        StandaloneDiagnostic StandaloneDiag,
                        llvm::StringMap<SourceLocation> &SrcLocCache);

} // namespace clang

#endif // STANDALONEDIAGNOSTICS
