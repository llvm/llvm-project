//===--- SemaAPINotesInternal.h - API Notes Sema Internals ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_SEMA_SEMAAPINOTESINTERNAL_H
#define LLVM_CLANG_LIB_SEMA_SEMAAPINOTESINTERNAL_H

#include "clang/APINotes/Types.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>

namespace clang {
class Sema;
struct APINotesParameterSelectorCandidates;
namespace api_notes {
class APINotesReader;
}

/// One stored exact Where.Parameters selector tracked for diagnostics.
struct APINotesSelectorDiagnosticEntry {
  api_notes::APINotesFunctionSelectorKey BroadKey;
  llvm::SmallVector<std::string, 4> Parameters;
  bool Used = false;
};

/// Source name and location for a declaration seen by Sema.
struct APINotesSelectorDiagnosticName {
  SourceLocation Loc;
  std::string Name;
};

/// Tracks exact Where.Parameters selectors from one API notes reader.
///
/// Sema marks selectors as used when a visible declaration matches them. It
/// also records broad/name-only declarations seen in the translation unit, so
/// end-of-TU diagnostics can warn about exact selectors for known names that
/// were never matched.
struct APINotesSelectorDiagnosticReaderState {
  /// Maps exact selector keys to entries in Selectors.
  llvm::DenseMap<api_notes::APINotesFunctionSelectorKey, unsigned>
      SelectorIndices;

  /// Maps broad/name-only keys to a declaration location/name used for
  /// diagnostics.
  llvm::DenseMap<api_notes::APINotesFunctionSelectorKey,
                 APINotesSelectorDiagnosticName>
      SeenNames;

  llvm::SmallVector<APINotesSelectorDiagnosticEntry, 4> Selectors;

  void addSelector(api_notes::APINotesFunctionSelector Selector) {
    unsigned Index = Selectors.size();
    APINotesSelectorDiagnosticEntry Entry;
    Entry.BroadKey = Selector.Key.getWithoutParameterSelector();
    Entry.Parameters = std::move(Selector.Parameters);
    Selectors.push_back(std::move(Entry));
    SelectorIndices.insert({Selector.Key, Index});
  }

  void addSelectors(llvm::SmallVectorImpl<api_notes::APINotesFunctionSelector>
                        &NewSelectors) {
    Selectors.reserve(NewSelectors.size());
    SelectorIndices.reserve(NewSelectors.size());
    SeenNames.reserve(NewSelectors.size());
    for (auto &Selector : NewSelectors)
      addSelector(std::move(Selector));
  }

  void noteSeenDeclaration(const api_notes::APINotesFunctionSelectorKey &Key,
                           llvm::StringRef Name, SourceLocation Loc) {
    SeenNames.insert({Key.getWithoutParameterSelector(), {Loc, Name.str()}});
  }

  void markUsed(const api_notes::APINotesFunctionSelectorKey &Key) {
    auto KnownSelector = SelectorIndices.find(Key);
    if (KnownSelector != SelectorIndices.end())
      Selectors[KnownSelector->second].Used = true;
  }

  void markCandidatesUsed(
      llvm::function_ref<std::optional<api_notes::APINotesFunctionSelectorKey>(
          llvm::ArrayRef<std::string>)>
          GetSelectorKey,
      const APINotesParameterSelectorCandidates &Candidates);

  void diagnoseUnused(Sema &S) const;
};

/// Selector diagnostic state for all API notes readers used by one Sema.
struct APINotesSelectorDiagnosticState {
  llvm::DenseMap<api_notes::APINotesReader *,
                 APINotesSelectorDiagnosticReaderState>
      Readers;

  APINotesSelectorDiagnosticReaderState &
  getOrCreateReaderState(api_notes::APINotesReader &Reader);

  void diagnoseUnused(Sema &S) const;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_SEMA_SEMAAPINOTESINTERNAL_H
